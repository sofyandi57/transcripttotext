"""mcp_server.py -- MCP server yang membungkus fitur YouTube Transcript AI
Powerhouse (transcript_service.py, ai_service.py, audio_service.py,
history_store.py) supaya bisa dipakai langsung dari Claude Desktop/Code
lewat percakapan biasa, tanpa buka Streamlit sama sekali.

Modul service (transcript_service, ai_service, audio_service, history_store)
sengaja TIDAK bergantung pada Streamlit sama sekali -- semua kredensial lewat
parameter fungsi -- jadi bisa dipanggil langsung dari sini tanpa perubahan
apa pun ke kode aslinya.

Kredensial diambil dari environment variable (diisi lewat "env" di config
MCP client, mis. claude_desktop_config.json), BUKAN st.secrets (itu API
khusus Streamlit, tidak tersedia di luar app Streamlit).

Jalankan manual untuk tes: `python mcp_server.py`
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

import ai_service as ai
import history_store as hs
from audio_service import AudioFetchError, transcribe_audio
from transcript_service import (
    BlockedError,
    NoTranscriptError,
    TranscriptFetchError,
    TranscriptResult,
    VideoIdError,
    VideoNotFoundError,
    build_proxy_config,
    extract_video_id,
    format_transcript_with_timestamps,
    get_transcript,
    get_video_metadata,
    list_available_languages,
)

mcp = FastMCP("youtube-transcript-ai")

# ---------------------------------------------------------------------------
# Kredensial -- semua dari environment variable, diisi lewat config MCP
# client kamu (lihat README.md bagian "MCP Server (Claude Desktop)").
# ---------------------------------------------------------------------------

_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
_YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
_WEBSHARE_USERNAME = os.environ.get("WEBSHARE_USERNAME", "")
_WEBSHARE_PASSWORD = os.environ.get("WEBSHARE_PASSWORD", "")
_PROXY_HOST = os.environ.get("PROXY_HOST", "")
_PROXY_PORT = os.environ.get("PROXY_PORT", "")

_proxy_config = build_proxy_config(_WEBSHARE_USERNAME, _WEBSHARE_PASSWORD, _PROXY_HOST, _PROXY_PORT)

# Cache in-memory per proses server -- supaya tool lanjutan (ringkasan, FAQ,
# Q&A, dst) tidak perlu ambil ulang transcript-nya kalau sudah pernah
# fetch_transcript() dalam sesi yang sama. Riwayat permanen tetap lewat
# history_store.py (JSON lokal) seperti di app Streamlit.
_transcript_cache: dict[str, TranscriptResult] = {}


def _require(video_id: str) -> TranscriptResult:
    if video_id in _transcript_cache:
        return _transcript_cache[video_id]
    entry = hs.get_entry(video_id)
    if entry and entry.full_text:
        result = TranscriptResult(
            video_id=video_id,
            language=entry.language,
            language_code="",
            is_generated=False,
            full_text=entry.full_text,
            segments=entry.segments,
        )
        _transcript_cache[video_id] = result
        return result
    raise ValueError(
        f"Video {video_id} belum diambil transcript-nya di sesi ini. "
        "Panggil fetch_transcript dulu."
    )


@mcp.tool()
def fetch_transcript(url_or_id: str, languages: list[str] | None = None) -> dict:
    """Ambil transcript video YouTube dari caption yang tersedia.

    Parameters
    ----------
    url_or_id: URL lengkap atau video ID YouTube.
    languages: Urutan preferensi bahasa (kode ISO, mis. ["id", "en"]).
        Default ["id", "en"] kalau tidak diisi.

    Kalau video tidak punya caption sama sekali, gunakan tool
    `transcribe_with_whisper` sebagai alternatif.
    """
    try:
        result = get_transcript(url_or_id, preferred_langs=languages, proxy_config=_proxy_config)
    except VideoIdError as e:
        raise ValueError(f"URL/ID video tidak valid: {e}") from e
    except BlockedError as e:
        raise ValueError(f"Diblokir YouTube: {e}") from e
    except NoTranscriptError as e:
        raise ValueError(
            f"{e} Coba tool transcribe_with_whisper sebagai alternatif "
            "(transkripsi dari audio, lebih lambat & pakai kuota Groq Whisper)."
        ) from e
    except VideoNotFoundError as e:
        raise ValueError(f"Video tidak ditemukan/tidak tersedia: {e}") from e
    except TranscriptFetchError as e:
        raise ValueError(str(e)) from e

    _transcript_cache[result.video_id] = result

    metadata = get_video_metadata(result.video_id, _YOUTUBE_API_KEY) if _YOUTUBE_API_KEY else None
    if not hs.get_entry(result.video_id):
        hs.add_entry(
            video_id=result.video_id,
            namespace=ai.video_id_to_namespace(result.video_id),
            title=(metadata.title if metadata else result.video_id),
            language=result.language,
            word_count=result.word_count,
            full_text=result.full_text,
            segments=result.segments,
        )

    return {
        "video_id": result.video_id,
        "language": result.language,
        "language_code": result.language_code,
        "is_generated": result.is_generated,
        "word_count": result.word_count,
        "title": metadata.title if metadata else None,
        "channel": metadata.channel_title if metadata else None,
        "full_text": result.full_text,
    }


@mcp.tool()
def transcribe_with_whisper(url_or_id: str, language: str | None = None) -> dict:
    """Transkripsi video YouTube dari AUDIO lewat Groq Whisper.

    Gunakan HANYA kalau `fetch_transcript` gagal karena video tidak punya
    caption sama sekali -- ini jalur fallback yang lebih lambat (download
    audio penuh) dan memakai kuota Groq Whisper API terpisah.
    """
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY belum diset di environment MCP server.")

    video_id = extract_video_id(url_or_id)
    try:
        result = transcribe_audio(
            url_or_id,
            video_id=video_id,
            groq_api_key=_GROQ_API_KEY,
            language=language,
            proxy_config=_proxy_config,
        )
    except AudioFetchError as e:
        raise ValueError(str(e)) from e

    _transcript_cache[result.video_id] = result

    if not hs.get_entry(result.video_id):
        hs.add_entry(
            video_id=result.video_id,
            namespace=ai.video_id_to_namespace(result.video_id),
            title=result.video_id,
            language=result.language,
            word_count=result.word_count,
            full_text=result.full_text,
            segments=result.segments,
        )

    return {
        "video_id": result.video_id,
        "language": result.language,
        "word_count": result.word_count,
        "full_text": result.full_text,
    }


@mcp.tool()
def list_transcript_languages(url_or_id: str) -> list[dict]:
    """Lihat daftar bahasa caption yang tersedia untuk sebuah video YouTube."""
    try:
        return list_available_languages(url_or_id, proxy_config=_proxy_config)
    except TranscriptFetchError as e:
        raise ValueError(str(e)) from e


@mcp.tool()
def summarize(video_id: str) -> str:
    """Buat ringkasan dari transcript video yang sudah diambil (fetch_transcript/transcribe_with_whisper)."""
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY belum diset di environment MCP server.")
    result = _require(video_id)
    try:
        summary = ai.summarize_transcript(result.full_text, _GROQ_API_KEY)
    except (ai.ConfigurationError, ai.QueryError, RuntimeError) as e:
        raise ValueError(str(e)) from e
    hs.update_entry(video_id, summary=summary)
    return summary


@mcp.tool()
def generate_faq(video_id: str, max_items: int = 10) -> list[dict]:
    """Buat daftar FAQ (pertanyaan+jawaban) dari transcript video, maksimal `max_items`."""
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY belum diset di environment MCP server.")
    result = _require(video_id)
    try:
        faq_items = ai.generate_faq(result.full_text, _GROQ_API_KEY, max_items=max_items)
    except (ai.ConfigurationError, ai.QueryError, RuntimeError) as e:
        raise ValueError(str(e)) from e
    hs.update_entry(video_id, faq_items=faq_items)
    return faq_items


@mcp.tool()
def translate(video_id: str) -> str:
    """Terjemahkan SELURUH transcript video ke Bahasa Indonesia (fallback ke Inggris kalau model menilai Indonesia kurang pas)."""
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY belum diset di environment MCP server.")
    result = _require(video_id)
    try:
        translation = ai.translate_transcript(result.full_text, _GROQ_API_KEY)
    except (ai.ConfigurationError, ai.QueryError, RuntimeError) as e:
        raise ValueError(str(e)) from e
    hs.update_entry(video_id, translation=translation)
    return translation


@mcp.tool()
def index_for_qa(video_id: str) -> str:
    """Index transcript video ke Pinecone supaya bisa dipakai untuk tool `ask_question` (RAG). Panggil sekali per video."""
    if not _GOOGLE_API_KEY or not _PINECONE_API_KEY:
        raise ValueError("GOOGLE_API_KEY dan PINECONE_API_KEY belum diset di environment MCP server.")
    result = _require(video_id)

    if ai.namespace_exists(_PINECONE_API_KEY, ai.video_id_to_namespace(video_id)):
        return f"Video {video_id} sudah pernah di-index sebelumnya -- langsung bisa pakai ask_question."

    try:
        ns = ai.index_transcript(video_id, result.full_text, _GOOGLE_API_KEY, _PINECONE_API_KEY)
    except (ai.ConfigurationError, ai.IndexingError) as e:
        raise ValueError(str(e)) from e
    return f"Video {video_id} berhasil di-index ({ns.chunk_count} chunk). Sekarang bisa pakai ask_question."


@mcp.tool()
def ask_question(video_id: str, question: str) -> dict:
    """Tanya-jawab bebas soal isi video (RAG) -- video harus sudah di-index lewat `index_for_qa` dulu."""
    if not (_GOOGLE_API_KEY and _GROQ_API_KEY and _PINECONE_API_KEY):
        raise ValueError("GOOGLE_API_KEY, GROQ_API_KEY, dan PINECONE_API_KEY harus diset di environment MCP server.")
    try:
        return ai.ask_question(video_id, question, _GOOGLE_API_KEY, _GROQ_API_KEY, _PINECONE_API_KEY)
    except (ai.ConfigurationError, ai.QueryError) as e:
        raise ValueError(str(e)) from e


@mcp.tool()
def get_transcript_text(video_id: str, with_timestamps: bool = False) -> str:
    """Ambil isi transcript penuh (atau versi dengan timestamp [mm:ss]) dari video yang sudah diproses sebelumnya."""
    result = _require(video_id)
    if with_timestamps and result.segments:
        return format_transcript_with_timestamps(result.segments)
    return result.full_text


@mcp.tool()
def list_history(limit: int = 20) -> list[dict]:
    """Lihat daftar video yang pernah diproses (riwayat lokal), terbaru dulu."""
    entries = hs.list_entries()
    entries.sort(key=lambda e: e.processed_at, reverse=True)
    return [
        {
            "video_id": e.video_id,
            "title": e.title,
            "language": e.language,
            "word_count": e.word_count,
            "processed_at": e.processed_at,
            "has_summary": bool(e.summary),
            "has_faq": bool(e.faq_items),
            "has_translation": bool(e.translation),
            "url": f"https://youtu.be/{e.video_id}",
        }
        for e in entries[:limit]
    ]


if __name__ == "__main__":
    mcp.run()
