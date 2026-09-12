"""audio_service.py — fallback transcript via audio extraction (yt-dlp + Groq Whisper).

Dipakai saat video TIDAK punya caption sama sekali (NoTranscriptError). Ini
jalur eksplisit (tombol "Transcribe via Whisper" di app.py), BUKAN fallback
otomatis -- karena biayanya jauh berbeda dari caption (download audio penuh +
kuota Groq Whisper) dan butuh waktu jauh lebih lama.

Sengaja TIDAK ada fallback ke faster-whisper lokal: Streamlit Cloud free tier
tidak punya GPU, CPU-nya lambat, dan model lokal butuh unduhan besar + risiko
timeout/OOM. Kalau Groq Whisper gagal/limit habis, error ditampilkan apa
adanya -- bukan diam-diam pindah ke model lokal yang lebih lambat.

Kredensial SELALU lewat parameter (bersumber dari st.secrets di app.py),
tidak pernah os.environ, konsisten dengan seluruh codebase.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from groq import Groq

from transcript_service import TranscriptFetchError, TranscriptResult

WHISPER_MODEL = "whisper-large-v3-turbo"  # lebih hemat kuota; "whisper-large-v3" = lebih akurat
MAX_UPLOAD_MB = 25          # batas ukuran file per request Groq Whisper
CHUNK_MINUTES = 10          # mp3 64kbps ~10 menit => ±5MB, aman di bawah batas


class AudioFetchError(TranscriptFetchError):
    """Gagal mengambil/transkripsi audio (download, ffmpeg, atau Groq Whisper)."""


def _proxy_to_url(proxy_config) -> str | None:
    """Ubah objek WebshareProxyConfig/GenericProxyConfig milik project jadi
    string URL polos yang bisa dipakai yt-dlp lewat flag --proxy."""
    if proxy_config is None:
        return None
    if isinstance(proxy_config, str):
        return proxy_config
    for attr in ("http_url", "url"):
        value = getattr(proxy_config, attr, None)
        if value:
            return value
    return None


def _download_audio(url: str, proxy: str | None, out_dir: str) -> str:
    out_tpl = os.path.join(out_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "64K",
        "-o", out_tpl,
        "--no-playlist", "--quiet", "--no-warnings",
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise AudioFetchError(f"Gagal download audio: {e.stderr.strip()[:300]}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioFetchError("Download audio timeout (video terlalu panjang atau koneksi lambat).") from e

    audio_path = os.path.join(out_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        raise AudioFetchError("Download audio selesai tapi file mp3 tidak ditemukan.")
    return audio_path


def _split_audio(mp3_path: str, minutes: int = CHUNK_MINUTES) -> list[str]:
    size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
    if size_mb <= MAX_UPLOAD_MB:
        return [mp3_path]

    out_dir = os.path.dirname(mp3_path)
    pattern = os.path.join(out_dir, "chunk_%03d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-f", "segment", "-segment_time", str(minutes * 60),
        "-c", "copy", pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise AudioFetchError("Gagal memotong audio (ffmpeg) untuk video yang panjang.") from e

    chunks = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("chunk_")
    )
    return chunks or [mp3_path]


def _transcribe_chunk(client: Groq, path: str, language: str | None) -> tuple[list[dict], str]:
    with open(path, "rb") as f:
        kwargs = dict(model=WHISPER_MODEL, file=f, response_format="verbose_json")
        if language:
            kwargs["language"] = language
        resp = client.audio.transcriptions.create(**kwargs)

    segments = [
        {
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": s["text"].strip(),
        }
        for s in getattr(resp, "segments", []) or []
        if s.get("text", "").strip()
    ]
    return segments, getattr(resp, "language", "") or ""


def transcribe_audio(
    url: str,
    video_id: str,
    groq_api_key: str,
    language: str | None = None,
    proxy_config=None,
) -> TranscriptResult:
    """Download audio video, potong jika perlu, transkripsi via Groq Whisper.

    Return TranscriptResult -- bentuk yang sama persis dengan hasil caption
    biasa, supaya ai_service.py/report_builder.py tidak perlu perubahan apa
    pun untuk memakainya.
    """
    if not groq_api_key:
        raise AudioFetchError("GROQ_API_KEY belum diset -- transkripsi Whisper butuh kunci Groq.")

    proxy = _proxy_to_url(proxy_config)
    client = Groq(api_key=groq_api_key)

    with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp_dir:
        audio_path = _download_audio(url, proxy, tmp_dir)
        chunks = _split_audio(audio_path)

        all_segments: list[dict] = []
        detected_language = language or ""
        offset = 0.0

        for chunk_path in chunks:
            try:
                segs, lang = _transcribe_chunk(client, chunk_path, language)
            except Exception as e:  # noqa: BLE001 -- semua error Groq Whisper dibungkus rapi
                raise AudioFetchError(f"Transkripsi Whisper gagal: {e}") from e

            for s in segs:
                s["start"] = round(s["start"] + offset, 2)
                s["end"] = round(s["end"] + offset, 2)
                all_segments.append(s)

            if all_segments:
                offset = all_segments[-1]["end"]
            detected_language = detected_language or lang

        if not all_segments:
            raise AudioFetchError("Whisper tidak menghasilkan teks apa pun dari audio video ini.")

        full_text = " ".join(s["text"] for s in all_segments)

        return TranscriptResult(
            video_id=video_id,
            language=detected_language or (language or "unknown"),
            language_code=(language or "auto"),
            is_generated=True,
            full_text=full_text,
            segments=all_segments,
        )
