"""
transcript_service.py

Logika inti untuk mengambil transcript YouTube.
Dipisah dari app.py supaya:
  1. Bisa ditest tanpa perlu jalankan Streamlit.
  2. Gampang di-extend ke platform lain (Vimeo, dll) nanti.

Catatan penting soal arsitektur:
- YouTube memblokir request transcript yang datang dari IP milik
  cloud provider (AWS, GCP, Azure, dan termasuk Streamlit Cloud).
- Solusi satu-satunya yang terbukti reliable adalah proxy RESIDENTIAL
  berputar (bukan "Proxy Server" biasa, dan bukan "Static Residential").
  Free tier proxy TIDAK akan cukup -- ini dikonfirmasi langsung oleh
  maintainer library youtube-transcript-api.
- Karena itu, service ini didesain supaya proxy adalah first-class
  citizen, bukan tambahan opsional yang mudah dilupakan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import WebshareProxyConfig


# ---------------------------------------------------------------------------
# Exceptions -- dibuat custom supaya lapisan UI (app.py) bisa menampilkan
# pesan yang sesuai konteks, bukan traceback mentah.
# ---------------------------------------------------------------------------

class TranscriptFetchError(Exception):
    """Base exception untuk semua kegagalan pengambilan transcript."""


class VideoIdError(TranscriptFetchError):
    """URL atau ID video tidak bisa dikenali."""


class BlockedError(TranscriptFetchError):
    """YouTube memblokir IP -- biasanya karena proxy belum dikonfigurasi
    atau proxy yang dipakai bukan tipe residential yang valid."""


class NoTranscriptError(TranscriptFetchError):
    """Video ada, tapi transcript tidak tersedia (caption dimatikan,
    bahasa tidak cocok, dll)."""


class VideoNotFoundError(TranscriptFetchError):
    """Video private / dihapus / region-locked."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    full_text: str
    segments: list = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# ---------------------------------------------------------------------------
# Video ID extraction
# ---------------------------------------------------------------------------

_ID_PATTERNS = [
    re.compile(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)"),  # watch?v=, /embed/, /v/
    re.compile(r"youtu\.be/([0-9A-Za-z_-]{11})"),
]


def extract_video_id(url_or_id: str) -> str:
    """Terima full URL YouTube (berbagai format) atau video ID mentah (11 char)."""
    url_or_id = url_or_id.strip()

    for pattern in _ID_PATTERNS:
        match = pattern.search(url_or_id)
        if match:
            return match.group(1)

    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id

    raise VideoIdError(
        f"Tidak bisa mengenali video ID dari input: '{url_or_id}'. "
        "Pastikan ini URL YouTube yang valid atau video ID 11 karakter."
    )


# ---------------------------------------------------------------------------
# Proxy configuration
# ---------------------------------------------------------------------------

def build_proxy_config(webshare_username: str | None, webshare_password: str | None):
    """
    Bangun konfigurasi proxy Webshare kalau kredensial tersedia.
    Return None kalau tidak ada kredensial -- pemanggil harus memutuskan
    apakah mau lanjut tanpa proxy (berisiko RequestBlocked di cloud) atau
    berhenti dengan pesan error yang jelas.
    """
    if webshare_username and webshare_password:
        return WebshareProxyConfig(
            proxy_username=webshare_username,
            proxy_password=webshare_password,
        )
    return None


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def get_transcript(
    url_or_id: str,
    preferred_langs: list[str] | None = None,
    translate_to: str | None = None,
    proxy_config=None,
) -> TranscriptResult:
    """
    Ambil transcript video YouTube.

    Parameters
    ----------
    url_or_id : str
        URL lengkap atau video ID.
    preferred_langs : list[str], optional
        Urutan preferensi bahasa kode ISO, misal ["id", "en"].
        Default: ["id", "en"].
    translate_to : str, optional
        Kode bahasa tujuan terjemahan otomatis (misal "en"), kalau
        transcript aslinya mendukung translate.
    proxy_config : WebshareProxyConfig | GenericProxyConfig | None
        Konfigurasi proxy. Wajib diisi kalau dijalankan di server cloud
        (Streamlit Cloud, dll) -- tanpa ini, hampir pasti kena BlockedError.

    Returns
    -------
    TranscriptResult

    Raises
    ------
    VideoIdError, BlockedError, NoTranscriptError, VideoNotFoundError
    """
    video_id = extract_video_id(url_or_id)

    if preferred_langs is None:
        preferred_langs = ["id", "en"]

    api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except RequestBlocked as e:
        raise BlockedError(
            "YouTube memblokir IP ini. Kalau aplikasi berjalan di server "
            "cloud (termasuk Streamlit Cloud), proxy residential WAJIB "
            "dikonfigurasi -- lihat README untuk setup Webshare."
        ) from e
    except TranscriptsDisabled as e:
        raise NoTranscriptError(
            f"Video {video_id}: pemilik video mematikan caption/transcript."
        ) from e
    except VideoUnavailable as e:
        raise VideoNotFoundError(
            f"Video {video_id}: tidak tersedia (private, dihapus, atau region-locked)."
        ) from e

    try:
        transcript_obj = transcript_list.find_transcript(preferred_langs)
    except NoTranscriptFound:
        available = list(transcript_list)
        if not available:
            raise NoTranscriptError(f"Video {video_id}: tidak ada transcript sama sekali.")
        transcript_obj = available[0]

    if translate_to and transcript_obj.language_code != translate_to:
        if transcript_obj.is_translatable:
            transcript_obj = transcript_obj.translate(translate_to)
        # kalau tidak translatable, diam-diam pakai bahasa asli --
        # pemanggil (UI) bisa cek transcript_obj.language_code untuk tahu.

    try:
        fetched = transcript_obj.fetch()
    except RequestBlocked as e:
        raise BlockedError(
            "YouTube memblokir IP ini saat mengambil isi transcript "
            "(list berhasil, tapi fetch gagal). Cek konfigurasi proxy."
        ) from e

    full_text = " ".join(seg.text for seg in fetched)

    return TranscriptResult(
        video_id=video_id,
        language=transcript_obj.language,
        language_code=transcript_obj.language_code,
        is_generated=transcript_obj.is_generated,
        full_text=full_text,
        segments=[{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched],
    )


def list_available_languages(url_or_id: str, proxy_config=None) -> list[dict]:
    """
    Utility tambahan: lihat semua bahasa transcript yang tersedia untuk
    sebuah video, tanpa harus fetch isinya. Berguna untuk UI dropdown.
    """
    video_id = extract_video_id(url_or_id)
    api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except RequestBlocked as e:
        raise BlockedError("YouTube memblokir IP ini. Cek konfigurasi proxy.") from e
    except TranscriptsDisabled as e:
        raise NoTranscriptError(f"Video {video_id}: caption dimatikan.") from e
    except VideoUnavailable as e:
        raise VideoNotFoundError(f"Video {video_id}: tidak tersedia.") from e

    return [
        {
            "language": t.language,
            "language_code": t.language_code,
            "is_generated": t.is_generated,
            "is_translatable": t.is_translatable,
        }
        for t in transcript_list
    ]
