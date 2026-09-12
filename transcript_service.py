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

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig


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

def build_proxy_config(
    webshare_username: str | None,
    webshare_password: str | None,
    proxy_host: str | None = None,
    proxy_port: str | int | None = None,
):
    """
    Bangun konfigurasi proxy. Ada dua mode:

    1. Rotating residential Webshare asli -- HANYA username + password
       (tanpa proxy_host/proxy_port). Library otomatis connect ke gateway
       rotating resmi Webshare (p.webshare.io) yang mengganti IP tiap
       request. Ini mode yang paling reliable untuk bypass blokir YouTube.

    2. Proxy statis generik -- username + password + proxy_host + proxy_port.
       Dipakai untuk paket "Proxy List"/"Proxy Server" (IP tetap, baik dari
       Webshare maupun provider lain manapun). IP TIDAK ikut rotasi otomatis,
       jadi lebih gampang ke-block YouTube dibanding mode 1 (lihat README
       bagian batasan proxy) -- tapi tetap lebih baik daripada tanpa proxy
       sama sekali kalau mode 1 belum tersedia.

    Return None kalau tidak ada kredensial sama sekali -- pemanggil harus
    memutuskan apakah mau lanjut tanpa proxy (berisiko RequestBlocked di
    cloud) atau berhenti dengan pesan error yang jelas.
    """
    if not webshare_username or not webshare_password:
        return None

    if proxy_host and proxy_port:
        proxy_url = f"http://{webshare_username}:{webshare_password}@{proxy_host}:{proxy_port}"
        return GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)

    return WebshareProxyConfig(
        proxy_username=webshare_username,
        proxy_password=webshare_password,
    )


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


# ---------------------------------------------------------------------------
# Format transcript dengan timestamp (untuk file download -- BUKAN untuk
# dikirim ke AI. Timestamp per baris cuma menambah noise/token buat LLM
# tanpa manfaat untuk ringkasan/FAQ/Q&A, jadi result.full_text yang polos
# tetap dipakai di ai_service.py; ini murni untuk keperluan baca manusia.)
# ---------------------------------------------------------------------------

def _format_seconds(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Ubah segments (dari TranscriptResult.segments) jadi teks '[mm:ss] isi' per baris."""
    return "\n".join(f"[{_format_seconds(seg['start'])}] {seg['text']}" for seg in segments)


# ---------------------------------------------------------------------------
# Metadata video via YouTube Data API v3 -- OPSIONAL. Beda dari
# youtube-transcript-api (yang ambil transcript tanpa API key resmi), ini
# butuh API key terpisah dari Google Cloud Console (bukan Google AI Studio).
# Kegagalan di sini (key kosong, quota habis, video tidak ditemukan) TIDAK
# BOLEH menghentikan alur utama -- transcript tetap harus bisa diambil tanpa
# metadata ini, makanya fungsi ini return None alih-alih raise exception.
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    title: str
    channel_title: str
    published_at: str       # "YYYY-MM-DD", tanggal upload ASLI di YouTube
    duration_display: str   # "19:32" atau "1:02:10"
    view_count: int
    description: str        # dipotong pendek, lihat _MAX_DESCRIPTION_CHARS


_MAX_DESCRIPTION_CHARS = 280


def _parse_iso8601_duration(duration: str) -> str:
    """Ubah durasi ISO 8601 dari YouTube (mis. 'PT1H2M10S') jadi 'h:mm:ss'/'mm:ss'."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return "0:00"
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def get_video_metadata(video_id: str, youtube_api_key: str) -> VideoMetadata | None:
    """
    Ambil judul, nama channel, tanggal upload asli, durasi, jumlah views,
    dan deskripsi singkat lewat YouTube Data API v3. Return None kalau key
    kosong atau request gagal apa pun sebabnya (fitur ini best-effort).
    """
    if not youtube_api_key:
        return None

    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "id": video_id,
                "part": "snippet,contentDetails,statistics",
                "key": youtube_api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            return None

        item = items[0]
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        statistics = item.get("statistics", {})

        description = (snippet.get("description") or "").strip()
        if len(description) > _MAX_DESCRIPTION_CHARS:
            description = description[:_MAX_DESCRIPTION_CHARS].rstrip() + "..."

        return VideoMetadata(
            title=snippet.get("title", ""),
            channel_title=snippet.get("channelTitle", ""),
            published_at=(snippet.get("publishedAt") or "")[:10],
            duration_display=_parse_iso8601_duration(content_details.get("duration", "")),
            view_count=int(statistics.get("viewCount", 0)),
            description=description,
        )
    except Exception:
        return None
