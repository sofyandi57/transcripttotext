"""
history_store.py

Penyimpanan metadata riwayat video (judul, kapan diproses, ringkasan
singkat) -- TERPISAH dari vector embedding di Pinecone.

Kenapa perlu file terpisah:
Pinecone menyimpan embedding + metadata PER CHUNK, bukan per video.
Untuk menampilkan daftar "riwayat video yang pernah dianalisa" di UI
(dengan judul yang enak dibaca manusia, bukan cuma hash namespace),
kita butuh pemetaan video_id -> info ringkas. Ini disimpan sebagai
JSON sederhana.

CATATAN PENTING -- keterbatasan penyimpanan di Streamlit Cloud:
File JSON ini disimpan di disk lokal container Streamlit Cloud, YANG
TIDAK PERSISTEN -- bisa hilang saat app di-redeploy atau sleep lalu
bangun lagi. Jadi:
  - Vector embedding (isi transcript) tetap aman permanen di Pinecone.
  - Tapi daftar "riwayat" (judul, kapan diproses) bisa hilang dan perlu
    dibangun ulang.
Ini trade-off yang disepakati di awal (lihat percakapan sebelumnya) --
kalau nanti riwayat perlu benar-benar permanen, opsi upgrade: simpan
JSON ini di Pinecone juga (sebagai satu vector dummy di namespace
khusus "_history"), atau pakai layanan storage terpisah (mis. Google
Sheets API, Supabase free tier, dll). Belum diimplementasikan di sini
supaya scope tetap terkendali.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = Path("video_history.json")


@dataclass
class VideoHistoryEntry:
    video_id: str
    namespace: str
    title: str
    language: str
    word_count: int
    processed_at: str  # ISO format


def _load_all() -> dict[str, dict]:
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict[str, dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_entry(
    video_id: str,
    namespace: str,
    title: str,
    language: str,
    word_count: int,
) -> VideoHistoryEntry:
    entry = VideoHistoryEntry(
        video_id=video_id,
        namespace=namespace,
        title=title,
        language=language,
        word_count=word_count,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )
    data = _load_all()
    data[video_id] = asdict(entry)
    _save_all(data)
    return entry


def get_entry(video_id: str) -> VideoHistoryEntry | None:
    data = _load_all()
    raw = data.get(video_id)
    return VideoHistoryEntry(**raw) if raw else None


def list_entries() -> list[VideoHistoryEntry]:
    data = _load_all()
    entries = [VideoHistoryEntry(**v) for v in data.values()]
    return sorted(entries, key=lambda e: e.processed_at, reverse=True)


def delete_entry(video_id: str) -> bool:
    data = _load_all()
    if video_id in data:
        del data[video_id]
        _save_all(data)
        return True
    return False
