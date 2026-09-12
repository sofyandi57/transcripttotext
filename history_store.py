"""
history_store.py

Penyimpanan metadata riwayat video (judul, kapan diproses, ringkasan
singkat) PLUS isi transcript penuh -- supaya bisa di-download lagi dari
tab Riwayat tanpa fetch ulang ke YouTube. TERPISAH dari vector embedding
di Pinecone (Pinecone dipakai untuk RAG/similarity search, bukan buat
nyimpen transcript polos yang gampang diambil balik).

Kenapa perlu file terpisah:
Pinecone menyimpan embedding + metadata PER CHUNK (dan chunk-nya saling
overlap untuk keperluan retrieval), bukan per video secara utuh. Untuk
menampilkan daftar "riwayat video yang pernah dianalisa" di UI (dengan
judul yang enak dibaca manusia, bukan cuma hash namespace) DAN untuk
menyediakan tombol download transcript yang persis sama dengan aslinya,
kita butuh salinan tersendiri. Ini disimpan sebagai JSON sederhana.

CATATAN PENTING -- keterbatasan penyimpanan di Streamlit Cloud:
File JSON ini disimpan di disk lokal container Streamlit Cloud, YANG
TIDAK PERSISTEN -- bisa hilang saat app di-redeploy atau sleep lalu
bangun lagi. Jadi:
  - Vector embedding (isi transcript, buat Q&A) tetap aman permanen di
    Pinecone.
  - Tapi daftar "riwayat" (judul, transcript tersimpan, kapan diproses)
    bisa hilang dan perlu dibangun ulang -- proses lagi video yang sama,
    sistem akan skip re-indexing (namespace sudah ada) tapi transcript
    utuh & tombol download di Riwayat perlu "terisi ulang" lewat itu.
Ini trade-off yang disepakati di awal (lihat percakapan sebelumnya) --
kalau nanti riwayat perlu benar-benar permanen, opsi upgrade: simpan
JSON ini di Pinecone juga (sebagai satu vector dummy di namespace
khusus "_history"), atau pakai layanan storage terpisah (mis. Google
Sheets API, Supabase free tier, dll). Belum diimplementasikan di sini
supaya scope tetap terkendali.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
    # full_text/segments: disimpan supaya transcript bisa di-download lagi
    # dari tab Riwayat tanpa fetch ulang ke YouTube. Default kosong supaya
    # entry LAMA (dari sebelum field ini ada) tetap bisa dibaca tanpa error --
    # tinggal tidak akan ada tombol download untuk entry lama itu.
    full_text: str = ""
    segments: list = field(default_factory=list)
    # summary/faq_items: cache hasil AI supaya video yang sudah pernah
    # diringkas/dibikinin FAQ TIDAK perlu panggil Groq lagi kalau video yang
    # sama diproses ulang -- langsung tampilkan yang tersimpan. Ini
    # penghematan token nyata: kuota Groq gratis kecil, jangan bayar dua kali
    # untuk hasil yang sama persis.
    summary: str = ""
    faq_items: list = field(default_factory=list)


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
    full_text: str = "",
    segments: list | None = None,
) -> VideoHistoryEntry:
    entry = VideoHistoryEntry(
        video_id=video_id,
        namespace=namespace,
        title=title,
        language=language,
        word_count=word_count,
        processed_at=datetime.now(timezone.utc).isoformat(),
        full_text=full_text,
        segments=segments or [],
    )
    data = _load_all()
    data[video_id] = asdict(entry)
    _save_all(data)
    return entry


def update_entry(video_id: str, **fields) -> VideoHistoryEntry | None:
    """
    Update sebagian field entry yang sudah ada (mis. summary/faq_items
    setelah dibuat, karena add_entry() dipanggil lebih dulu saat indexing,
    sebelum ringkasan/FAQ sempat dibuat). Return None kalau entry belum ada
    sama sekali (video belum pernah di-index) -- pemanggil harus sudah pasti
    video ini ada di riwayat sebelum update.
    """
    data = _load_all()
    if video_id not in data:
        return None
    data[video_id].update(fields)
    _save_all(data)
    return VideoHistoryEntry(**data[video_id])


def get_entry(video_id: str) -> VideoHistoryEntry | None:
    data = _load_all()
    raw = data.get(video_id)
    return VideoHistoryEntry(**raw) if raw else None


def list_entries() -> list[VideoHistoryEntry]:
    data = _load_all()
    # **v pakai default dataclass untuk field yang belum ada di entry lama
    # (full_text/segments) -- filter dulu ke key yang dikenal biar aman
    # kalau suatu saat ada key asing lain di JSON lama.
    known_fields = {f for f in VideoHistoryEntry.__dataclass_fields__}
    entries = [VideoHistoryEntry(**{k: v for k, v in item.items() if k in known_fields}) for item in data.values()]
    return sorted(entries, key=lambda e: e.processed_at, reverse=True)


def delete_entry(video_id: str) -> bool:
    data = _load_all()
    if video_id in data:
        del data[video_id]
        _save_all(data)
        return True
    return False
