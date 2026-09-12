"""
ai_service.py

Logika AI: embedding, penyimpanan ke Pinecone, ringkasan, dan Q&A (RAG)
berbasis transcript video. Dipisah dari app.py dan transcript_service.py
supaya masing-masing lapisan punya tanggung jawab jelas:

  transcript_service.py -> ambil transcript mentah
  ai_service.py          -> proses transcript jadi ringkasan & jawab pertanyaan
  app.py                 -> UI saja

CATATAN PENTING SOAL PROVIDER & NAMA MODEL:
Dua provider AI dipakai untuk dua hal yang beda:
  - Gemini  -> HANYA untuk embedding (index_transcript, ask_question).
               Dipilih karena video yang sudah di-index sebelumnya di
               Pinecone pakai vector Gemini 768 dimensi -- ganti provider
               embedding berarti semua video lama harus di-index ulang.
  - Groq    -> untuk ringkasan & jawaban Q&A (summarize_transcript,
               ask_question). Dipindah dari Gemini karena free tier Gemini
               gampang kena rate limit (429); Groq punya free tier yang
               jauh lebih longgar dan hosting-nya cepat.

Model API berganti dengan cepat (dalam hitungan bulan, model lama
di-shutdown). Supaya kode ini tidak "mati" begitu provider mematikan model
lama, nama model TIDAK di-hardcode di dalam fungsi -- semua diteruskan
sebagai parameter dengan default yang gampang diubah di satu tempat saja
(lihat DEFAULT_CHAT_MODEL dan DEFAULT_EMBEDDING_MODEL di bawah).

Kalau suatu saat muncul error semacam "model not found" atau "model
decommissioned": untuk Groq cek https://console.groq.com/docs/models,
untuk Gemini cek https://ai.google.dev/gemini-api/docs/models.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Konfigurasi default -- ubah di sini kalau provider mengganti/mematikan model.
# ---------------------------------------------------------------------------

DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"       # Groq -- ringkasan & Q&A
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"  # Gemini -- embedding saja
EMBEDDING_DIMENSION = 768  # Matryoshka: bisa 3072/1536/768. 768 dipilih untuk hemat storage Pinecone free tier.

PINECONE_INDEX_NAME = "youtube-transcript-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"  # wajib us-east-1 untuk free tier ("Starter") Pinecone

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

MAX_FAQ_ITEMS = 10

# Free tier Groq membatasi TPM (token per menit) cukup kecil (terlihat ~8000
# di akun yang dites, tapi bisa beda per akun/model) -- ini rolling/leaky
# bucket, BUKAN hard reset tiap menit, jadi retry singkat setelah delay
# pendek biasanya cukup untuk lolos. Transcript yang lebih besar dari
# SUMMARY_CHUNK_SIZE karakter (~3000 token) diproses map-reduce (potong,
# ringkas per bagian, gabung) supaya TIDAK PERNAH melebihi TPM dalam satu
# request, berapa pun panjang videonya (podcast 1-2 jam sekalipun).
SUMMARY_CHUNK_SIZE = 12000
SUMMARY_CHUNK_OVERLAP = 300

RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY_SECONDS = 6


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AIServiceError(Exception):
    """Base exception untuk semua kegagalan di layer AI."""


class ConfigurationError(AIServiceError):
    """API key tidak lengkap/tidak valid."""


class IndexingError(AIServiceError):
    """Gagal saat proses embedding atau upload ke Pinecone."""


class QueryError(AIServiceError):
    """Gagal saat proses tanya-jawab."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoNamespace:
    """Representasi satu video sebagai satu namespace di Pinecone index."""
    video_id: str
    namespace: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Helper: ekstrak teks dari response LLM
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """
    response.content dari ChatGoogleGenerativeAI bisa berupa string biasa
    (versi lama langchain-google-genai) ATAU list of content-blocks, misal
    [{"type": "text", "text": "...", "extras": {...}}] (versi baru, saat
    model mengembalikan bagian bertipe). Fungsi ini menormalkan keduanya
    jadi satu string biar aman ditampilkan lewat st.markdown().
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _is_rate_limit_error(e: Exception) -> bool:
    """Deteksi error TPM/rate-limit Groq (413 'tokens' atau 429) dari pesan exception."""
    msg = str(e).lower()
    return "rate_limit_exceeded" in msg or "429" in msg or ("413" in msg and "token" in msg)


def _invoke_with_retry(chain, payload: dict):
    """
    Panggil chain.invoke() dengan retry singkat kalau kena rate limit Groq.
    TPM Groq adalah rolling/leaky bucket (pulih dalam hitungan detik, bukan
    nunggu genap satu menit) -- jadi delay pendek + retry biasanya cukup,
    terutama kalau sebelumnya ada beberapa panggilan AI beruntun (ringkasan,
    FAQ, Q&A) yang menghabiskan kuota menit itu.
    """
    last_error: Exception | None = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        try:
            return chain.invoke(payload)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < RATE_LIMIT_MAX_RETRIES - 1:
                time.sleep(RATE_LIMIT_BASE_DELAY_SECONDS * (attempt + 1))
                continue
            raise
    raise last_error  # pragma: no cover -- selalu return atau raise di dalam loop


def _split_for_llm(text: str) -> list[str]:
    """Potong teks jadi bagian <= SUMMARY_CHUNK_SIZE karakter untuk map-reduce."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Namespace helper
# ---------------------------------------------------------------------------

def video_id_to_namespace(video_id: str) -> str:
    """
    Ubah video_id jadi namespace Pinecone yang aman.
    video_id YouTube (11 char alfanumerik + - _) sebenarnya sudah aman
    dipakai langsung, tapi di-hash tetap dilakukan supaya konsisten kalau
    nanti platform lain (Vimeo dll) punya format ID yang beda karakter.
    """
    return hashlib.sha256(video_id.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pinecone index management
# ---------------------------------------------------------------------------

def ensure_index_exists(pinecone_api_key: str) -> None:
    """
    Pastikan index Pinecone sudah ada. Kalau belum, buat baru.
    Index ini SATU untuk semua video -- video yang berbeda dipisah lewat
    namespace, bukan index terpisah (index terpisah akan boros kuota
    free tier yang cuma kasih 1 index).
    """
    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def _get_index(pinecone_api_key: str):
    """Kembalikan handle index kalau sudah ada, None kalau belum dibuat sama sekali."""
    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(PINECONE_INDEX_NAME):
        return None
    return pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# Proxy config override -- disimpan permanen di Pinecone (bukan file lokal),
# supaya perubahan IP proxy dari UI TIDAK hilang saat app di-redeploy atau
# "sleep" lalu bangun lagi di Streamlit Cloud (beda dengan video_history.json
# yang memang didesain sebagai penyimpanan sementara -- lihat history_store.py).
#
# Disimpan sebagai satu vector dummy (bukan embedding sungguhan) di namespace
# khusus "_app_config", datanya sendiri ada di metadata. Ini trik yang sama
# yang disebut sebagai upgrade path di README untuk riwayat video.
# ---------------------------------------------------------------------------

CONFIG_NAMESPACE = "_app_config"
PROXY_CONFIG_ID = "proxy_override"


def save_proxy_override(pinecone_api_key: str, proxy_host: str, proxy_port: str) -> None:
    """Simpan/timpa IP & port proxy aktif -- permanen sampai diganti/direset lagi."""
    if not pinecone_api_key:
        raise ConfigurationError("PINECONE_API_KEY harus diisi untuk menyimpan konfigurasi proxy.")
    if not proxy_host or not proxy_port:
        raise IndexingError("Host dan port proxy tidak boleh kosong.")

    ensure_index_exists(pinecone_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)

    dummy_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)  # Pinecone menolak vector semua-nol

    try:
        index.upsert(
            vectors=[
                {
                    "id": PROXY_CONFIG_ID,
                    "values": dummy_vector,
                    "metadata": {
                        "proxy_host": proxy_host,
                        "proxy_port": str(proxy_port),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ],
            namespace=CONFIG_NAMESPACE,
        )
    except Exception as e:
        raise IndexingError(f"Gagal menyimpan konfigurasi proxy ke Pinecone: {e}") from e


def get_proxy_override(pinecone_api_key: str) -> dict | None:
    """
    Ambil IP & port proxy yang tersimpan, kalau ada. Return None kalau belum
    pernah di-override -- pemanggil harus fallback ke PROXY_HOST/PROXY_PORT
    di Secrets.
    """
    if not pinecone_api_key:
        return None

    index = _get_index(pinecone_api_key)
    if index is None:
        return None

    try:
        result = index.fetch(ids=[PROXY_CONFIG_ID], namespace=CONFIG_NAMESPACE)
    except Exception:
        return None

    record = result.vectors.get(PROXY_CONFIG_ID)
    if not record:
        return None

    metadata = record.metadata or {}
    proxy_host = metadata.get("proxy_host")
    proxy_port = metadata.get("proxy_port")
    if not proxy_host or not proxy_port:
        return None

    return {
        "proxy_host": proxy_host,
        "proxy_port": proxy_port,
        "updated_at": metadata.get("updated_at", ""),
    }


def delete_proxy_override(pinecone_api_key: str) -> None:
    """Hapus override -- app kembali pakai PROXY_HOST/PROXY_PORT dari Secrets."""
    index = _get_index(pinecone_api_key)
    if index is None:
        return
    try:
        index.delete(ids=[PROXY_CONFIG_ID], namespace=CONFIG_NAMESPACE)
    except Exception as e:
        raise IndexingError(f"Gagal menghapus override proxy: {e}") from e


def namespace_exists(pinecone_api_key: str, namespace: str) -> bool:
    """Cek apakah sebuah video (namespace) sudah pernah diproses & disimpan sebelumnya."""
    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(PINECONE_INDEX_NAME):
        return False
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    existing_namespaces = stats.get("namespaces", {})
    return namespace in existing_namespaces and existing_namespaces[namespace].get("vector_count", 0) > 0


def list_processed_videos(pinecone_api_key: str) -> list[str]:
    """
    Ambil daftar semua namespace yang sudah ada di index -- ini dipakai
    untuk menampilkan 'riwayat video' di UI.
    Catatan: ini return namespace (hash), bukan video_id asli -- mapping
    video_id <-> namespace <-> judul video perlu disimpan terpisah
    (lihat history_store.py) karena Pinecone sendiri tidak menyimpan
    metadata di level namespace, cuma di level per-vector.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(PINECONE_INDEX_NAME):
        return []
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    return list(stats.get("namespaces", {}).keys())


# ---------------------------------------------------------------------------
# Embedding & indexing
# ---------------------------------------------------------------------------

def _build_embeddings(google_api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=google_api_key,
        output_dimensionality=EMBEDDING_DIMENSION,
        task_type="retrieval_document",
    )


def index_transcript(
    video_id: str,
    full_text: str,
    google_api_key: str,
    pinecone_api_key: str,
) -> VideoNamespace:
    """
    Pecah transcript jadi chunk, buat embedding, simpan ke Pinecone di
    namespace khusus video ini. Kalau video ini sudah pernah diproses
    sebelumnya, fungsi ini akan menimpa (re-index) -- dipanggil pemanggil
    hanya kalau memang belum ada (cek dulu pakai namespace_exists()).
    """
    if not google_api_key or not pinecone_api_key:
        raise ConfigurationError(
            "GOOGLE_API_KEY dan PINECONE_API_KEY harus diisi untuk memakai fitur AI."
        )

    ensure_index_exists(pinecone_api_key)
    namespace = video_id_to_namespace(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        raise IndexingError(f"Video {video_id}: transcript kosong, tidak ada yang bisa di-index.")

    try:
        embeddings = _build_embeddings(google_api_key)
        PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace,
            metadatas=[{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))],
            pinecone_api_key=pinecone_api_key,
        )
    except Exception as e:
        raise IndexingError(f"Gagal menyimpan embedding ke Pinecone: {e}") from e

    return VideoNamespace(video_id=video_id, namespace=namespace, chunk_count=len(chunks))


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah asisten yang meringkas transcript video secara akurat dan padat. "
     "Jangan menambahkan informasi yang tidak ada di transcript. "
     "Jawab dalam Bahasa Indonesia kecuali diminta lain."),
    ("human",
     "Ringkas transcript video berikut. Buat dalam format:\n"
     "1. Ringkasan umum (2-3 kalimat)\n"
     "2. Poin-poin penting (bullet points)\n"
     "3. Kesimpulan/takeaway utama\n\n"
     "Transcript:\n{transcript}"),
])

_SUMMARY_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu meringkas SATU BAGIAN (bukan keseluruhan) dari transcript video yang "
     "lebih panjang. Ringkas bagian ini secara padat, pertahankan semua detail "
     "penting -- ringkasan ini akan digabung dengan bagian lain untuk membuat "
     "ringkasan akhir. Jangan menambahkan informasi di luar teks ini."),
    ("human", "Bagian transcript:\n{chunk}"),
])


def summarize_transcript(
    full_text: str,
    groq_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Ringkas transcript lewat Groq. Free tier Groq membatasi TPM (token per
    menit) cukup kecil -- transcript video panjang (podcast 1-2 jam, bisa
    15.000-25.000+ kata) TIDAK MUAT dikirim sekaligus dalam satu request.
    Kalau transcript lebih panjang dari SUMMARY_CHUNK_SIZE karakter, dipotong
    lalu diringkas per-bagian (map), baru ringkasan-ringkasan itu digabung
    jadi satu ringkasan akhir (reduce) -- pola map-reduce standar, memastikan
    setiap request individual selalu di bawah limit TPM berapa pun panjang
    videonya.
    """
    if not groq_api_key:
        raise ConfigurationError("GROQ_API_KEY harus diisi untuk membuat ringkasan.")

    try:
        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.3, max_tokens=1500)

        if len(full_text) <= SUMMARY_CHUNK_SIZE:
            chain = _SUMMARY_PROMPT | llm
            response = _invoke_with_retry(chain, {"transcript": full_text})
            return _extract_text(response.content)

        chunks = _split_for_llm(full_text)
        map_chain = _SUMMARY_MAP_PROMPT | llm
        partial_summaries = [
            _extract_text(_invoke_with_retry(map_chain, {"chunk": chunk}).content) for chunk in chunks
        ]

        combined = "\n\n".join(partial_summaries)
        reduce_chain = _SUMMARY_PROMPT | llm
        final_response = _invoke_with_retry(reduce_chain, {"transcript": combined})
        return _extract_text(final_response.content)
    except Exception as e:
        raise QueryError(f"Gagal membuat ringkasan: {e}") from e


# ---------------------------------------------------------------------------
# Q&A (RAG)
# ---------------------------------------------------------------------------

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu menjawab pertanyaan HANYA berdasarkan konteks transcript video "
     "yang diberikan. Kalau jawabannya tidak ada di konteks, katakan dengan "
     "jujur bahwa informasi itu tidak dibahas di video ini -- jangan mengarang. "
     "Jawab dalam Bahasa Indonesia kecuali diminta lain."),
    ("human",
     "Konteks dari transcript video:\n{context}\n\nPertanyaan: {question}"),
])


def ask_question(
    video_id: str,
    question: str,
    google_api_key: str,
    groq_api_key: str,
    pinecone_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
    top_k: int = 5,
) -> dict:
    """
    Jawab pertanyaan soal isi video, berdasarkan retrieval dari Pinecone
    (bukan seluruh transcript sekaligus) -- ini yang membuat Q&A untuk
    video panjang tetap presisi dan hemat token. Embedding pencarian pakai
    Gemini (google_api_key, harus konsisten dengan yang dipakai saat
    index_transcript), jawaban akhir dibuat oleh Groq (groq_api_key).

    Returns
    -------
    dict dengan keys: "answer", "sources" (list of chunk text yang dipakai)
    """
    if not google_api_key or not groq_api_key or not pinecone_api_key:
        raise ConfigurationError(
            "GOOGLE_API_KEY, GROQ_API_KEY, dan PINECONE_API_KEY harus diisi untuk bertanya soal video."
        )

    namespace = video_id_to_namespace(video_id)

    if not namespace_exists(pinecone_api_key, namespace):
        raise QueryError(
            f"Video {video_id} belum di-index. Proses transcript-nya dulu "
            "sebelum bisa tanya-jawab."
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            google_api_key=google_api_key,
            output_dimensionality=EMBEDDING_DIMENSION,
            task_type="retrieval_query",  # beda dari task_type saat indexing ("retrieval_document")
        )
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
            pinecone_api_key=pinecone_api_key,
        )
        relevant_docs: list[Document] = vector_store.similarity_search(question, k=top_k)

        context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.2, max_tokens=1000)
        chain = _QA_PROMPT | llm
        response = _invoke_with_retry(chain, {"context": context, "question": question})

        return {
            "answer": _extract_text(response.content),
            "sources": [doc.page_content for doc in relevant_docs],
        }
    except QueryError:
        raise
    except Exception as e:
        raise QueryError(f"Gagal menjawab pertanyaan: {e}") from e


# ---------------------------------------------------------------------------
# FAQ otomatis
# ---------------------------------------------------------------------------

class _FAQItem(BaseModel):
    question: str = Field(description="Pertanyaan yang relevan dengan isi video")
    answer: str = Field(description="Jawaban singkat dan akurat, hanya berdasarkan transcript")


class _FAQList(BaseModel):
    items: list[_FAQItem] = Field(description="Daftar FAQ, urut dari paling penting")


_FAQ_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu membuat FAQ (pertanyaan yang sering ditanyakan penonton) dari transcript "
     "video. Pilih pertanyaan yang paling relevan dan penting yang kemungkinan besar "
     "ingin diketahui penonton soal isi video ini. Jawaban HARUS berdasarkan transcript "
     "saja -- jangan mengarang informasi yang tidak ada di sana. "
     "Jawab dalam Bahasa Indonesia kecuali diminta lain."),
    ("human",
     "Buat maksimal {max_items} FAQ dari transcript video berikut:\n\n{transcript}"),
])

_FAQ_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu membuat draft FAQ dari SATU BAGIAN (bukan keseluruhan) transcript video "
     "yang lebih panjang. Pilih maksimal 4 pertanyaan paling penting yang relevan "
     "dengan bagian ini saja. Jawaban HARUS berdasarkan teks ini saja -- jangan "
     "mengarang. Jawab dalam Bahasa Indonesia kecuali diminta lain."),
    ("human", "Bagian transcript:\n{chunk}"),
])

_FAQ_REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu diberi kumpulan draft FAQ dari berbagai bagian sebuah video. Pilih "
     "maksimal {max_items} yang PALING penting dan relevan untuk keseluruhan "
     "video, gabungkan/hilangkan yang duplikat atau mirip. Boleh menghaluskan "
     "kalimat, tapi jangan ubah makna atau mengarang fakta baru di luar draft."),
    ("human", "Draft FAQ:\n{drafts}"),
])


def generate_faq(
    full_text: str,
    groq_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
    max_items: int = MAX_FAQ_ITEMS,
) -> list[dict]:
    """
    Buat daftar FAQ (maksimal `max_items`) dari transcript, lewat Groq
    Structured Output API (method="json_schema" -- constrained decoding,
    output DIJAMIN sesuai schema; default "function_calling" pernah
    menghasilkan JSON tidak valid untuk daftar sepanjang ini).

    Transcript yang lebih panjang dari SUMMARY_CHUNK_SIZE karakter diproses
    map-reduce sama seperti summarize_transcript(): draft FAQ dibuat per
    bagian (map), lalu satu panggilan terakhir memilih/menggabung draft
    terbaik (reduce) -- supaya tidak pernah melebihi limit TPM Groq berapa
    pun panjang videonya.

    Returns
    -------
    list of {"question": str, "answer": str}, maksimal `max_items` item.
    """
    if not groq_api_key:
        raise ConfigurationError("GROQ_API_KEY harus diisi untuk membuat FAQ.")

    try:
        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.3, max_tokens=2000)
        structured_llm = llm.with_structured_output(_FAQList, method="json_schema")

        if len(full_text) <= SUMMARY_CHUNK_SIZE:
            chain = _FAQ_PROMPT | structured_llm
            result: _FAQList = _invoke_with_retry(chain, {"transcript": full_text, "max_items": max_items})
            return [{"question": item.question, "answer": item.answer} for item in result.items[:max_items]]

        chunks = _split_for_llm(full_text)
        map_chain = _FAQ_MAP_PROMPT | structured_llm
        drafts: list[str] = []
        for chunk in chunks:
            partial: _FAQList = _invoke_with_retry(map_chain, {"chunk": chunk})
            drafts.extend(f"Q: {item.question}\nA: {item.answer}" for item in partial.items)

        reduce_chain = _FAQ_REDUCE_PROMPT | structured_llm
        final: _FAQList = _invoke_with_retry(
            reduce_chain, {"drafts": "\n\n".join(drafts), "max_items": max_items}
        )
        return [{"question": item.question, "answer": item.answer} for item in final.items[:max_items]]
    except Exception as e:
        raise QueryError(f"Gagal membuat FAQ: {e}") from e
