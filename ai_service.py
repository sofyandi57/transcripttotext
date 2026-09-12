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


def summarize_transcript(
    full_text: str,
    groq_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Ringkas transcript lewat Groq. Untuk video panjang (>1 jam), transcript
    bisa 15.000-25.000+ kata -- model Groq default (Llama 3.3 70B) punya
    context window 128K token, cukup untuk sebagian besar video tanpa perlu
    chunking (beda dengan indexing, yang memang perlu di-chunk untuk
    retrieval presisi). Video YANG SANGAT panjang (>~2.5 jam) bisa melebihi
    ini -- kalau muncul error context length, itu tandanya.
    """
    if not groq_api_key:
        raise ConfigurationError("GROQ_API_KEY harus diisi untuk membuat ringkasan.")

    try:
        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.3)
        chain = _SUMMARY_PROMPT | llm
        response = chain.invoke({"transcript": full_text})
        return _extract_text(response.content)
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

        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.2)
        chain = _QA_PROMPT | llm
        response = chain.invoke({"context": context, "question": question})

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


def generate_faq(
    full_text: str,
    groq_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
    max_items: int = MAX_FAQ_ITEMS,
) -> list[dict]:
    """
    Buat daftar FAQ (maksimal `max_items`) dari transcript, lewat Groq
    structured output (function calling) supaya hasilnya list Q&A yang
    rapi, bukan teks bebas yang perlu di-parse manual.

    Returns
    -------
    list of {"question": str, "answer": str}, maksimal `max_items` item
    (model kadang mengabaikan batas jumlah di prompt -- dipotong manual
    di sini sebagai jaminan).
    """
    if not groq_api_key:
        raise ConfigurationError("GROQ_API_KEY harus diisi untuk membuat FAQ.")

    try:
        llm = ChatGroq(model=chat_model, api_key=groq_api_key, temperature=0.3)
        # method="json_schema" -- Groq's dedicated Structured Output API (constrained
        # decoding, output DIJAMIN sesuai schema). Default "function_calling" pernah
        # menghasilkan JSON tidak valid (tool_use_failed) untuk daftar sepanjang ini.
        structured_llm = llm.with_structured_output(_FAQList, method="json_schema")
        chain = _FAQ_PROMPT | structured_llm
        result: _FAQList = chain.invoke({"transcript": full_text, "max_items": max_items})
        return [{"question": item.question, "answer": item.answer} for item in result.items[:max_items]]
    except Exception as e:
        raise QueryError(f"Gagal membuat FAQ: {e}") from e
