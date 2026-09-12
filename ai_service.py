"""
ai_service.py

Logika AI: embedding, penyimpanan ke Pinecone, ringkasan, dan Q&A (RAG)
berbasis transcript video. Dipisah dari app.py dan transcript_service.py
supaya masing-masing lapisan punya tanggung jawab jelas:

  transcript_service.py -> ambil transcript mentah
  ai_service.py          -> proses transcript jadi ringkasan & jawab pertanyaan
  app.py                 -> UI saja

CATATAN PENTING SOAL NAMA MODEL:
Model Gemini berganti dengan cepat (dalam hitungan bulan, model lama
di-shutdown). Supaya kode ini tidak "mati" begitu Google mematikan model
lama, nama model TIDAK di-hardcode di dalam fungsi -- semua diteruskan
sebagai parameter dengan default yang gampang diubah di satu tempat saja
(lihat DEFAULT_CHAT_MODEL dan DEFAULT_EMBEDDING_MODEL di bawah).

Kalau suatu saat muncul error semacam "model not found" atau "model
deprecated", kemungkinan besar cukup update dua konstanta ini -- cek
https://ai.google.dev/gemini-api/docs/models untuk nama model terbaru.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Konfigurasi default -- ubah di sini kalau Google mengganti/mematikan model.
# ---------------------------------------------------------------------------

DEFAULT_CHAT_MODEL = "gemini-flash-latest"       # alias, otomatis ikut versi stabil terbaru
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 768  # Matryoshka: bisa 3072/1536/768. 768 dipilih untuk hemat storage Pinecone free tier.

PINECONE_INDEX_NAME = "youtube-transcript-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"  # wajib us-east-1 untuk free tier ("Starter") Pinecone

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


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
    google_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Ringkas transcript. Untuk video panjang (>1 jam), transcript bisa
    15.000-25.000+ kata -- ini masih aman untuk context window Gemini
    Flash (bisa >1 juta token), jadi TIDAK perlu chunking untuk
    summarization (beda dengan indexing, yang memang perlu di-chunk
    untuk retrieval presisi).
    """
    if not google_api_key:
        raise ConfigurationError("GOOGLE_API_KEY harus diisi untuk membuat ringkasan.")

    try:
        llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=google_api_key, temperature=0.3)
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
    pinecone_api_key: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
    top_k: int = 5,
) -> dict:
    """
    Jawab pertanyaan soal isi video, berdasarkan retrieval dari Pinecone
    (bukan seluruh transcript sekaligus) -- ini yang membuat Q&A untuk
    video panjang tetap presisi dan hemat token.

    Returns
    -------
    dict dengan keys: "answer", "sources" (list of chunk text yang dipakai)
    """
    if not google_api_key or not pinecone_api_key:
        raise ConfigurationError(
            "GOOGLE_API_KEY dan PINECONE_API_KEY harus diisi untuk bertanya soal video."
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

        llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=google_api_key, temperature=0.2)
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
