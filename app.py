"""
app.py -- Streamlit UI untuk YouTube Transcript Grabber + AI Q&A.

Arsitektur:
  transcript_service.py -> ambil transcript mentah dari YouTube
  ai_service.py          -> embedding, index ke Pinecone, ringkasan, Q&A
  history_store.py       -> metadata riwayat video (judul, dll)
  app.py (file ini)      -> UI & orkestrasi, tidak ada logika bisnis di sini
"""

import streamlit as st

import ai_service as ai
import history_store as hs
from transcript_service import (
    BlockedError,
    NoTranscriptError,
    TranscriptFetchError,
    VideoIdError,
    VideoNotFoundError,
    build_proxy_config,
    extract_video_id,
    get_transcript,
    list_available_languages,
)

st.set_page_config(page_title="Transcript AI Powerhouse", page_icon="🧠", layout="centered")

st.title("🧠 YouTube Transcript AI Powerhouse")
st.caption("Ambil transcript, dapatkan ringkasan, dan tanya-jawab soal isi video pakai AI.")

# ---------------------------------------------------------------------------
# Konfigurasi API key -- diisi langsung oleh pengguna di UI (sidebar).
# Key HANYA disimpan di session_state (memori sesi browser saat ini),
# tidak pernah ditulis ke disk -- hilang begitu tab ditutup/direfresh.
# Kalau deployer sudah isi lewat Streamlit Secrets, itu dipakai sebagai
# nilai default supaya tidak perlu diketik ulang tiap sesi.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("🔑 API Keys")
    st.caption(
        "Isi API key kamu sendiri. Key ini hanya dipakai untuk sesi kamu saat ini "
        "dan tidak disimpan di server."
    )
    google_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.secrets.get("GOOGLE_API_KEY", ""),
        type="password",
        key="google_api_key_input",
        help="Gratis di https://aistudio.google.com/apikey",
    )
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        value=st.secrets.get("PINECONE_API_KEY", ""),
        type="password",
        key="pinecone_api_key_input",
        help="Gratis (tier Starter) di https://www.pinecone.io/",
    )
    st.divider()
    st.caption("Konfigurasi di bawah ini opsional, diisi oleh pengelola app lewat Secrets:")
    webshare_username = st.secrets.get("WEBSHARE_USERNAME", "")
    webshare_password = st.secrets.get("WEBSHARE_PASSWORD", "")
    proxy_host = st.secrets.get("PROXY_HOST", "")
    proxy_port = st.secrets.get("PROXY_PORT", "")
    proxy_config = build_proxy_config(webshare_username, webshare_password, proxy_host, proxy_port)
    if not proxy_config:
        st.caption(
            "⚠️ Proxy belum dikonfigurasi -- transcript kemungkinan gagal "
            "diambil kalau app ini jalan di Streamlit Cloud."
        )

if not google_api_key or not pinecone_api_key:
    st.info(
        "👈 Isi **Google Gemini API Key** dan **Pinecone API Key** kamu di sidebar untuk "
        "mengaktifkan ringkasan otomatis dan tanya-jawab AI.",
        icon="🔑",
    )

ai_ready = bool(google_api_key and pinecone_api_key)

# ---------------------------------------------------------------------------
# Tabs: Proses Video Baru | Riwayat
# ---------------------------------------------------------------------------

tab_new, tab_history = st.tabs(["📼 Proses Video Baru", "🗂️ Riwayat"])

with tab_new:
    url_input = st.text_input(
        "URL atau Video ID YouTube",
        placeholder="https://youtu.be/xxxxxxxxxxx atau xxxxxxxxxxx",
        key="url_input_new",
    )

    col1, col2 = st.columns(2)
    with col1:
        lang_priority = st.text_input(
            "Prioritas bahasa (pisah koma)", value="id,en", key="lang_priority_new"
        )
    with col2:
        video_title = st.text_input(
            "Judul video (opsional, untuk riwayat)",
            placeholder="Kosongkan -> pakai video ID",
            key="title_new",
        )

    fetch_clicked = st.button("Ambil & Proses Transcript", type="primary", use_container_width=True)

    if fetch_clicked:
        if not url_input.strip():
            st.error("Isi URL atau video ID dulu.")
        else:
            preferred_langs = [l.strip() for l in lang_priority.split(",") if l.strip()]

            with st.spinner("Mengambil transcript..."):
                try:
                    result = get_transcript(
                        url_input, preferred_langs=preferred_langs, proxy_config=proxy_config
                    )
                    st.session_state["current_result"] = result
                    st.success(
                        f"Transcript berhasil diambil. Bahasa: **{result.language}** · "
                        f"{result.word_count} kata"
                    )
                except VideoIdError as e:
                    st.error(f"❌ {e}")
                except BlockedError as e:
                    st.error(f"🚫 {e}")
                except NoTranscriptError as e:
                    st.error(f"📭 {e}")
                    try:
                        langs = list_available_languages(url_input, proxy_config=proxy_config)
                        if langs:
                            st.info("Bahasa yang tersedia untuk video ini:")
                            st.table(langs)
                    except TranscriptFetchError:
                        pass
                except VideoNotFoundError as e:
                    st.error(f"🔒 {e}")
                except TranscriptFetchError as e:
                    st.error(f"⚠️ {e}")

    # Kalau transcript sudah berhasil diambil, tampilkan opsi lanjutan
    if "current_result" in st.session_state:
        result = st.session_state["current_result"]

        with st.expander("📄 Lihat transcript mentah"):
            st.text_area("Transcript", value=result.full_text, height=200, key="raw_transcript_display")
            st.download_button(
                "⬇️ Download .txt",
                data=result.full_text,
                file_name=f"transcript_{result.video_id}.txt",
                mime="text/plain",
            )

        st.divider()

        if not ai_ready:
            st.info(
                "Isi Google Gemini API Key dan Pinecone API Key di sidebar untuk mengaktifkan "
                "ringkasan otomatis dan tanya-jawab AI."
            )
        else:
            namespace = ai.video_id_to_namespace(result.video_id)
            already_indexed = ai.namespace_exists(pinecone_api_key, namespace)

            if already_indexed:
                st.info("✅ Video ini sudah pernah diproses sebelumnya -- langsung bisa tanya-jawab di bawah.")
            else:
                if st.button("🔎 Index video ini untuk Q&A", use_container_width=True):
                    with st.spinner("Membuat embedding dan menyimpan ke Pinecone..."):
                        try:
                            ns_info = ai.index_transcript(
                                video_id=result.video_id,
                                full_text=result.full_text,
                                google_api_key=google_api_key,
                                pinecone_api_key=pinecone_api_key,
                            )
                            hs.add_entry(
                                video_id=result.video_id,
                                namespace=ns_info.namespace,
                                title=video_title.strip() or result.video_id,
                                language=result.language,
                                word_count=result.word_count,
                            )
                            st.success(f"Video ter-index ({ns_info.chunk_count} chunk). Siap untuk Q&A.")
                            already_indexed = True
                        except ai.ConfigurationError as e:
                            st.error(f"⚙️ {e}")
                        except ai.IndexingError as e:
                            st.error(f"❌ {e}")

            st.divider()

            # --- Ringkasan ---
            st.subheader("📝 Ringkasan")
            if st.button("Buat Ringkasan", use_container_width=True):
                with st.spinner("Membuat ringkasan (bisa beberapa detik untuk video panjang)..."):
                    try:
                        summary = ai.summarize_transcript(result.full_text, google_api_key)
                        st.session_state["current_summary"] = summary
                    except ai.AIServiceError as e:
                        st.error(f"❌ {e}")

            if "current_summary" in st.session_state:
                st.markdown(st.session_state["current_summary"])

            st.divider()

            # --- Q&A Chat ---
            st.subheader("💬 Tanya soal isi video")
            if not already_indexed:
                st.caption("Index video ini dulu (tombol di atas) sebelum bisa bertanya.")
            else:
                question = st.text_input("Pertanyaan kamu", key="qa_question_input")
                if st.button("Tanya", use_container_width=True) and question.strip():
                    with st.spinner("Mencari jawaban..."):
                        try:
                            qa_result = ai.ask_question(
                                video_id=result.video_id,
                                question=question,
                                google_api_key=google_api_key,
                                pinecone_api_key=pinecone_api_key,
                            )
                            st.markdown(f"**Jawaban:** {qa_result['answer']}")
                            with st.expander("Lihat sumber kutipan dari transcript"):
                                for i, src in enumerate(qa_result["sources"], 1):
                                    st.caption(f"Kutipan {i}:")
                                    st.text(src[:300] + ("..." if len(src) > 300 else ""))
                        except ai.QueryError as e:
                            st.error(f"❌ {e}")
                        except ai.ConfigurationError as e:
                            st.error(f"⚙️ {e}")

with tab_history:
    st.subheader("Video yang sudah diproses")
    st.caption(
        "⚠️ Daftar ini disimpan di penyimpanan lokal app, yang bisa reset saat app "
        "di-redeploy di Streamlit Cloud. Data embedding di Pinecone tetap aman -- "
        "tapi kalau daftar ini kosong padahal kamu yakin sudah pernah proses video, "
        "cukup masukkan lagi video ID/URL yang sama di tab sebelah; sistem akan "
        "mendeteksi video itu sudah ter-index dan tidak akan proses ulang dari nol."
    )

    entries = hs.list_entries()
    if not entries:
        st.info("Belum ada riwayat video yang diproses.")
    else:
        for entry in entries:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{entry.title}**")
                    st.caption(
                        f"Video ID: `{entry.video_id}` · {entry.language} · "
                        f"{entry.word_count} kata · diproses {entry.processed_at[:10]}"
                    )
                with col2:
                    if st.button("Hapus", key=f"del_{entry.video_id}"):
                        hs.delete_entry(entry.video_id)
                        st.rerun()

st.divider()
st.caption(
    "Dibangun dengan youtube-transcript-api, LangChain, Gemini, dan Pinecone. "
    "Belum mendukung Vimeo/platform lain -- menyusul."
)
