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
# Konfigurasi -- semuanya dari Streamlit Secrets (persisten, dikelola
# pengelola app lewat App settings -> Secrets). Tidak ada input key di UI:
# pengunjung app tidak pernah melihat atau mengisi API key apa pun.
# Satu-satunya nilai yang wajar diubah dari waktu ke waktu adalah
# PROXY_HOST/PROXY_PORT (kalau IP proxy yang dipakai ke-block YouTube).
# ---------------------------------------------------------------------------

google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")

webshare_username = st.secrets.get("WEBSHARE_USERNAME", "")
webshare_password = st.secrets.get("WEBSHARE_PASSWORD", "")

# IP & port proxy: kalau ada override tersimpan di Pinecone (diisi lewat tab
# "Proxy"), itu yang dipakai -- permanen sampai diganti/direset, tidak
# hilang saat app di-redeploy/sleep. Kalau belum pernah di-override, fallback
# ke PROXY_HOST/PROXY_PORT di Secrets.
proxy_override = ai.get_proxy_override(pinecone_api_key) if pinecone_api_key else None
if proxy_override:
    proxy_host = proxy_override["proxy_host"]
    proxy_port = proxy_override["proxy_port"]
else:
    proxy_host = st.secrets.get("PROXY_HOST", "")
    proxy_port = st.secrets.get("PROXY_PORT", "")

proxy_config = build_proxy_config(webshare_username, webshare_password, proxy_host, proxy_port)

with st.sidebar:
    st.subheader("⚙️ Status Konfigurasi")
    st.caption("Diisi pengelola app lewat Secrets -- bukan sesuatu yang perlu diisi pengunjung.")
    st.write("🔑 Google Gemini:", "✅ siap" if google_api_key else "❌ belum diisi")
    st.write("🔑 Pinecone:", "✅ siap" if pinecone_api_key else "❌ belum diisi")
    st.write("🌐 Proxy:", "✅ siap" if proxy_config else "❌ belum diisi")

missing_warnings = []
if not proxy_config:
    missing_warnings.append(
        "**Proxy** belum dikonfigurasi -- transcript kemungkinan gagal diambil kalau app "
        "ini jalan di Streamlit Cloud."
    )
if not google_api_key:
    missing_warnings.append("**GOOGLE_API_KEY** belum diisi -- fitur ringkasan & Q&A tidak akan jalan.")
if not pinecone_api_key:
    missing_warnings.append("**PINECONE_API_KEY** belum diisi -- fitur ringkasan & Q&A tidak akan jalan.")

if missing_warnings:
    with st.expander("⚠️ Ada konfigurasi yang belum lengkap", expanded=True):
        for w in missing_warnings:
            st.warning(w, icon="⚠️")
        st.caption("Lihat README.md bagian 'Setup' untuk cara mengisi semua ini lewat Streamlit Secrets.")

ai_ready = bool(google_api_key and pinecone_api_key)

# ---------------------------------------------------------------------------
# Tabs: Proses Video Baru | Riwayat | Proxy
# ---------------------------------------------------------------------------

tab_new, tab_history, tab_proxy = st.tabs(["📼 Proses Video Baru", "🗂️ Riwayat", "🌐 Proxy"])

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
                "GOOGLE_API_KEY dan PINECONE_API_KEY belum diisi di Secrets -- ringkasan "
                "otomatis dan tanya-jawab AI belum aktif."
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

with tab_proxy:
    st.subheader("Ganti IP Proxy")
    st.caption(
        "Username & password proxy tetap dari Secrets (WEBSHARE_USERNAME/PASSWORD) -- "
        "cuma IP & port yang diganti di sini. Perubahan disimpan **permanen** di Pinecone, "
        "jadi tidak hilang walau app di-redeploy atau 'sleep' lalu bangun lagi."
    )

    if not pinecone_api_key:
        st.warning(
            "PINECONE_API_KEY belum diisi di Secrets -- fitur ganti proxy butuh Pinecone "
            "untuk menyimpan perubahan secara permanen.",
            icon="⚠️",
        )
    else:
        if proxy_override:
            st.success(
                f"IP aktif saat ini: **{proxy_override['proxy_host']}:{proxy_override['proxy_port']}** "
                f"(di-set manual, terakhir diubah {proxy_override['updated_at'][:19].replace('T', ' ')} UTC)"
            )
        elif proxy_host and proxy_port:
            st.info(f"IP aktif saat ini: **{proxy_host}:{proxy_port}** (default dari Secrets)")
        else:
            st.warning("Belum ada proxy yang dikonfigurasi sama sekali.", icon="⚠️")

        with st.form("proxy_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_host = st.text_input("IP Proxy", value=proxy_host, placeholder="mis. 150.241.118.80")
            with col2:
                new_port = st.text_input("Port", value=proxy_port, placeholder="mis. 6082")
            submitted = st.form_submit_button("💾 Simpan sebagai IP aktif", type="primary", use_container_width=True)

        if submitted:
            if not new_host.strip() or not new_port.strip():
                st.error("IP dan port tidak boleh kosong.")
            else:
                try:
                    ai.save_proxy_override(pinecone_api_key, new_host.strip(), new_port.strip())
                    st.success(f"Tersimpan. IP aktif sekarang: {new_host.strip()}:{new_port.strip()}")
                    st.rerun()
                except (ai.ConfigurationError, ai.IndexingError) as e:
                    st.error(f"❌ {e}")

        if proxy_override and st.button("↩️ Reset ke default Secrets (PROXY_HOST/PROXY_PORT)"):
            try:
                ai.delete_proxy_override(pinecone_api_key)
                st.success("Override dihapus, kembali ke default Secrets.")
                st.rerun()
            except ai.IndexingError as e:
                st.error(f"❌ {e}")

st.divider()
st.caption(
    "Dibangun dengan youtube-transcript-api, LangChain, Gemini, dan Pinecone. "
    "Belum mendukung Vimeo/platform lain -- menyusul."
)
