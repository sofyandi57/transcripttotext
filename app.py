"""
app.py -- Streamlit UI untuk YouTube Transcript Grabber + AI Q&A.

Arsitektur:
  transcript_service.py -> ambil transcript mentah dari YouTube
  ai_service.py          -> embedding, index ke Pinecone, ringkasan, Q&A
  history_store.py       -> metadata riwayat video (judul, dll)
  report_builder.py      -> generate laporan PDF (ringkasan+transcript+Q&A)
  app.py (file ini)      -> UI & orkestrasi, tidak ada logika bisnis di sini
"""

import re
from datetime import date

import streamlit as st

import ai_service as ai
import history_store as hs
import report_builder
from transcript_service import (
    BlockedError,
    NoTranscriptError,
    TranscriptFetchError,
    VideoIdError,
    VideoNotFoundError,
    build_proxy_config,
    extract_video_id,
    format_transcript_with_timestamps,
    get_transcript,
    get_video_metadata,
    list_available_languages,
)


def _build_download_filename(narasumber: str, video_id: str, upload_date: str = "", ext: str = "txt") -> str:
    """
    Nama file download: narasumber_tanggal.ext. Kalau narasumber kosong,
    fallback ke video_id supaya tetap unik. Tanggal yang dipakai adalah
    tanggal upload ASLI video (dari YouTube Data API, kalau tersedia) --
    fallback ke tanggal proses hari ini kalau YOUTUBE_API_KEY tidak diisi
    atau lookup-nya gagal.
    """
    slug_source = narasumber.strip() or video_id
    slug = re.sub(r"[^\w\-]+", "_", slug_source).strip("_") or video_id
    tanggal = upload_date or date.today().isoformat()
    return f"{slug}_{tanggal}.{ext}"


st.set_page_config(page_title="Transcript AI Powerhouse", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Gate password -- sederhana, satu password bersama disimpan di Secrets
# (APP_PASSWORD), bukan di kode. Kalau APP_PASSWORD tidak diisi, gate ini
# dilewati sama sekali (app terbuka seperti biasa) -- supaya tidak
# mengunci diri sendiri kalau lupa/belum sempat setup.
# ---------------------------------------------------------------------------

app_password = st.secrets.get("APP_PASSWORD", "")

if app_password and not st.session_state.get("authenticated"):
    st.title("🔒 Login")
    with st.form("login_form"):
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk", type="primary", use_container_width=True)
    if submitted:
        if password_input == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Password salah.")
    st.stop()

st.title("🧠 YouTube Transcript AI Powerhouse")
st.caption("Transcript, ringkasan, dan tanya-jawab video YouTube.")

# ---------------------------------------------------------------------------
# Konfigurasi -- semuanya dari Streamlit Secrets (persisten, dikelola
# pengelola app lewat App settings -> Secrets). Tidak ada input key di UI:
# pengunjung app tidak pernah melihat atau mengisi API key apa pun.
# Satu-satunya nilai yang wajar diubah dari waktu ke waktu adalah
# PROXY_HOST/PROXY_PORT (kalau IP proxy yang dipakai ke-block YouTube).
# ---------------------------------------------------------------------------

google_api_key = st.secrets.get("GOOGLE_API_KEY", "")   # embedding saja (Pinecone)
groq_api_key = st.secrets.get("GROQ_API_KEY", "")        # ringkasan & Q&A
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
youtube_api_key = st.secrets.get("YOUTUBE_API_KEY", "")  # opsional -- judul/metadata otomatis

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
    st.subheader("⚙️ Status")
    st.write("Gemini:", "✅" if google_api_key else "❌")
    st.write("Groq:", "✅" if groq_api_key else "❌")
    st.write("Pinecone:", "✅" if pinecone_api_key else "❌")
    st.write("Proxy:", "✅" if proxy_config else "❌")
    st.write("YouTube Data API:", "✅" if youtube_api_key else "➖ opsional")

missing_warnings = []
if not proxy_config:
    missing_warnings.append("Proxy belum diisi.")
if not google_api_key:
    missing_warnings.append("GOOGLE_API_KEY belum diisi.")
if not groq_api_key:
    missing_warnings.append("GROQ_API_KEY belum diisi.")
if not pinecone_api_key:
    missing_warnings.append("PINECONE_API_KEY belum diisi.")

if missing_warnings:
    with st.expander("⚠️ Konfigurasi belum lengkap", expanded=True):
        for w in missing_warnings:
            st.warning(w, icon="⚠️")
        st.caption("Isi lewat Streamlit Secrets -- lihat README.")

ai_ready = bool(google_api_key and groq_api_key and pinecone_api_key)

# ---------------------------------------------------------------------------
# Tabs: Proses Video Baru | Riwayat | Proxy
# ---------------------------------------------------------------------------

tab_new, tab_history, tab_proxy = st.tabs(["📼 Video Baru", "🗂️ Riwayat", "🌐 Proxy"])

with tab_new:
    # Field teks pakai key "berversi" (form_version) supaya tombol Clear
    # bisa reset tampilannya secara pasti -- session_state.pop(key) + rerun()
    # saja TIDAK selalu cukup untuk text_input di Streamlit (widget frontend
    # kadang tetap menampilkan value lama walau state Python-nya sudah
    # dihapus). Ganti key -> Streamlit menganggapnya widget baru -> pasti kosong.
    if "form_version" not in st.session_state:
        st.session_state["form_version"] = 0
    fv = st.session_state["form_version"]

    url_input = st.text_input(
        "URL / ID Video",
        placeholder="https://youtu.be/xxxxxxxxxxx",
        key=f"url_input_new_{fv}",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        lang_priority = st.text_input(
            "Bahasa",
            value="id,en",
            key=f"lang_priority_new_{fv}",
            help="Prioritas bahasa transcript, pisahkan dengan koma",
        )
    with col2:
        video_title = st.text_input(
            "Judul",
            placeholder="Untuk riwayat",
            key=f"title_new_{fv}",
            help="Opsional -- kosongkan untuk pakai video ID",
        )
    with col3:
        narasumber = st.text_input(
            "Narasumber",
            placeholder="mis. Michele Yeoh",
            key=f"narasumber_new_{fv}",
            help="Opsional -- jadi nama file saat download",
        )

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        fetch_clicked = st.button("Ambil Transcript", type="primary", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("current_result", None)
        st.session_state.pop("current_summary", None)
        st.session_state.pop("current_faq", None)
        st.session_state.pop("current_pdf", None)
        st.session_state.pop("current_metadata", None)
        st.session_state.pop("qa_history", None)
        st.session_state.pop(f"qa_question_input_{fv}", None)
        st.session_state["form_version"] = fv + 1
        st.rerun()

    if fetch_clicked:
        # Bersihkan hasil video sebelumnya dulu -- supaya kalau fetch video baru
        # ini gagal, transcript/ringkasan video LAMA tidak nyangkut kelihatan
        # seolah-olah itu punya video yang baru saja dicoba.
        st.session_state.pop("current_result", None)
        st.session_state.pop("current_summary", None)
        st.session_state.pop("current_faq", None)
        st.session_state.pop("current_pdf", None)
        st.session_state.pop("current_metadata", None)

        if not url_input.strip():
            st.error("Isi URL/ID video dulu.")
        else:
            preferred_langs = [l.strip() for l in lang_priority.split(",") if l.strip()]

            with st.spinner("Mengambil transcript..."):
                try:
                    result = get_transcript(
                        url_input, preferred_langs=preferred_langs, proxy_config=proxy_config
                    )
                    st.session_state["current_result"] = result
                    st.session_state["current_metadata"] = get_video_metadata(
                        result.video_id, youtube_api_key
                    )
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
                            st.info("Bahasa tersedia:")
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
        metadata = st.session_state.get("current_metadata")

        # Judul efektif: input manual > judul otomatis dari YouTube > video_id.
        # Dipakai konsisten untuk riwayat, nama file, dan laporan PDF.
        effective_title = video_title.strip() or (metadata.title if metadata else "") or result.video_id

        if metadata:
            st.markdown(f"**{metadata.title}**")
            st.caption(
                f"{metadata.channel_title} · {metadata.duration_display} · "
                f"{metadata.view_count:,} views · diupload {metadata.published_at}".replace(",", ".")
            )
            if metadata.description:
                with st.expander("Deskripsi video"):
                    st.text(metadata.description)

        with st.expander("📄 Transcript"):
            # Key di-per-video (bukan statis) -- supaya widget selalu benar-benar
            # baru saat ganti video, dan tidak ada risiko menampilkan isi transcript
            # video SEBELUMNYA gara-gara Streamlit menganggap ini widget yang sama.
            # Preview pakai versi BERTIMESTAMP juga -- supaya sama persis dengan
            # isi file yang di-download, tidak bikin bingung ("kok di preview
            # nggak ada timestamp-nya, di file kok ada").
            transcript_with_timestamps = format_transcript_with_timestamps(result.segments)
            st.text_area(
                "Transcript",
                value=transcript_with_timestamps,
                height=200,
                key=f"raw_transcript_display_{result.video_id}",
            )
            download_name = _build_download_filename(
                narasumber, result.video_id, upload_date=metadata.published_at if metadata else ""
            )
            st.download_button(
                "⬇️ Download .txt",
                data=transcript_with_timestamps,
                file_name=download_name,
                mime="text/plain",
                help="File berisi timestamp [mm:ss] per baris.",
            )

        st.divider()

        if not ai_ready:
            st.info("Ringkasan & Q&A belum aktif -- API key belum diisi.")
        else:
            namespace = ai.video_id_to_namespace(result.video_id)
            already_indexed = ai.namespace_exists(pinecone_api_key, namespace)

            if already_indexed:
                st.info("✅ Sudah ter-index -- langsung bisa tanya-jawab.")
            else:
                if st.button("🔎 Index untuk Q&A", use_container_width=True):
                    with st.spinner("Membuat embedding..."):
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
                                title=effective_title,
                                language=result.language,
                                word_count=result.word_count,
                            )
                            st.success(f"Ter-index ({ns_info.chunk_count} chunk).")
                            already_indexed = True
                        except ai.ConfigurationError as e:
                            st.error(f"⚙️ {e}")
                        except ai.IndexingError as e:
                            st.error(f"❌ {e}")

            st.divider()

            # --- Ringkasan ---
            st.subheader("📝 Ringkasan")
            if st.button("Buat Ringkasan", use_container_width=True):
                with st.spinner("Membuat ringkasan..."):
                    try:
                        summary = ai.summarize_transcript(result.full_text, groq_api_key)
                        st.session_state["current_summary"] = summary
                    except ai.AIServiceError as e:
                        st.error(f"❌ {e}")

            if "current_summary" in st.session_state:
                st.markdown(st.session_state["current_summary"])

            st.divider()

            # --- FAQ otomatis ---
            st.subheader("❓ FAQ")
            if st.button("Buat FAQ", use_container_width=True):
                with st.spinner("Membuat FAQ..."):
                    try:
                        st.session_state["current_faq"] = ai.generate_faq(result.full_text, groq_api_key)
                    except ai.AIServiceError as e:
                        st.error(f"❌ {e}")

            if "current_faq" in st.session_state:
                for item in st.session_state["current_faq"]:
                    with st.expander(item["question"]):
                        st.markdown(item["answer"])

            st.divider()

            # --- Q&A Chat ---
            st.subheader("💬 Tanya Jawab")
            qa_history = st.session_state.setdefault("qa_history", {}).setdefault(result.video_id, [])

            if not already_indexed:
                st.caption("Index dulu sebelum bertanya.")
            else:
                question = st.text_input("Pertanyaan", key=f"qa_question_input_{fv}")
                if st.button("Tanya", use_container_width=True) and question.strip():
                    with st.spinner("Mencari jawaban..."):
                        try:
                            qa_result = ai.ask_question(
                                video_id=result.video_id,
                                question=question,
                                google_api_key=google_api_key,
                                groq_api_key=groq_api_key,
                                pinecone_api_key=pinecone_api_key,
                            )
                            qa_history.append(
                                {
                                    "question": question,
                                    "answer": qa_result["answer"],
                                    "sources": qa_result["sources"],
                                }
                            )
                        except ai.QueryError as e:
                            st.error(f"❌ {e}")
                        except ai.ConfigurationError as e:
                            st.error(f"⚙️ {e}")

                for qa in reversed(qa_history):
                    st.markdown(f"**Q: {qa['question']}**")
                    st.markdown(qa["answer"])
                    with st.expander("Sumber kutipan"):
                        for i, src in enumerate(qa["sources"], 1):
                            st.caption(f"Kutipan {i}:")
                            st.text(src[:300] + ("..." if len(src) > 300 else ""))
                    st.divider()

            # --- Laporan PDF ---
            st.subheader("📄 Laporan PDF")
            if st.button("Buat PDF", use_container_width=True):
                pdf_bytes = report_builder.build_pdf_report(
                    title=effective_title,
                    video_id=result.video_id,
                    language=result.language,
                    word_count=result.word_count,
                    metadata=metadata,
                    summary=st.session_state.get("current_summary"),
                    transcript=transcript_with_timestamps,
                    faq_items=st.session_state.get("current_faq", []),
                    qa_history=qa_history,
                )
                st.session_state["current_pdf"] = pdf_bytes

            if "current_pdf" in st.session_state:
                st.download_button(
                    "⬇️ Download PDF",
                    data=st.session_state["current_pdf"],
                    file_name=_build_download_filename(
                        narasumber, result.video_id, upload_date=metadata.published_at if metadata else "", ext="pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )

with tab_history:
    st.subheader("Video Diproses")
    st.caption("Bisa reset saat redeploy -- data Pinecone tetap aman.", help="Kalau daftar kosong padahal video pernah diproses, masukkan lagi video-nya -- sistem deteksi otomatis dari Pinecone, tidak proses ulang dari nol.")

    entries = hs.list_entries()
    if not entries:
        st.info("Belum ada riwayat.")
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
    st.subheader("Mode Proxy")
    st.caption(
        "Username & password tetap dari Secrets.",
        help="Kosongkan IP & Port di bawah untuk pakai mode ROTATING asli Webshare "
        "(direkomendasikan) -- IP otomatis berganti tiap request, auto-retry 10x kalau "
        "kena block. Isi IP & Port hanya kalau mau pin ke satu IP statis tertentu "
        "(tersimpan permanen di Pinecone, tapi TIDAK ikut rotasi otomatis).",
    )

    if not pinecone_api_key:
        st.warning("PINECONE_API_KEY belum diisi -- fitur ini butuh Pinecone.", icon="⚠️")
    elif not webshare_username or not webshare_password:
        st.warning("WEBSHARE_USERNAME/PASSWORD belum diisi -- proxy belum aktif.", icon="⚠️")
    else:
        if proxy_override:
            updated = proxy_override["updated_at"][:19].replace("T", " ")
            st.info(f"Mode **IP statis**: {proxy_override['proxy_host']}:{proxy_override['proxy_port']}")
            st.caption(f"Diubah manual {updated} UTC")
        elif proxy_host and proxy_port:
            st.info(f"Mode **IP statis** (default Secrets): {proxy_host}:{proxy_port}")
        else:
            st.success("Mode **Rotating** aktif -- IP berganti otomatis tiap request.")

        with st.form("proxy_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_host = st.text_input("IP (kosongkan = rotating)", value=proxy_host, placeholder="mis. 150.241.118.80")
            with col2:
                new_port = st.text_input("Port", value=proxy_port, placeholder="6082")
            submitted = st.form_submit_button("💾 Simpan", type="primary", use_container_width=True)

        if submitted:
            if not new_host.strip() and not new_port.strip():
                try:
                    ai.delete_proxy_override(pinecone_api_key)
                    st.success("IP dikosongkan -- kembali ke mode rotating.")
                    st.rerun()
                except ai.IndexingError as e:
                    st.error(f"❌ {e}")
            elif not new_host.strip() or not new_port.strip():
                st.error("Isi IP dan Port berdua, atau kosongkan berdua untuk mode rotating.")
            else:
                try:
                    ai.save_proxy_override(pinecone_api_key, new_host.strip(), new_port.strip())
                    st.success(f"Tersimpan (IP statis): {new_host.strip()}:{new_port.strip()}")
                    st.rerun()
                except (ai.ConfigurationError, ai.IndexingError) as e:
                    st.error(f"❌ {e}")

        if proxy_override and st.button("↩️ Reset ke mode rotating"):
            try:
                ai.delete_proxy_override(pinecone_api_key)
                st.success("Kembali ke mode rotating.")
                st.rerun()
            except ai.IndexingError as e:
                st.error(f"❌ {e}")

st.divider()
st.caption("Powered by youtube-transcript-api, LangChain, Gemini & Pinecone.")
