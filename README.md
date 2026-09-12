# YouTube Transcript AI Powerhouse

Aplikasi Streamlit yang mengambil transcript video YouTube, lalu memakai AI
(Gemini + LangChain + Pinecone) untuk membuat ringkasan otomatis dan
menjawab pertanyaan soal isi video -- termasuk video panjang (podcast >1 jam).

**Scope saat ini:** YouTube saja. Platform lain (Vimeo, dll) menyusul.

---

## Arsitektur singkat

```
transcript-app/
├── app.py                    # UI Streamlit -- orkestrasi saja, tanpa logika bisnis
├── transcript_service.py     # Ambil transcript mentah dari YouTube
├── ai_service.py             # Embedding, index ke Pinecone, ringkasan, Q&A (RAG)
├── history_store.py          # Metadata riwayat video (judul, dll) -- JSON lokal
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example  # Template -- BUKAN secret asli
├── .gitignore
└── README.md
```

**Kenapa Q&A pakai RAG (retrieval), bukan "kirim semua transcript ke prompt"?**
Untuk video panjang (podcast 1-2 jam), transcript bisa 15.000-25.000+ kata.
Mengirim semuanya setiap kali user bertanya itu boros token dan lebih lambat.
Dengan RAG: transcript dipecah jadi chunk kecil, di-embed, disimpan di
Pinecone -- saat user bertanya, cuma chunk yang paling relevan yang
diambil dan dikirim ke Gemini. Untuk **ringkasan**, sebaliknya, seluruh
transcript memang dikirim langsung (context window Gemini Flash cukup
besar untuk ini) -- tidak perlu RAG untuk task ini.

---

## ⚠️ Batasan yang perlu dipahami

### 1. Proxy wajib untuk deploy publik (soal YouTube blocking)
YouTube memblokir request dari IP cloud provider (termasuk Streamlit
Cloud). Solusi yang terbukti reliable: proxy **residential** berbayar
(Webshare, ~$6/bulan). Free tier proxy TIDAK cukup -- dikonfirmasi
langsung oleh maintainer `youtube-transcript-api`.

### 2. Riwayat video (`video_history.json`) TIDAK permanen di Streamlit Cloud
Streamlit Cloud tidak punya disk storage yang persisten -- file JSON ini
bisa hilang saat app di-redeploy atau "sleep" lalu bangun lagi.

**Yang tetap aman:** isi transcript dalam bentuk embedding di Pinecone
tidak akan hilang (itu penyimpanan cloud eksternal, bukan disk lokal
Streamlit). Kalau daftar riwayat di tab "Riwayat" kosong padahal kamu
yakin sudah pernah proses video tertentu, cukup masukkan lagi video
ID/URL yang sama -- sistem akan mendeteksi video itu sudah ter-index
(lewat namespace di Pinecone) dan langsung bisa Q&A tanpa index ulang.

**Kalau riwayat permanen benar-benar penting buatmu**, upgrade path yang
belum diimplementasikan di v1 ini: simpan `video_history.json` ke storage
eksternal juga (Google Sheets API, Supabase free tier, atau bahkan simpan
sebagai satu vector dummy di Pinecone namespace terpisah).

### 3. Model AI berubah cepat -- cek `ai_service.py` kalau ada error "model not found"
Google sering mematikan model Gemini lama (misal `gemini-2.5-flash` akan
shutdown Oktober 2026). Nama model TIDAK di-hardcode tersebar di kode --
semua terpusat di dua konstanta di awal `ai_service.py`:
```python
DEFAULT_CHAT_MODEL = "gemini-flash-latest"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
```
Kalau muncul error model deprecated, cek
[daftar model terbaru](https://ai.google.dev/gemini-api/docs/models) dan
update dua baris ini saja.

---

## Setup Lokal (development)

```bash
git clone <repo-kamu>
cd transcript-app
pip install -r requirements.txt
```

### A. Setup Proxy (Webshare) -- untuk ambil transcript

1. Daftar di [webshare.io](https://www.webshare.io/)
2. Beli paket **"Residential"** (bukan "Proxy Server"/"Static Residential")
3. Ambil username & password dari dashboard Webshare

### B. Setup Gemini API Key -- untuk ringkasan & Q&A

1. Buka [Google AI Studio](https://aistudio.google.com/apikey)
2. Buat API key baru (gratis, ada free tier rate-limited)

### C. Setup Pinecone -- untuk penyimpanan embedding

1. Daftar di [pinecone.io](https://www.pinecone.io/) (free tier "Starter",
   tidak perlu kartu kredit)
2. Ambil API key dari dashboard Pinecone
3. Index akan **dibuat otomatis oleh aplikasi** saat pertama kali dipakai
   (region dikunci ke `us-east-1` -- ini wajib untuk free tier)

### D. Isi kredensial

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`, isi keempat nilai:
```toml
WEBSHARE_USERNAME = "..."
WEBSHARE_PASSWORD = "..."
GOOGLE_API_KEY = "..."
PINECONE_API_KEY = "..."
```

### E. Jalankan

```bash
streamlit run app.py
```

---

## Setup di Streamlit Cloud (deploy publik)

1. Push repo ke GitHub (pastikan `secrets.toml` dan `video_history.json`
   **tidak** ikut ke-push -- cek `git status`, keduanya sudah di-ignore).
2. Deploy lewat [share.streamlit.io](https://share.streamlit.io).
3. Buka app kamu → **⋮** (titik tiga) → **Settings** → tab **Secrets**.
4. Paste (isi dengan nilai asli kamu):
   ```toml
   WEBSHARE_USERNAME = "username-asli"
   WEBSHARE_PASSWORD = "password-asli"
   GOOGLE_API_KEY = "google-api-key-asli"
   PINECONE_API_KEY = "pinecone-api-key-asli"
   ```
5. **Save**. App restart otomatis dan langsung baca secrets ini.

---

## Cara pakai

1. Tab **"Proses Video Baru"** → paste URL/ID video YouTube → **Ambil & Proses Transcript**
2. (Opsional) isi judul video supaya gampang dikenali di riwayat
3. Klik **"Index video ini untuk Q&A"** (sekali per video -- kalau video
   sudah pernah di-index sebelumnya, app otomatis mendeteksi dan skip langkah ini)
4. Klik **"Buat Ringkasan"** untuk ringkasan otomatis
5. Ketik pertanyaan di kolom **"Tanya soal isi video"** untuk Q&A
6. Tab **"Riwayat"** → lihat semua video yang pernah diproses

---

## Roadmap (belum dikerjakan)

- [ ] Support Vimeo
- [ ] ~~Support Netflix~~ — tidak akan dikerjakan (DRM, bukan keterbatasan kode)
- [ ] Riwayat yang benar-benar permanen (lihat batasan #2 di atas)
- [ ] Multi-turn conversation (chat history tersimpan per video, bukan single-shot Q&A)
- [ ] Batch processing banyak video sekaligus

---

## Troubleshooting

| Error | Penyebab | Solusi |
|---|---|---|
| `BlockedError` saat ambil transcript | IP diblokir YouTube | Pastikan proxy Webshare "Residential" sudah dikonfigurasi |
| `ConfigurationError` | API key Gemini/Pinecone kosong | Cek Secrets sudah lengkap dan nama key persis sama (`GOOGLE_API_KEY`, `PINECONE_API_KEY`) |
| `IndexingError` | Gagal simpan embedding ke Pinecone | Cek Pinecone API key valid, dan region `us-east-1` (dikunci di kode untuk free tier) |
| `QueryError: belum di-index` | Coba tanya sebelum klik "Index video ini" | Klik tombol index dulu |
| Model Gemini error "not found"/"deprecated" | Google mematikan model lama | Update `DEFAULT_CHAT_MODEL`/`DEFAULT_EMBEDDING_MODEL` di `ai_service.py` |
| Riwayat kosong padahal sudah pernah proses | Streamlit Cloud reset storage lokal | Masukkan ulang video ID yang sama -- sistem deteksi otomatis lewat Pinecone namespace |
