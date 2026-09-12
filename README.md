# YouTube Transcript AI Powerhouse

Aplikasi Streamlit yang mengambil transcript video YouTube, lalu memakai AI
untuk membuat ringkasan otomatis, FAQ, tanya-jawab (Q&A), dan laporan PDF
soal isi video -- termasuk video panjang (podcast >1 jam).

**Scope saat ini:** YouTube saja. Platform lain (Vimeo, dll) menyusul.

---

## Arsitektur singkat

```
transcript-app/
├── app.py                    # UI Streamlit -- orkestrasi saja, tanpa logika bisnis
├── transcript_service.py     # Ambil transcript mentah dari YouTube
├── ai_service.py             # Embedding, index ke Pinecone, ringkasan, FAQ, Q&A (RAG)
├── report_builder.py         # Generate laporan PDF (ringkasan+FAQ+transcript+Q&A)
├── history_store.py          # Metadata riwayat video (judul, dll) -- JSON lokal
├── requirements.txt
├── packages.txt              # Apt package (font Unicode untuk PDF) -- Streamlit Cloud
├── .streamlit/
│   └── secrets.toml.example  # Template -- BUKAN secret asli
├── .gitignore
└── README.md
```

**Dua provider AI, dua tujuan berbeda:**
- **Gemini** -- HANYA untuk embedding (index_transcript, ask_question). Video
  yang sudah di-index sebelumnya pakai vector Gemini 768 dimensi; ganti
  provider embedding berarti semua video lama harus di-index ulang.
- **Groq** (Llama/GPT-OSS, gratis) -- untuk ringkasan, FAQ, dan jawaban Q&A.
  Awalnya pakai Gemini juga, tapi dipindah karena free tier Gemini gampang
  kena rate limit (429); Groq jauh lebih longgar dan cepat.

**Kenapa Q&A pakai RAG (retrieval), bukan "kirim semua transcript ke prompt"?**
Untuk video panjang (podcast 1-2 jam), transcript bisa 15.000-25.000+ kata.
Mengirim semuanya setiap kali user bertanya itu boros token dan lebih lambat.
Dengan RAG: transcript dipecah jadi chunk kecil, di-embed, disimpan di
Pinecone -- saat user bertanya, cuma chunk yang paling relevan yang
diambil dan dikirim ke Groq. Untuk **ringkasan** dan **FAQ**, sebaliknya,
seluruh transcript dikirim langsung (context window model Groq default
128K+ token, cukup untuk sebagian besar video) -- tidak perlu RAG.

---

## ⚠️ Batasan yang perlu dipahami

### 1. Proxy wajib untuk deploy publik (soal YouTube blocking)
YouTube memblokir request dari IP cloud provider (termasuk Streamlit
Cloud). Solusi paling reliable: proxy **residential rotating** berbayar
(mis. Webshare, ~$6/bulan) -- ini isi `WEBSHARE_USERNAME`/`WEBSHARE_PASSWORD`
saja di Secrets, TANPA `PROXY_HOST`/`PROXY_PORT`.

**Kalau yang kamu punya adalah paket IP statis** (mis. "Proxy List" di
Webshare, atau provider lain manapun) -- isi juga `PROXY_HOST`/`PROXY_PORT`
di Secrets. IP statis TIDAK ikut rotasi otomatis, jadi lebih gampang
ke-block YouTube dibanding rotating asli, tapi tetap lebih baik daripada
tanpa proxy sama sekali. Kalau IP itu ke-block, ganti langsung dari tab
**"🌐 Proxy"** di app (tersimpan permanen di Pinecone, lihat bagian
"Cara pakai" di bawah) -- tidak perlu balik ke Secrets tiap kali ganti IP.

### 2. Riwayat video (`video_history.json`) TIDAK permanen di Streamlit Cloud
Streamlit Cloud tidak punya disk storage yang persisten -- file JSON ini
bisa hilang saat app di-redeploy atau "sleep" lalu bangun lagi.

**Yang tetap aman:** isi transcript dalam bentuk embedding di Pinecone
tidak akan hilang (itu penyimpanan cloud eksternal, bukan disk lokal
Streamlit) -- begitu juga override IP proxy (lihat batasan #1). Kalau
daftar riwayat di tab "Riwayat" kosong padahal kamu yakin sudah pernah
proses video tertentu, cukup masukkan lagi video ID/URL yang sama --
sistem akan mendeteksi video itu sudah ter-index (lewat namespace di
Pinecone) dan langsung bisa Q&A tanpa index ulang.

### 3. Model AI berubah cepat -- cek `ai_service.py` kalau ada error "model not found"
Provider AI sering mematikan/mengganti model lama dalam hitungan bulan.
Nama model TIDAK di-hardcode tersebar di kode -- semua terpusat di dua
konstanta di awal `ai_service.py`:
```python
DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"        # Groq
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"  # Gemini
```
Kalau muncul error model deprecated/not found: untuk Groq cek
[console.groq.com/docs/models](https://console.groq.com/docs/models),
untuk Gemini cek [daftar model Gemini](https://ai.google.dev/gemini-api/docs/models),
lalu update baris yang relevan saja.

### 4. Laporan PDF butuh font Unicode di server (sudah diatur, tapi perlu tahu)
Hasil ringkasan/FAQ/Q&A dari LLM sering pakai tanda kutip pintar ("smart
quotes") atau em dash yang butuh font Unicode supaya tidak tampil sebagai
"?" di PDF. `packages.txt` di repo ini sudah berisi `fonts-dejavu-core`
supaya Streamlit Cloud otomatis install font itu lewat apt. Development
lokal (terutama Mac) biasanya tidak punya font ini di path yang dicari
`report_builder.py` -- PDF tetap jadi, cuma karakter spesial itu jadi "?".

---

## Setup Lokal (development)

```bash
git clone <repo-kamu>
cd transcript-app
pip install -r requirements.txt
```

### A. Setup Proxy -- untuk ambil transcript

1. Daftar di provider proxy pilihan (mis. [webshare.io](https://www.webshare.io/))
2. Kalau bisa, pilih paket **"Residential" rotating** (bukan "Proxy
   Server"/"Static Residential") -- paling reliable bypass blokir YouTube
3. Ambil username & password. Kalau providermu cuma kasih IP statis
   (bukan rotating), catat juga host & port-nya (lihat batasan #1 di atas)

### B. Setup Groq API Key -- untuk ringkasan, FAQ & Q&A

1. Buka [console.groq.com/keys](https://console.groq.com/keys)
2. Login (Google/GitHub), buat API key baru (gratis, rate limit generous)

### C. Setup Gemini API Key -- untuk embedding

1. Buka [Google AI Studio](https://aistudio.google.com/apikey)
2. Buat API key baru (gratis, ada free tier rate-limited)

### D. Setup Pinecone -- untuk penyimpanan embedding & config

1. Daftar di [pinecone.io](https://www.pinecone.io/) (free tier "Starter",
   tidak perlu kartu kredit)
2. Ambil API key dari dashboard Pinecone
3. Index akan **dibuat otomatis oleh aplikasi** saat pertama kali dipakai
   (region dikunci ke `us-east-1` -- ini wajib untuk free tier)

### E. Isi kredensial

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`, isi minimal `GOOGLE_API_KEY`, `GROQ_API_KEY`,
`PINECONE_API_KEY`, dan kredensial proxy (lihat template di file itu untuk
detail `PROXY_HOST`/`PROXY_PORT` opsional).

### F. Jalankan

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
   GOOGLE_API_KEY = "google-api-key-asli"
   GROQ_API_KEY = "groq-api-key-asli"
   PINECONE_API_KEY = "pinecone-api-key-asli"
   WEBSHARE_USERNAME = "username-proxy-asli"
   WEBSHARE_PASSWORD = "password-proxy-asli"
   # Opsional -- hanya kalau proxy kamu IP statis, lihat batasan #1
   PROXY_HOST = "ip-proxy-asli"
   PROXY_PORT = "port-proxy-asli"
   ```
5. **Save**. App restart otomatis dan langsung baca secrets ini.

Semua key di atas dikelola pengelola app lewat Secrets -- **pengunjung app
tidak pernah melihat atau mengisi API key apa pun** di UI.

---

## Cara pakai

1. Tab **"📼 Video Baru"** → paste URL/ID video YouTube → **Ambil Transcript**
2. (Opsional) isi Judul & Narasumber -- Narasumber jadi nama file saat download
3. Klik **"🔎 Index untuk Q&A"** (sekali per video -- kalau video sudah
   pernah di-index sebelumnya, app otomatis mendeteksi dan skip langkah ini)
4. Klik **"Buat Ringkasan"** untuk ringkasan otomatis
5. Klik **"Buat FAQ"** untuk daftar pertanyaan-jawaban otomatis (maks. 10)
6. Ketik pertanyaan sendiri di **"💬 Tanya Jawab"** untuk Q&A bebas
7. Klik **"Buat PDF"** → **"⬇️ Download PDF"** untuk laporan lengkap
   (ringkasan + FAQ + transcript + riwayat tanya-jawab)
8. Tab **"🗂️ Riwayat"** → lihat semua video yang pernah diproses
9. Tab **"🌐 Proxy"** → ganti IP proxy aktif kapan saja (kalau IP yang
   dipakai ke-block YouTube) tanpa perlu ubah Secrets

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
| `BlockedError` saat ambil transcript | IP diblokir YouTube | Pastikan proxy sudah dikonfigurasi di Secrets; kalau sudah ada tapi tetap gagal, ganti IP lewat tab "Proxy" |
| `ConfigurationError` | API key Gemini/Groq/Pinecone kosong | Cek Secrets sudah lengkap dan nama key persis sama (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`) |
| `IndexingError` | Gagal simpan embedding ke Pinecone | Cek Pinecone API key valid, dan region `us-east-1` (dikunci di kode untuk free tier) |
| `QueryError: belum di-index` | Coba tanya sebelum klik "Index untuk Q&A" | Klik tombol index dulu |
| Groq error "model not found"/"decommissioned" | Groq mematikan/mengganti lineup model | Update `DEFAULT_CHAT_MODEL` di `ai_service.py` -- cek [console.groq.com/docs/models](https://console.groq.com/docs/models) |
| FAQ gagal, error "tool_use_failed"/JSON tidak valid | Model kesulitan structured output lewat function-calling | Sudah ditangani lewat `method="json_schema"` (constrained decoding) di `generate_faq()` -- kalau masih terjadi, cek model yang dipakai mendukung Groq Structured Output API |
| Karakter aneh ("?") di PDF | Font Unicode tidak ketemu di server | Pastikan `packages.txt` (`fonts-dejavu-core`) ikut ke-push ke GitHub -- Streamlit Cloud install otomatis saat deploy |
| Riwayat kosong padahal sudah pernah proses | Streamlit Cloud reset storage lokal | Masukkan ulang video ID yang sama -- sistem deteksi otomatis lewat Pinecone namespace |
