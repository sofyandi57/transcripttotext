# YouTube Transcript AI Powerhouse

Aplikasi Streamlit yang mengambil transcript video YouTube (termasuk video
**tanpa caption sama sekali**, lewat transkripsi audio otomatis), lalu
memakai AI untuk membuat ringkasan, FAQ, terjemahan, tanya-jawab (Q&A), dan
laporan PDF soal isi video -- termasuk video panjang (podcast >1 jam) dan
video berbahasa apa pun (multi-bahasa, termasuk skrip non-Latin seperti
Thai/Jepang/Korea/Cina/Arab/Persia).

**Scope saat ini:** YouTube saja. Platform lain (Vimeo, dll) menyusul.

**Fitur utama:**
- Ambil transcript YouTube dengan pilihan bahasa (multi-bahasa, termasuk
  skrip non-Latin) -- lihat [Multi-bahasa](#multi-bahasa--terjemahan)
- **Fallback Whisper**: kalau video tidak punya caption sama sekali,
  transkripsi otomatis dari audio via Groq Whisper (opt-in, tombol
  terpisah) -- lihat [Transkripsi via Whisper](#transkripsi-audio-via-whisper-fallback-tanpa-caption)
- Ringkasan & FAQ otomatis (maks. 10 item) dari transcript
- **Terjemahan** transcript penuh ke Indonesia (atau Inggris sebagai
  fallback) untuk video berbahasa asing
- Tanya-jawab bebas (RAG via Pinecone) soal isi video
- Laporan PDF lengkap (ringkasan + FAQ + transcript + terjemahan + riwayat Q&A),
  dengan dukungan font Unicode termasuk RTL (Arab/Persia)
- Metadata video otomatis (judul, channel, durasi, views, tanggal upload)
  via YouTube Data API v3 -- opsional
- Riwayat video dengan hyperlink ke video asli dan download ulang transcript
- Password gate sederhana (opsional) untuk membatasi akses app publik
- Resilient terhadap rate limit Groq (TPM & TPD) -- fallback model otomatis
  + caching supaya tidak boros kuota

---

## Arsitektur singkat

```
transcript-app/
├── app.py                    # UI Streamlit -- orkestrasi saja, tanpa logika bisnis
├── transcript_service.py     # Ambil transcript & metadata (judul, channel, dll) dari YouTube
├── audio_service.py          # Fallback: download audio (yt-dlp) + transkripsi Groq Whisper
├── ai_service.py             # Embedding, index ke Pinecone, ringkasan, FAQ, terjemahan, Q&A (RAG)
├── report_builder.py         # Generate laporan PDF (ringkasan+FAQ+transcript+terjemahan+Q&A)
├── history_store.py          # Metadata riwayat video (judul, dll) -- JSON lokal
├── requirements.txt
├── packages.txt              # Apt package (font Unicode + ffmpeg) -- Streamlit Cloud
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

**Resilient terhadap rate limit Groq.** Ada dua jenis limit Groq: TPM
(token per menit, pulih dalam hitungan detik) dan TPD (token per hari,
pulih setelah beberapa jam). `_invoke_resilient()` di `ai_service.py`
otomatis retry dengan backoff untuk TPM, pindah ke model fallback lain
kalau kena TPD, dan kalau semua model habis kuota -- tampilkan pesan
error yang jelas (bukan JSON mentah dari API). Ringkasan/FAQ/jawaban Q&A
juga di-**cache** per video di riwayat, supaya pertanyaan/aksi yang sama
tidak memanggil Groq berulang kali.

---

## Multi-bahasa & Terjemahan

Transcript bisa diambil dalam bahasa apa pun yang tersedia di video --
termasuk skrip non-Latin (Thai, Jepang, Korea, Cina Sederhana/Tradisional,
Hindi, Tamil, Arab, Persia, dll -- lihat daftar lengkap di dropdown
"Bahasa" pada UI, atau isi kode ISO manual di field "Kode bahasa lain"
untuk bahasa yang belum ada di daftar). Laporan PDF otomatis memilih font
yang sesuai skrip yang terdeteksi (DejaVu Sans untuk Latin/Cyrillic/Yunani/
Turki, Noto Sans untuk Thai/Tamil/Devanagari/CJK/Arab), dan untuk Arab/
Persia teks di-shape dengan benar (huruf tersambung, arah RTL) lewat
`uharfbuzz`.

**Fitur Terjemahan** muncul otomatis di UI kalau bahasa video terdeteksi
BUKAN Indonesia dan BUKAN Inggris. Klik tombol terjemahan untuk
menerjemahkan **seluruh transcript** (bukan cuma ringkasan) ke Bahasa
Indonesia -- kalau model menilai terjemahan ke Indonesia kurang pas untuk
konten tersebut, otomatis fallback ke Inggris. Hasil terjemahan ikut masuk
ke laporan PDF dan di-cache di riwayat video.

---

## Transkripsi Audio via Whisper (fallback tanpa caption)

Sebagian video YouTube tidak punya caption/subtitle sama sekali (auto-generated
maupun manual) -- biasanya video lama, live recording, atau upload dari
channel kecil. Untuk kasus ini, aplikasi menawarkan jalur alternatif:
download audio video (via `yt-dlp`) lalu transkripsi otomatis pakai
**Groq Whisper API** (`whisper-large-v3-turbo`).

**Penting -- ini fitur opt-in, bukan otomatis:**
- Saat "Ambil Transcript" gagal karena video tidak punya caption, muncul
  tombol terpisah **"🎙️ Transcribe via Whisper"** -- transkripsi baru
  jalan kalau tombol ini diklik.
- Ini disengaja: proses download audio penuh + pemakaian kuota Groq
  Whisper jauh lebih mahal (waktu & kuota) dibanding ambil caption biasa,
  jadi harus jadi keputusan sadar dari pengguna, bukan fallback diam-diam.
- Hasil transkripsi Whisper diperlakukan identik dengan transcript biasa
  setelah berhasil -- bisa diringkas, dibuatkan FAQ, di-index untuk Q&A,
  diterjemahkan, dan masuk laporan PDF seperti biasa.
- Video yang sangat panjang otomatis dipecah jadi beberapa bagian (chunk)
  sebelum dikirim ke Groq Whisper, karena ada batas ukuran file per
  request (25MB) -- ini ditangani otomatis, tidak perlu aksi manual.
- Tidak ada fallback ke model Whisper lokal (mis. `faster-whisper`) --
  Streamlit Cloud free tier tidak punya GPU dan CPU-nya terlalu lambat
  untuk transkripsi audio yang layak pakai. Kalau Groq Whisper gagal atau
  kuota habis, error ditampilkan apa adanya.
- Butuh `ffmpeg` di server (sudah ada di `packages.txt`) dan `GROQ_API_KEY`
  yang sama dengan yang dipakai untuk ringkasan/FAQ/Q&A.

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

### 5. Judul/metadata video otomatis butuh API key TERPISAH (opsional)
Judul video, nama channel, durasi, jumlah views, tanggal upload asli, dan
deskripsi diambil lewat **YouTube Data API v3** -- ini API key BEDA dari
Gemini/Groq/Pinecone, dari **Google Cloud Console** (bukan Google AI
Studio). Tanpa `YOUTUBE_API_KEY`, app tetap jalan normal: judul pakai input
manual (atau video ID kalau kosong), nama file download pakai tanggal saat
diproses (bukan tanggal upload asli), dan tidak ada info channel/durasi/views.

### 6. Transkripsi Whisper butuh proxy yang sama & bisa lebih lambat
Download audio (`yt-dlp`) memakai proxy yang sama dengan pengambilan
caption (lihat batasan #1) -- kalau IP server ke-block YouTube untuk
caption, kemungkinan besar juga ke-block untuk download audio. Selain itu,
proses download + transkripsi audio jauh lebih lambat dari ambil caption
(bisa beberapa menit untuk video panjang) dan memakai kuota Groq Whisper
API terpisah dari kuota chat/ringkasan -- lihat
[Transkripsi Audio via Whisper](#transkripsi-audio-via-whisper-fallback-tanpa-caption)
di atas.

### 7. Password gate hanya proteksi ringan (opsional)
Kalau `APP_PASSWORD` diisi di Secrets, app menampilkan halaman login
sederhana sebelum bisa dipakai. Ini bukan sistem autentikasi
production-grade (tidak ada rate-limiting percobaan login, tidak ada
multi-user/role) -- cukup untuk mencegah orang random mengakses app kalau
linknya publik. Kalau `APP_PASSWORD` kosong, app terbuka seperti biasa
tanpa login sama sekali.

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

### E. Setup YouTube Data API Key -- opsional, untuk judul/metadata otomatis

1. Buka [console.cloud.google.com](https://console.cloud.google.com)
2. Buat/pilih project → **APIs & Services** → **Library** → cari
   **"YouTube Data API v3"** → **Enable**
3. **Credentials** → **Create Credentials** → **API Key**
4. Skip langkah ini kalau tidak perlu -- lihat batasan #5 di atas untuk apa
   yang berubah tanpa key ini

### F. Isi kredensial

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`, isi minimal `GOOGLE_API_KEY`, `GROQ_API_KEY`,
`PINECONE_API_KEY`, dan kredensial proxy (lihat template di file itu untuk
detail `PROXY_HOST`/`PROXY_PORT` opsional, dan `YOUTUBE_API_KEY` opsional).

### G. Jalankan

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
   # Opsional -- judul/metadata video otomatis, lihat batasan #5
   YOUTUBE_API_KEY = "youtube-data-api-key-asli"
   # Opsional -- kunci app dengan password sederhana, lihat batasan #7
   APP_PASSWORD = "password-app-pilihanmu"
   ```
5. **Save**. App restart otomatis dan langsung baca secrets ini.

Semua key di atas dikelola pengelola app lewat Secrets -- **pengunjung app
tidak pernah melihat atau mengisi API key apa pun** di UI.

---

## Cara pakai

1. (Opsional) kalau `APP_PASSWORD` diisi, masukkan password dulu di halaman login
2. Tab **"📼 Video Baru"** → pilih bahasa yang diinginkan → paste URL/ID
   video YouTube → **Ambil Transcript** (kalau `YOUTUBE_API_KEY` diisi,
   judul/channel/durasi/views/deskripsi otomatis muncul -- tidak perlu isi
   manual)
   - Kalau video **tidak punya caption sama sekali**, muncul tombol
     **"🎙️ Transcribe via Whisper"** sebagai alternatif -- klik untuk
     transkripsi dari audio (lihat [Transkripsi Audio via Whisper](#transkripsi-audio-via-whisper-fallback-tanpa-caption))
3. (Opsional) isi Judul & Narasumber -- Judul manual selalu menang atas judul
   otomatis kalau diisi; Narasumber jadi nama file saat download. Buka
   expander **"📄 Transcript"** → **"⬇️ Download .txt"** untuk file transcript
   dengan timestamp `[mm:ss]` per baris (nama file pakai tanggal upload asli
   video kalau `YOUTUBE_API_KEY` diisi, kalau tidak pakai tanggal proses)
4. Klik **"🔎 Index untuk Q&A"** (sekali per video -- kalau video sudah
   pernah di-index sebelumnya, app otomatis mendeteksi dan skip langkah ini)
5. Klik **"Buat Ringkasan"** untuk ringkasan otomatis
6. Klik **"Buat FAQ"** untuk daftar pertanyaan-jawaban otomatis (maks. 10)
7. Kalau video berbahasa asing (bukan Indonesia/Inggris), klik tombol
   **"🔤 Terjemahan"** untuk menerjemahkan seluruh transcript
8. Ketik pertanyaan sendiri di **"💬 Tanya Jawab"** untuk Q&A bebas
9. Klik **"Buat PDF"** → **"⬇️ Download PDF"** untuk laporan lengkap
   (ringkasan + FAQ + transcript + terjemahan + riwayat tanya-jawab)
10. Tab **"🗂️ Riwayat"** → lihat semua video yang pernah diproses, buka
    hyperlink ke video asli, atau download ulang transcript-nya
11. Tab **"🌐 Proxy"** → ganti IP proxy aktif kapan saja (kalau IP yang
    dipakai ke-block YouTube) tanpa perlu ubah Secrets

---

## Roadmap (belum dikerjakan)

- [ ] Support Vimeo, X/Twitter, TikTok, dan platform video lain (Whisper
      saat ini masih khusus fallback untuk YouTube tanpa caption)
- [ ] ~~Support Netflix~~ — tidak akan dikerjakan (DRM, bukan keterbatasan kode)
- [ ] Riwayat yang benar-benar permanen (lihat batasan #2 di atas)
- [ ] Multi-turn conversation (chat history tersimpan per video, bukan single-shot Q&A)
- [ ] Batch processing banyak video sekaligus
- [ ] Upload file audio manual (di luar link YouTube) untuk transkripsi Whisper

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
| Ringkasan/FAQ gagal "Request too large ... tokens per minute (TPM)" | Kuota TPM free tier Groq (kecil, rolling per menit) habis karena beberapa aksi AI beruntun | Sudah ditangani otomatis lewat retry+backoff dan map-reduce chunking di `ai_service.py` -- kalau masih gagal setelah 3x retry, tunggu sebentar lalu coba lagi manual |
| Judul/metadata video tidak muncul otomatis | `YOUTUBE_API_KEY` belum diisi, atau lookup gagal (quota habis/video tidak ditemukan) | Cek Secrets ada `YOUTUBE_API_KEY` yang valid -- ini opsional, app tetap jalan tanpanya (fallback ke judul manual & tanggal proses) |
| Riwayat kosong padahal sudah pernah proses | Streamlit Cloud reset storage lokal | Masukkan ulang video ID yang sama -- sistem deteksi otomatis lewat Pinecone namespace |
| `AudioFetchError: Gagal download audio` saat pakai Whisper | yt-dlp gagal akses video (diblokir/private/region-lock), atau proxy sama seperti caption | Cek proxy sudah dikonfigurasi (lihat batasan #1 & #6); coba ganti IP proxy lewat tab "Proxy" |
| `AudioFetchError: Transkripsi Whisper gagal` | Groq Whisper API error (kuota habis, format audio tidak didukung, dll) | Cek `GROQ_API_KEY` masih valid dan kuota Whisper belum habis; coba lagi beberapa saat kemudian |
| Tombol Whisper tidak muncul padahal video tanpa caption | `GROQ_API_KEY` kosong di Secrets | Isi `GROQ_API_KEY` -- tombol tetap muncul tapi klik akan menampilkan error kalau key kosong |
| Halaman "🔒 Login" tidak bisa dilewati | `APP_PASSWORD` di Secrets tidak cocok dengan yang diketik | Cek ejaan/spasi di Secrets, atau hapus `APP_PASSWORD` dari Secrets kalau ingin menonaktifkan gate |
