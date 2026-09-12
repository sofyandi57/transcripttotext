"""
report_builder.py

Bikin laporan PDF (ringkasan + transcript + riwayat tanya-jawab) dari satu
video yang sudah diproses. Dipisah dari app.py dan ai_service.py karena ini
murni presentasi/output, tidak ada logika AI di sini.

CATATAN SOAL FONT:
fpdf2 secara default cuma punya font core PDF (Helvetica dkk) yang terbatas
di karakter Latin-1 -- tanda kutip pintar/em dash dari hasil LLM (Groq/Gemini)
bakal tampil sebagai "?" kalau dipaksa pakai font ini. Supaya PDF-nya rapi,
kode ini coba pakai font DejaVu Sans (Unicode penuh utk Latin/Cyrillic/
Yunani -- termasuk karakter Turki seperti ğ/ş/ı, itu tetap Latin Extended)
yang HARUS sudah ter-install di sistem lewat `packages.txt` (apt package
`fonts-dejavu-core`) -- ini yang bikin Streamlit Cloud bisa render Unicode
dengan benar.

Kalau font itu tidak ketemu (misal development lokal di Mac tanpa apt),
otomatis fallback ke Helvetica + karakter non-Latin-1 diganti "?" -- PDF
tetap jadi, cuma tipografi LLM (smart quotes dll) jadi kurang rapi.

CATATAN SOAL SCRIPT NON-LATIN (Thai, CJK, Tamil, Devanagari/Hindi):
DejaVu Sans TIDAK punya glyph untuk script-script itu. Ringkasan/FAQ/Q&A
di PDF ini SELALU dalam Bahasa Indonesia (lihat prompt di ai_service.py),
jadi bagian itu aman -- tapi bagian "Transcript Lengkap" berisi kutipan
ASLI dari video, jadi kalau videonya berbahasa Thai/Jepang/Korea/Cina/
Tamil/Hindi, teksnya butuh font lain. `_detect_script()` mendeteksi ini
dan `build_pdf_report()` pakai font Noto yang sesuai KHUSUS untuk bagian
transcript itu saja (bagian lain tetap DejaVu). Font Noto per-script
dipasang lewat `packages.txt` (fonts-noto-core untuk Thai/Tamil/Devanagari,
fonts-noto-cjk untuk Cina/Jepang/Korea).

⚠️ Path & format file Noto di bawah ini BELUM diverifikasi langsung di
Streamlit Cloud (tidak bisa apt-get di macOS buat testing lokal) -- kalau
font tidak ketemu di path yang diduga, otomatis fallback ke DejaVu (yang
juga tidak punya glyph-nya, jadi tampil "?") tanpa bikin PDF gagal total.
Kalau ternyata pathnya beda, cek `fc-list` di Streamlit Cloud (lewat log
atau shell kalau ada aksesnya) dan update daftar path di bawah.
"""

from __future__ import annotations

import os
from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from transcript_service import VideoMetadata

# Path standar DejaVu Sans di Debian/Ubuntu (dipasang lewat packages.txt).
_DEJAVU_REGULAR_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_DEJAVU_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]

# Font Noto per-script untuk bagian transcript non-Latin (lihat catatan di
# atas soal path yang belum terverifikasi). "cjk" pakai .ttc (font
# collection) -- fpdf2 butuh index sub-font eksplisit untuk itu.
_SCRIPT_FONT_PATHS: dict[str, list[str]] = {
    "thai": [
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf",
    ],
    "tamil": [
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansTamil-Regular.ttf",
    ],
    "devanagari": [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ],
    "cjk": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ],
    "arabic": [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansArabic-Regular.ttf",
    ],
}

# Script yang butuh text SHAPING (huruf saling sambung + arah RTL) --
# beda dari Thai/CJK/Tamil/Devanagari yang cukup ganti font saja. Arab dan
# Persia (Farsi) sama-sama pakai alfabet Arab, jadi satu font/logic yang
# sama untuk keduanya. Butuh `uharfbuzz` ter-install (lihat requirements.txt)
# -- tanpa itu, fpdf2 tetap jalan tapi huruf tampil terpisah-pisah (tidak
# tersambung) dan urutan kata bisa salah.
_RTL_SCRIPTS = {"arabic"}


def _find_font(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _detect_script(text: str, threshold: int = 20) -> str | None:
    """
    Deteksi script non-Latin dominan dalam teks lewat Unicode code point
    range, buat pilih font PDF yang tepat untuk bagian transcript. Return
    None kalau teks dominan Latin/Cyrillic/Yunani (termasuk Turki -- masih
    Latin Extended, DejaVu Sans sudah cover) -- font default DejaVu sudah
    cukup, tidak perlu font tambahan.

    `threshold`: minimal jumlah karakter script itu supaya tidak salah
    deteksi gara-gara 1-2 karakter nyasar (emoji, dll).
    """
    # Arab & Persia (Farsi) sama-sama pakai alfabet Arab (Persia nambah
    # beberapa huruf spt پ چ ژ گ yang tetap masuk blok Unicode Arabic ini),
    # jadi satu kategori "arabic" untuk keduanya.
    counts = {"cjk": 0, "thai": 0, "tamil": 0, "devanagari": 0, "arabic": 0}
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7A3:
            counts["cjk"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["thai"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tamil"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFEFF:
            counts["arabic"] += 1
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] >= threshold else None


class _ReportPDF(FPDF):
    def __init__(self, unicode_font: bool):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.unicode_font = unicode_font
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(18, 18, 18)

    def sanitize(self, text: str, unicode_ok: bool = False) -> str:
        if self.unicode_font or unicode_ok:
            return text
        return text.encode("latin-1", "replace").decode("latin-1")

    def _line(self, text: str, h: float, unicode_ok: bool = False):
        # new_x/new_y default fpdf2 (XPos.RIGHT) TIDAK kembali ke margin kiri
        # setelah multi_cell -- kalau tidak dipaksa LMARGIN/NEXT di sini,
        # panggilan multi_cell berikutnya kehabisan lebar horizontal.
        self.multi_cell(0, h, self.sanitize(text, unicode_ok=unicode_ok), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h1(self, text: str):
        self.set_font(self.font_family_name, "B", 16)
        self._line(text, 9)
        self.ln(2)

    def h2(self, text: str):
        self.set_font(self.font_family_name, "B", 12)
        self._line(text, 7)
        self.ln(1)

    def body(self, text: str, family: str | None = None, rtl: bool = False):
        # `family` override -- dipakai buat bagian transcript kalau script-nya
        # butuh font lain dari default (lihat script_font_name di
        # build_pdf_report). Kalau None, pakai font default seperti biasa.
        # unicode_ok=True karena family override selalu font Unicode yang
        # berhasil di-register (lihat pemanggilnya) -- jangan di-Latin-1-kan.
        #
        # `rtl` -- untuk Arab/Persia: huruf harus disambung (shaping) dan
        # dibaca kanan-ke-kiri. Tanpa ini, huruf tampil terpisah-pisah dan
        # urutannya salah meski font-nya sudah benar. Dimatikan lagi setelah
        # section ini supaya tidak memengaruhi teks Indonesia berikutnya.
        if rtl:
            self.set_text_shaping(use_shaping_engine=True, direction="rtl")
        self.set_font(family or self.font_family_name, "", 10)
        self._line(text, 5.5, unicode_ok=bool(family))
        self.ln(1)
        if rtl:
            self.set_text_shaping(use_shaping_engine=False)
        self.set_font(self.font_family_name, "", 10)  # kembalikan ke default

    def caption(self, text: str):
        self.set_font(self.font_family_name, "I", 8.5)
        self.set_text_color(110, 110, 110)
        self._line(text, 5)
        self.set_text_color(0, 0, 0)
        self.ln(1)


def build_pdf_report(
    title: str,
    video_id: str,
    language: str,
    word_count: int,
    summary: str | None,
    transcript: str,
    faq_items: list[dict],
    qa_history: list[dict],
    metadata: VideoMetadata | None = None,
    translation: str | None = None,
) -> bytes:
    """
    Susun laporan PDF satu video.

    Parameters
    ----------
    title : judul video (atau video_id kalau tidak diisi)
    video_id, language, word_count : metadata video dari transcript
    summary : hasil ringkasan (bisa None kalau belum pernah dibuat)
    transcript : isi transcript penuh
    translation : terjemahan LENGKAP ke Indonesia/Inggris (None kalau
        video sudah berbahasa id/en, atau belum pernah diterjemahkan)
    faq_items : list of {"question": str, "answer": str} (FAQ otomatis)
    qa_history : list of {"question": str, "answer": str} (tanya-jawab manual)
    metadata : VideoMetadata dari YouTube Data API (opsional -- None kalau
        YOUTUBE_API_KEY tidak diisi/lookup gagal) -- channel, tanggal upload
        asli, durasi, views, deskripsi.

    Returns
    -------
    bytes isi file PDF, siap dipakai untuk st.download_button.
    """
    regular_path = _find_font(_DEJAVU_REGULAR_PATHS)
    bold_path = _find_font(_DEJAVU_BOLD_PATHS)
    unicode_font = bool(regular_path and bold_path)

    pdf = _ReportPDF(unicode_font=unicode_font)

    if unicode_font:
        pdf.add_font("DejaVu", "", regular_path)
        pdf.add_font("DejaVu", "B", bold_path)
        pdf.add_font("DejaVu", "I", regular_path)
        pdf.font_family_name = "DejaVu"
    else:
        pdf.font_family_name = "Helvetica"

    # Transcript bisa berbahasa apa saja (Thai/Jepang/Korea/Cina/Tamil/Hindi
    # dll) -- DejaVu tidak punya glyph untuk script itu. Deteksi & pasang
    # font Noto yang sesuai KHUSUS untuk bagian "Transcript Lengkap"; bagian
    # lain (Ringkasan/FAQ/Q&A) selalu Bahasa Indonesia jadi tetap pakai
    # DejaVu seperti biasa.
    transcript_font_family: str | None = None
    script = _detect_script(transcript)
    if script:
        script_path = _find_font(_SCRIPT_FONT_PATHS[script])
        if script_path:
            try:
                if script == "cjk":
                    pdf.add_font("ScriptFont", "", script_path, collection_font_number=0)
                else:
                    pdf.add_font("ScriptFont", "", script_path)
                transcript_font_family = "ScriptFont"
            except Exception:
                transcript_font_family = None  # font ada tapi gagal di-load -- fallback diam-diam ke default

    pdf.add_page()

    pdf.h1(title or video_id)

    caption_parts = [f"Video ID: {video_id}", f"Bahasa: {language}", f"{word_count} kata"]
    if metadata:
        caption_parts += [
            f"Channel: {metadata.channel_title}",
            f"Durasi: {metadata.duration_display}",
            f"{metadata.view_count:,} views".replace(",", "."),
            f"Diupload: {metadata.published_at}",
        ]
    caption_parts.append(f"Laporan dibuat: {date.today().isoformat()}")
    pdf.caption("  |  ".join(caption_parts))

    if metadata and metadata.description:
        pdf.ln(1)
        pdf.caption(f"Deskripsi: {metadata.description}")

    pdf.ln(3)

    # Terjemahan selalu Latin (Indonesia/Inggris per prompt di ai_service.py)
    # apa pun script sumbernya -- tidak butuh font/shaping khusus di sini.
    if translation:
        pdf.h2("Terjemahan")
        pdf.body(translation.strip())
        pdf.ln(3)

    pdf.h2("Ringkasan")
    pdf.body(summary.strip() if summary else "(Belum ada ringkasan yang dibuat untuk video ini.)")
    pdf.ln(3)

    pdf.h2(f"FAQ ({len(faq_items)})")
    if not faq_items:
        pdf.body("(Belum ada FAQ yang dibuat untuk video ini.)")
    else:
        for i, item in enumerate(faq_items, 1):
            pdf.set_font(pdf.font_family_name, "B", 10)
            pdf._line(f"{i}. {item['question']}", 5.5)
            pdf.body(item["answer"])
            pdf.ln(1)
    pdf.ln(2)

    pdf.h2("Transcript Lengkap")
    pdf.body(
        transcript.strip() or "(Transcript kosong.)",
        family=transcript_font_family,
        rtl=(script in _RTL_SCRIPTS),
    )
    pdf.ln(3)

    pdf.h2(f"Riwayat Tanya-Jawab ({len(qa_history)})")
    if not qa_history:
        pdf.body("(Belum ada pertanyaan yang diajukan untuk video ini.)")
    else:
        for i, qa in enumerate(qa_history, 1):
            pdf.set_font(pdf.font_family_name, "B", 10)
            pdf._line(f"{i}. {qa['question']}", 5.5)
            pdf.body(qa["answer"])
            pdf.ln(1)

    return bytes(pdf.output())
