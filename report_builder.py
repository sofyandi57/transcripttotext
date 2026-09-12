"""
report_builder.py

Bikin laporan PDF (ringkasan + transcript + riwayat tanya-jawab) dari satu
video yang sudah diproses. Dipisah dari app.py dan ai_service.py karena ini
murni presentasi/output, tidak ada logika AI di sini.

CATATAN SOAL FONT:
fpdf2 secara default cuma punya font core PDF (Helvetica dkk) yang terbatas
di karakter Latin-1 -- tanda kutip pintar/em dash dari hasil LLM (Groq/Gemini)
bakal tampil sebagai "?" kalau dipaksa pakai font ini. Supaya PDF-nya rapi,
kode ini coba pakai font DejaVu Sans (Unicode penuh) yang HARUS sudah
ter-install di sistem lewat `packages.txt` (apt package `fonts-dejavu-core`)
-- ini yang bikin Streamlit Cloud bisa render Unicode dengan benar.

Kalau font itu tidak ketemu (misal development lokal di Mac tanpa apt),
otomatis fallback ke Helvetica + karakter non-Latin-1 diganti "?" -- PDF
tetap jadi, cuma tipografi LLM (smart quotes dll) jadi kurang rapi.
"""

from __future__ import annotations

import os
from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Path standar DejaVu Sans di Debian/Ubuntu (dipasang lewat packages.txt).
_DEJAVU_REGULAR_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_DEJAVU_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _find_font(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


class _ReportPDF(FPDF):
    def __init__(self, unicode_font: bool):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.unicode_font = unicode_font
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(18, 18, 18)

    def sanitize(self, text: str) -> str:
        if self.unicode_font:
            return text
        return text.encode("latin-1", "replace").decode("latin-1")

    def _line(self, text: str, h: float):
        # new_x/new_y default fpdf2 (XPos.RIGHT) TIDAK kembali ke margin kiri
        # setelah multi_cell -- kalau tidak dipaksa LMARGIN/NEXT di sini,
        # panggilan multi_cell berikutnya kehabisan lebar horizontal.
        self.multi_cell(0, h, self.sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h1(self, text: str):
        self.set_font(self.font_family_name, "B", 16)
        self._line(text, 9)
        self.ln(2)

    def h2(self, text: str):
        self.set_font(self.font_family_name, "B", 12)
        self._line(text, 7)
        self.ln(1)

    def body(self, text: str):
        self.set_font(self.font_family_name, "", 10)
        self._line(text, 5.5)
        self.ln(1)

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
) -> bytes:
    """
    Susun laporan PDF satu video.

    Parameters
    ----------
    title : judul video (atau video_id kalau tidak diisi)
    video_id, language, word_count : metadata video
    summary : hasil ringkasan (bisa None kalau belum pernah dibuat)
    transcript : isi transcript penuh
    faq_items : list of {"question": str, "answer": str} (FAQ otomatis)
    qa_history : list of {"question": str, "answer": str} (tanya-jawab manual)

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

    pdf.add_page()

    pdf.h1(title or video_id)
    pdf.caption(
        f"Video ID: {video_id}  |  Bahasa: {language}  |  {word_count} kata  |  "
        f"Laporan dibuat: {date.today().isoformat()}"
    )
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
    pdf.body(transcript.strip() or "(Transcript kosong.)")
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
