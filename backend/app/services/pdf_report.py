"""PDF match report generator — branded A4 reports for coaches to share."""

import io
import logging
import re
from datetime import datetime
from fpdf import FPDF

logger = logging.getLogger(__name__)


def sanitize(text: str) -> str:
    """Strip all non-latin1 characters for PDF font compatibility."""
    # Replace common Unicode first
    text = (text
        .replace("\u2014", "-").replace("\u2013", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2026", "...").replace("\u2022", "-")
        .replace("\u00a0", " ")
    )
    # Strip anything that can't encode to latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class MatchReportPDF(FPDF):
    """Custom PDF with Manager Mentor branding."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 6, "Manager Mentor - AI Match Analysis", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(156, 163, 175)
        self.cell(0, 8, f"Page {self.page_no()} | Generated {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}", align="C")

    def add_title_block(self, title: str, opponent: str, formation: str, duration_str: str, date_str: str):
        """Green header block with match info."""
        self.set_fill_color(6, 78, 59)
        self.rect(10, self.get_y(), 190, 30, "F")
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.set_xy(16, self.get_y() + 4)
        self.cell(0, 10, sanitize(title), new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 10)
        self.set_text_color(200, 230, 210)
        meta_parts = []
        if opponent:
            meta_parts.append(f"vs {opponent}")
        if formation:
            meta_parts.append(f"Formation: {formation}")
        if duration_str:
            meta_parts.append(duration_str)
        if date_str:
            meta_parts.append(date_str)
        self.set_x(16)
        self.cell(0, 8, "   |   ".join(meta_parts), new_x="LMARGIN", new_y="NEXT")
        self.set_y(self.get_y() + 10)
        self.set_text_color(26, 26, 26)

    def add_section(self, title: str, content: str, color: tuple[int, int, int]):
        """Add a colored section with title and content."""
        if not content:
            return
        content = sanitize(content)

        # Check if we need a new page (at least 40mm needed)
        if self.get_y() > 240:
            self.add_page()

        # Section title bar
        self.set_fill_color(*color)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

        # Content
        self.set_text_color(40, 40, 40)
        self.set_font("Helvetica", "", 10)

        # Process content line by line
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                self.ln(3)
                continue

            # Headers
            if stripped.startswith("### "):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(80, 80, 80)
                self.multi_cell(0, 6, stripped[4:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
            elif stripped.startswith("## "):
                self.set_font("Helvetica", "B", 11)
                self.set_text_color(6, 78, 59)
                self.multi_cell(0, 6, stripped[3:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
            elif stripped.startswith("# "):
                self.set_font("Helvetica", "B", 12)
                self.set_text_color(6, 78, 59)
                self.multi_cell(0, 7, stripped[2:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped[2:])
                self.cell(6, 6, chr(8226))
                self.multi_cell(0, 6, f" {clean}", new_x="LMARGIN", new_y="NEXT")
            elif re.match(r'^\d+\.', stripped):
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                self.multi_cell(0, 6, clean, new_x="LMARGIN", new_y="NEXT")
            else:
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                self.multi_cell(0, 6, clean, new_x="LMARGIN", new_y="NEXT")

        self.ln(6)


async def generate_pdf(match: dict, analyses: list[dict]) -> bytes:
    """Generate a PDF match report. Returns PDF bytes."""
    pdf = MatchReportPDF()
    pdf.add_page()

    # Match info
    title = match.get("title", "Match Report")
    opponent = match.get("opponent", "")
    formation = match.get("formation", "")
    duration = match.get("duration_seconds")
    duration_str = f"{int(duration // 60)} minutes" if duration else ""
    date_str = ""
    if match.get("created_at"):
        try:
            dt = datetime.fromisoformat(match["created_at"].replace("Z", "+00:00"))
            date_str = dt.strftime("%d %B %Y")
        except (ValueError, TypeError):
            pass

    pdf.add_title_block(title, opponent, formation, duration_str, date_str)

    # Extract sections from analyses
    coaching_advice = ""
    tactical_raw = ""
    highlights_raw = ""
    player_analysis = ""

    for a in analyses:
        if a.get("status") != "complete":
            continue
        if a.get("coaching_advice") and not coaching_advice:
            coaching_advice = a["coaching_advice"]
        if a.get("tactical_raw") and not tactical_raw:
            tactical_raw = a["tactical_raw"]
        if a.get("highlights_raw") and not highlights_raw:
            highlights_raw = a["highlights_raw"]
        if a.get("player_analysis_raw") and not player_analysis:
            player_analysis = a["player_analysis_raw"]

    # Add sections with brand colors
    pdf.add_section("Coaching Insights", coaching_advice, (6, 78, 59))      # emerald
    pdf.add_section("Tactical Analysis", tactical_raw, (30, 64, 120))       # blue
    pdf.add_section("Key Moments", highlights_raw, (120, 53, 15))           # amber
    pdf.add_section("Player Analysis", player_analysis, (22, 78, 99))       # cyan

    # Output
    buf = io.BytesIO()
    pdf.output(buf)
    pdf_bytes = buf.getvalue()
    logger.info("Generated PDF report for match %s (%d bytes)", match.get("id"), len(pdf_bytes))
    return pdf_bytes
