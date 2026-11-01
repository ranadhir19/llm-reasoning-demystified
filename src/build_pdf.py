"""Build the article PDF from article/article.md with ReportLab Platypus.

Markdown conventions understood by this builder (the article is written
to them):
  "# "  title (first line)         "## "  section heading
  "### " subsection heading        "**Name**" line after title: author
  "[FIGURE:file.png|Caption]"      figure with numbered caption
  "[EQ] ... "                      centred display equation line
  "| a | b |" rows                 a table (first row = header)
  other non-empty lines            body paragraphs (consecutive lines
                                   are joined into one paragraph)
  "**bold**" and "*italic*"        inline markup (minimal)

Output: article/Demystifying_LLM_Reasoning.pdf with page numbers and
the footer "Demystifying LLM Reasoning - R. Ghosh".
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "article", "article.md")
OUT = os.path.join(ROOT, "article", "Demystifying_LLM_Reasoning.pdf")
FIGDIR = os.path.join(ROOT, "figures")

FOOTER = "Demystifying LLM Reasoning \u2014 R. Ghosh"

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold",
                            fontSize=19, leading=24, alignment=TA_CENTER,
                            spaceAfter=6),
    "author": ParagraphStyle("author", fontName="Helvetica",
                             fontSize=12, leading=16, alignment=TA_CENTER,
                             spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14,
                         leading=18, spaceBefore=16, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                         leading=15, spaceBefore=12, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=11,
                           leading=15, alignment=TA_JUSTIFY,
                           spaceAfter=7),
    "eq": ParagraphStyle("eq", fontName="Times-Italic", fontSize=11.5,
                         leading=16, alignment=TA_CENTER, spaceBefore=4,
                         spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
                              fontSize=9, leading=12, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=12),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9.5,
                           leading=12.5),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold",
                            fontSize=9.5, leading=12.5),
    "bullet": ParagraphStyle("bullet", fontName="Times-Roman", fontSize=11,
                             leading=15, alignment=TA_JUSTIFY,
                             leftIndent=16, firstLineIndent=0,
                             bulletIndent=6, spaceAfter=3),
    "quote": ParagraphStyle("quote", fontName="Times-Italic", fontSize=11,
                            leading=15, alignment=TA_JUSTIFY,
                            leftIndent=20, rightIndent=20, spaceAfter=7),
}


def inline(text):
    text = text.replace("&", "&amp;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(2 * cm, 1.1 * cm, FOOTER)
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


def parse(md):
    lines = md.split("\n")
    flow = []
    fig_no = 0
    i = 0
    para_buf = []

    def flush_para():
        if para_buf:
            text = " ".join(x.strip() for x in para_buf)
            flow.append(Paragraph(inline(text), styles["body"]))
            para_buf.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|")
                         .split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    i += 1
                    continue
                rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                data = []
                for ri, r in enumerate(rows):
                    row = []
                    for c in r + [""] * (ncol - len(r)):
                        row.append(Paragraph(
                            inline(c),
                            styles["cellb" if ri == 0 else "cell"]))
                    data.append(row)
                avail = A4[0] - 4 * cm
                t = Table(data, colWidths=[avail / ncol] * ncol,
                          repeatRows=1, hAlign="CENTER")
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                flow.append(t)
                flow.append(Spacer(1, 8))
            continue
        if s.startswith("[FIGURE:"):
            flush_para()
            fig_no += 1
            inner = s[len("[FIGURE:"):-1]
            fname, _, caption = inner.partition("|")
            path = os.path.join(FIGDIR, fname.strip())
            from PIL import Image as PILImage
            iw, ih = PILImage.open(path).size
            width = A4[0] - 4 * cm
            height = width * ih / iw
            maxh = 9.5 * cm
            if height > maxh:
                height = maxh
                width = height * iw / ih
            flow.append(Image(path, width=width, height=height,
                              hAlign="CENTER"))
            flow.append(Paragraph(f"Figure {fig_no}. {inline(caption)}",
                                  styles["caption"]))
            i += 1
            continue
        if s.startswith("[EQ] "):
            flush_para()
            flow.append(Paragraph(inline(s[5:]), styles["eq"]))
            i += 1
            continue
        if s.startswith("### "):
            flush_para()
            flow.append(Paragraph(inline(s[4:]), styles["h2"]))
        elif s.startswith("## "):
            flush_para()
            flow.append(Paragraph(inline(s[3:]), styles["h1"]))
        elif s.startswith("# "):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), styles["title"]))
        elif s.startswith("**") and s.endswith("**") and len(flow) <= 2:
            flush_para()
            flow.append(Paragraph(inline(s), styles["author"]))
        elif s == "":
            flush_para()
        elif s == "[PAGEBREAK]":
            flush_para()
            flow.append(PageBreak())
        elif s.startswith("- "):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), styles["bullet"],
                                  bulletText="\u2022"))
        elif s.startswith("> "):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), styles["quote"]))
        else:
            para_buf.append(line)
        i += 1
    flush_para()
    return flow


def main():
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2 * cm, bottomMargin=1.8 * cm,
                          title="Demystifying LLM Reasoning: From Language "
                                "to Mathematical Constructs",
                          author="Ranadhir Ghosh")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                       onPage=footer)])
    doc.build(parse(md))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    sys.exit(main())
