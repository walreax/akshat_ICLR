# -*- coding: utf-8 -*-
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

SRC = r"C:\Users\Akshat Jha\Downloads\write-up-longer.docx"
IMG_DIR = r"C:\Users\Akshat Jha\akshat_ICLR\writeups\example_images"
NUM_IMAGES = 13
COLS = 3

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def round_picture_corners(run, adj="20000"):
    """Change the inline picture's preset geometry from rect to roundRect."""
    prst_geoms = run._r.findall(f".//{{{A_NS}}}prstGeom")
    for pg in prst_geoms:
        pg.set("prst", "roundRect")
        # clear existing avLst children, then set the corner-radius adjustment
        for child in list(pg):
            pg.remove(child)
        av_lst = OxmlElement("a:avLst")
        gd = OxmlElement("a:gd")
        gd.set("name", "adj")
        gd.set("fmla", f"val {adj}")
        av_lst.append(gd)
        pg.append(av_lst)


doc = docx.Document(SRC)
paras = doc.paragraphs

heading_idx = None
for i, p in enumerate(paras):
    if p.text.strip() == "3.6 Example of the Prompt and the Experimental Setup":
        heading_idx = i
        break
if heading_idx is None:
    raise RuntimeError("Could not find the 3.6 section heading; aborting without changes.")

heading_para = paras[heading_idx]

# Collect every paragraph right after the heading whose own text is empty
# (this also matches the image-only paragraphs from the previous pass, since
# an inline picture contributes nothing to .text) up to the next real heading.
clear_slots = []
j = heading_idx + 1
while j < len(paras) and paras[j].text.strip() == "":
    clear_slots.append(paras[j])
    j += 1

if len(clear_slots) < 1:
    raise RuntimeError("No paragraph slots available after the heading; aborting without changes.")

# Wipe any existing content (text or pictures) from those paragraphs so we
# start clean, then delete all but one of them (we'll insert our own table
# right after that first slot and leave the rest as trailing blank spacers).
for p in clear_slots:
    for r in list(p.runs):
        r._r.getparent().remove(r._r)

anchor_para = clear_slots[0]

rows = (NUM_IMAGES + COLS - 1) // COLS
table = doc.add_table(rows=rows, cols=COLS)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = True

# Remove the table's borders/style noise by using a plain style, then move
# it out of its auto-appended position (end of document) to right after
# the anchor paragraph.
table.style = doc.styles["Normal Table"] if "Normal Table" in [s.name for s in doc.styles] else table.style
anchor_para._p.addnext(table._tbl)

img_idx = 0
for r in range(rows):
    for c in range(COLS):
        if img_idx >= NUM_IMAGES:
            break
        cell = table.cell(r, c)
        cell.text = ""
        img_p = cell.paragraphs[0]
        img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = img_p.add_run()
        img_path = f"{IMG_DIR}\\example_{img_idx:02d}.jpg"
        run.add_picture(img_path, width=Inches(1.85))
        round_picture_corners(run)

        cap_p = cell.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_p.paragraph_format.space_before = Pt(2)
        cap_run = cap_p.add_run(f"Example {img_idx:02d}")
        cap_run.italic = True
        cap_run.font.size = Pt(9)

        img_idx += 1

doc.save(SRC)
print("saved. rows:", rows, "cols:", COLS, "images placed:", img_idx)
