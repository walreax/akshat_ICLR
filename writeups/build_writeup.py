# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn

doc = Document()

# ---- base style ----
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
normal.paragraph_format.line_spacing = 1.15

sections = doc.sections
for s in sections:
    s.left_margin = Inches(1)
    s.right_margin = Inches(1)
    s.top_margin = Inches(0.9)
    s.bottom_margin = Inches(0.9)


def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(18)
    return p


def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(11)
    r.font.color.rgb = None
    return p


def add_heading(text, size=13):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    return p


def add_para(text):
    p = doc.add_paragraph()
    p.add_run(text)
    return p


def add_bullet(label, body):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(6)
    r1 = p.add_run(label)
    r1.bold = True
    p.add_run(body)
    return p


# ================= TITLE =================
add_title("Mechanistic Interpretability of Diffusion Models")
add_subtitle("A concept-attention framework for interpreting text-to-image transformers")

# ================= 1 =================
add_heading("1. The Concept-Attention Idea")
add_para(
    "The starting point for this line of work is the idea that a diffusion transformer's "
    "internal representation of a prompt can be decomposed into a set of discrete, "
    "human-interpretable \u201cconcepts,\u201d each with its own spatial footprint over the "
    "generated image. This framing follows Tinaz, Fabian & Soltanolkotabi, \u201cEmergence and "
    "Evolution of Interpretable Concepts in Diffusion Models\u201d (NeurIPS 2025), which "
    "introduces concept-attention as a way to read out what a model is actually representing "
    "at a given block and denoising timestep \u2014 as opposed to treating the model purely as a "
    "black box scored only on the pixels it eventually produces."
)
add_para(
    "The practical appeal of this framing for our purposes is that it gives us a route to an "
    "image-quality signal that is grounded in what the model computed, not just in how the "
    "output looks to an external scorer (CLIP, BLIP-2, VQAScore). If a model represents the "
    "concepts in a prompt cleanly and consistently, that is itself evidence of quality \u2014 "
    "independent of, and potentially complementary to, whatever a downstream vision-language "
    "scorer says about the final image."
)

# ================= 2 =================
add_heading("2. Extracting Concepts: Sparse Autoencoders on Block Residuals")
add_para(
    "Concepts are not read directly off the model's raw hidden states \u2014 those are dense, "
    "high-dimensional, and not aligned with any single human-interpretable axis. Instead, we "
    "train a Sparse Autoencoder (SAE) on the residual update that each transformer block "
    "writes to the image stream:"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
r = p.add_run("\u0394\u2113,\u209c  =  block_output \u2212 block_input")
r.italic = True
r.font.size = Pt(12)

add_para(
    "for a given block \u2113 at denoising step t. This residual, rather than the block's raw "
    "output, is what the SAE is trained on. Sparse coding on the residual is the standard "
    "mechanistic-interpretability move: because the SAE is over-complete (many more latent "
    "features than input dimensions) and trained with a sparsity constraint, individual "
    "latent directions tend to line up with distinct, separable concepts instead of "
    "superpositions of many concepts at once \u2014 which is what makes them interpretable in "
    "the first place. One SAE is trained per (block, timestep) pair, since a block's role in "
    "the computation \u2014 and therefore what it represents \u2014 is not assumed to be the same "
    "at every point in the denoising trajectory."
)
add_para(
    "Once trained, a SAE latent's activation across image tokens can be reshaped back into a "
    "spatial map \u2014 exactly like a cross-attention heatmap \u2014 giving each concept a "
    "location, an extent, and a strength within the generated image."
)

# ================= 3 =================
add_heading("3. An Image as a Composition of Concepts")
add_para(
    "With per-concept spatial maps in hand, a generated image can be treated not as a single "
    "opaque object but as a composition of concepts, and image quality can be studied through "
    "the relationships between those concepts rather than through the pixels alone: Are the "
    "concepts the prompt calls for actually present in the representation? Are they spatially "
    "well-localized, or diffuse and uncertain? Do concepts that should be distinct (e.g. two "
    "different characters, or a concrete object versus a metaphorical one) actually occupy "
    "distinct regions of the image, or do they collapse into each other? This relational view "
    "is what the three metrics below are designed to make quantitative."
)

# ================= 4 =================
add_heading("4. Three Proposed Metrics")
add_bullet(
    "Concept Centroid Deviation \u2014 ",
    "for a fixed concept, the spread of its spatial centroid across independent random seeds "
    "(same prompt, same block, same timestep, different noise). Low deviation means the model "
    "consistently places the concept in the same region regardless of seed \u2014 i.e. it is "
    "representing the concept, not guessing at it. High deviation suggests the concept is not "
    "reliably encoded."
)
add_bullet(
    "Cross-Concept Index (overlap) \u2014 ",
    "the cosine similarity between the spatial maps of two different concepts within the same "
    "image. A well-formed image should keep distinct concepts spatially separated; a high "
    "cross-concept index flags cases where the model is conflating two things it should be "
    "keeping apart."
)
add_bullet(
    "Attention Concentration Index (ACI) \u2014 ",
    "1 minus the normalized Shannon entropy of a concept's spatial map. A concentrated, "
    "confident representation has low entropy (high ACI); a diffuse, uncertain one has high "
    "entropy (low ACI). This is essentially asking how \u201cin focus\u201d each concept's "
    "footprint is."
)
add_para(
    "Together, these three give a per-concept, per-image profile: where a concept lives "
    "(centroid), how reliably it lives there (deviation), how sharply it is represented "
    "(ACI), and whether it is properly disentangled from other concepts in the same image "
    "(cross-concept index)."
)

# ================= 5 =================
add_heading("5. Where This Stands, and What's Still Open")
add_para(
    "The pipeline above is implemented and validated end-to-end on both FLUX (double- and "
    "single-stream blocks, a 4-block \u00d7 4-timestep SAE grid) and SD3 (a 5-block SAE set), and "
    "produces exactly the per-concept metrics described above at scale. The open question the "
    "project is currently working through is validation: do these mechanistic metrics track "
    "anything a standard image-quality scorer (CLIP, BLIP-2, VQAScore) does not already "
    "capture? An early correlation pass against our own CLIP/BLIP-2/VQAScore benchmark shows "
    "a directional signal \u2014 SAE concentration tracks VQAScore noticeably more closely than "
    "it tracks CLIP \u2014 but this needs to be run at full scale and, more importantly, checked "
    "against real human quality judgments rather than against other automated metrics alone. "
    "The principled version of that check is an incremental-validity regression: fit the "
    "standard metrics to human ratings, then test whether the mechanistic metrics explain any "
    "of the variance in human judgment that the standard metrics leave on the table. That "
    "human-rating collection is the next major piece of infrastructure this project needs "
    "before the concept-attention metrics can be validated as more than a self-consistent "
    "internal signal."
)

doc.save(r"C:\Users\Akshat Jha\akshat_ICLR\writeups\Mechanistic_Interpretability_of_Diffusion_Models.docx")
print("saved")
