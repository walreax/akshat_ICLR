# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
normal.paragraph_format.line_spacing = 1.15

for s in doc.sections:
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


def add_author(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(12)


def add_affil(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)


def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)


def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)


def add_para(text, justify=True):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(text)
    return p


def add_bullet(label, body):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(6)
    r1 = p.add_run(label)
    r1.bold = True
    p.add_run(body)
    return p


def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9.5)


def set_cell_text(cell, text, bold=False, align_right=False, size=10):
    cell.text = ""
    p = cell.paragraphs[0]
    if align_right:
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)


def shade_cell(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


# ================= TITLE BLOCK =================
add_title("Mechanistic Interpretability of Diffusion Models:")
add_title("Concept-Level Metrics for Text-to-Image Generation Quality")
add_author("Akshat Jha")
add_affil("Independent research project, submitted for ICLR review")

# ================= ABSTRACT =================
add_h1("Abstract")
add_para(
    "Standard automated metrics for text-to-image generation, CLIP Score, BLIP-2 image-text "
    "matching, and VQAScore among them, evaluate a generated image only from the outside: they "
    "compare the final pixels to the prompt without reference to what the generative model "
    "actually computed to produce those pixels. This project evaluates three modern diffusion "
    "transformers, FLUX.1-schnell, Stable Diffusion 3 (SD3-medium), and PixArt-alpha, on a "
    "shared set of 600 prompts using the standard metrics above plus Inception Score, and in "
    "parallel develops a mechanistic alternative: sparse autoencoders (SAEs) trained on the "
    "residual activations of individual transformer blocks, used to decompose a generated "
    "image into a set of interpretable concepts and to score those concepts directly. Three "
    "concept-level metrics are proposed: Concept Centroid Deviation, the Cross-Concept Index, "
    "and the Attention Concentration Index. We report the completed four-metric benchmark "
    "across all three models, document concrete cases where the standard metrics disagree with "
    "one another in ways that reveal known blind spots (particularly CLIP's weakness on "
    "figurative language and on generic, low-salience scenes), and present preliminary "
    "correlation results between the mechanistic metrics and the standard benchmark. A "
    "self-hosted human evaluation tool has also been built to eventually calibrate the "
    "mechanistic metrics against real human quality judgments, which is identified as the "
    "critical next step for validating the approach."
)

# ================= 1 INTRODUCTION =================
add_h1("1. Introduction")
add_para(
    "Text-to-image diffusion models are typically compared using automated scorers that judge "
    "only the relationship between a finished image and its prompt. CLIP Score measures the "
    "cosine similarity between a shared image and text embedding. BLIP-2 image-text matching "
    "scores the probability that a caption and an image correspond. VQAScore reformulates the "
    "comparison as a visual question answering problem and scores the model's confidence that "
    "the image satisfies the prompt. Inception Score measures the diversity and confidence of a "
    "separate classifier's predictions over a batch of generated images, without reference to "
    "the prompt at all. Each of these is useful, and each has documented failure modes: CLIP in "
    "particular is known to behave like a bag of words, rewarding the presence of the right "
    "objects while being largely blind to composition, counting, spatial relationships, and "
    "negation."
)
add_para(
    "All four of these metrics share a deeper limitation. They treat the generative model as a "
    "black box and never look at what happens between the prompt and the pixels. Two images "
    "can receive the same CLIP score for very different reasons: one because the model cleanly "
    "represented every concept in the prompt, the other because the model got lucky, or because "
    "CLIP itself failed to penalize a real compositional error. This project's central "
    "hypothesis is that a metric grounded in the model's internal representation, specifically "
    "in how cleanly it represents individual concepts from the prompt and how those concepts "
    "relate to one another spatially, can capture image quality signal that these black box "
    "scorers miss, and that this signal is worth testing directly against human judgment rather "
    "than assumed."
)

# ================= 2 BACKGROUND =================
add_h1("2. Background: Concept-Attention and Sparse Autoencoders")
add_para(
    "The concept-attention framing used here follows Tinaz, Fabian, and Soltanolkotabi, "
    "\u201cEmergence and Evolution of Interpretable Concepts in Diffusion Models\u201d "
    "(NeurIPS 2025, arXiv:2504.15473). The central object of study in that work is not the raw "
    "cross-attention map a diffusion transformer produces, but a sparse autoencoder trained per "
    "architectural block and per denoising timestep on the residual update the block writes to "
    "the image stream. Formally, for block l at denoising step t, the training signal is"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
r = p.add_run("\u0394(l, t) = block_output(l, t) minus block_input(l, t)")
r.italic = True
r.font.size = Pt(12)
add_para(
    "the difference between what goes into the block and what comes out of it, rather than the "
    "raw hidden state itself. An interpretable concept, under this view, is a column of the "
    "sparse autoencoder's decoder matrix: a single direction in activation space that the "
    "sparsity constraint has isolated because it recurs across many prompts and images. A "
    "concept's spatial footprint in a specific generated image is simply how strongly its "
    "corresponding latent fires across the image's spatial tokens, reshaped back into a two "
    "dimensional map in the same way a cross-attention heatmap would be."
)
add_para(
    "Sparse autoencoders are used here rather than the dense hidden state directly because a "
    "block's raw activation is a superposition of many features at once and does not, by "
    "itself, correspond to any single interpretable idea. Training an over-complete "
    "autoencoder (more latent features than input dimensions) under an activation sparsity "
    "penalty is the standard mechanistic interpretability technique for pulling individual, "
    "separable concepts back out of that superposition."
)

# ================= 3 METHOD =================
add_h1("3. Experimental Setup")

add_h2("3.1 Models and Image Generation")
add_para(
    "Three models were evaluated: FLUX.1-schnell (a distilled, guidance-free model run for four "
    "inference steps), SD3-medium (run for the standard twenty-eight steps with its full text "
    "encoder stack, including the T5 encoder, enabled for a fair comparison against the other "
    "two models), and PixArt-alpha. All three models generated images for the same set of 600 "
    "prompts, drawn from a mixture of poems and short stories, using a fixed, identical seed "
    "across models so that any measured differences reflect the models themselves rather than "
    "sampling variance."
)

add_h2("3.2 Standard Benchmark Metrics")
add_para(
    "Every one of the 1,800 generated images (600 prompts times three models) was scored with "
    "four independent, established metrics: CLIP Score, BLIP-2 image-text matching, VQAScore "
    "(using a CLIP-FlanT5-XL backbone), and Inception Score. This benchmark is complete for all "
    "three models."
)

add_h2("3.3 Mechanistic Concept Extraction")
add_para(
    "Concept extraction was implemented separately for each model family, matched to its "
    "transformer architecture. SD3 has a single flat stack of joint attention blocks, and five "
    "SAEs were trained, one per selected block. FLUX has two structurally different block "
    "types, double-stream blocks in which the image and text streams attend to each other "
    "through separate projections, and single-stream blocks in which the two streams are later "
    "concatenated into one sequence. For FLUX, one SAE was trained per combination of four "
    "representative blocks (the earliest and latest block of each stream type) and four "
    "denoising timesteps spanning the full trajectory, for sixteen SAEs in total. Because "
    "FLUX.1-schnell runs for only four inference steps, these four timesteps correspond to "
    "every single step the model takes, so the sixteen SAE grid gives complete coverage of the "
    "model's denoising trajectory rather than a subsample of it."
)

add_h2("3.4 Proposed Concept-Level Metrics")
add_bullet(
    "Concept Centroid Deviation. ",
    "For a fixed concept, prompt, block, and timestep, this is the spread of the concept's "
    "spatial centroid across several independent random seeds. A low value means the model "
    "places the concept in a consistent location regardless of the particular noise sample, "
    "which is evidence that the concept is genuinely represented rather than incidentally "
    "produced."
)
add_bullet(
    "Cross-Concept Index. ",
    "The cosine similarity between the spatial maps of two distinct concepts within the same "
    "image. Concepts that should be distinct, two different characters in a scene, or a "
    "concrete object and a metaphorical one, should occupy different regions of the image. A "
    "high cross-concept index flags cases where the model has conflated concepts that the "
    "prompt keeps separate."
)
add_bullet(
    "Attention Concentration Index (ACI). ",
    "Defined as one minus the normalized Shannon entropy of a concept's spatial map. A "
    "concentrated, confidently localized concept has low entropy and therefore a high ACI; a "
    "diffuse, uncertain one has high entropy and a low ACI."
)

add_h2("3.5 Human Evaluation Infrastructure")
add_para(
    "Because the concept-level metrics above are only useful if they track something a human "
    "rater would actually notice, a self-hosted web application was built to collect human "
    "quality judgments. Raters see one image at a time, blind to which model produced it, "
    "alongside its source passage, and rate it on two dimensions on a one to five scale: prompt "
    "correctness and overall coherence. Sessions are tracked anonymously through a cookie so a "
    "rater can leave and resume, and the tool includes an informed consent screen describing the "
    "study, since this constitutes human subjects data collection for an ICLR submission. This "
    "tool is built and functional; the collection of a substantial pool of real human ratings is "
    "ongoing work."
)

# ================= 4 RESULTS =================
add_h1("4. Results")

add_h2("4.1 Standard Metric Benchmark")
add_para(
    "Table 1 reports the completed four-metric benchmark across all 600 images per model. SD3 "
    "leads on every one of the four metrics. The central open question this project is built "
    "around is whether that ranking is stable once mechanistic, and not only perceptual, "
    "measures of quality are taken into account."
)

table = doc.add_table(rows=5, cols=4)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.style = "Light Grid Accent 1"
hdr = ["Metric", "SD3", "PixArt", "FLUX"]
for i, h in enumerate(hdr):
    set_cell_text(table.rows[0].cells[i], h, bold=True, align_right=(i > 0))
    shade_cell(table.rows[0].cells[i], "D9D9D9")
rows_data = [
    ("Inception Score", "15.13", "13.35", "13.72"),
    ("VQAScore", "0.641", "0.617", "0.623"),
    ("CLIP Score", "0.225", "0.214", "0.220"),
    ("BLIP-2 Score", "0.734", "0.664", "0.677"),
]
for r_idx, (name, sd3, pix, flux) in enumerate(rows_data, start=1):
    set_cell_text(table.rows[r_idx].cells[0], name)
    set_cell_text(table.rows[r_idx].cells[1], sd3, align_right=True)
    set_cell_text(table.rows[r_idx].cells[2], pix, align_right=True)
    set_cell_text(table.rows[r_idx].cells[3], flux, align_right=True)
add_caption("Table 1. Standard metric benchmark, 600 images per model, all values are per-model means.")

add_h2("4.2 Where the Standard Metrics Disagree With Each Other")
add_para(
    "Ranking every one of the 1,800 generated images by how far its CLIP score diverges from "
    "the average of its BLIP-2 and VQAScore values surfaces concrete, reproducible cases that "
    "match well-documented failure modes of CLIP-based scoring."
)
add_para(
    "The clearest case is poem_0166, an SD3 generation of John Donne's \u201cSong\u201d "
    "(\u201cGo and catch a falling star, get with child a mandrake root\u2026\u201d), a highly "
    "figurative, metaphorical text. CLIP scored this image 0.120, close to the lowest CLIP score "
    "recorded for any SD3 image in the benchmark, while BLIP-2 scored it 0.946 and VQAScore "
    "scored it 0.638. This matches CLIP's known weakness on figurative and abstract language, "
    "since it was trained primarily on literal photographic captions and has limited capacity "
    "to bridge metaphor to a literal visual scene."
)
add_para(
    "A related case is story_0288, a FLUX generation of a generic passage (\u201cpeople gathered "
    "around to take part in\u2026 had a wonderful time\u201d) with no single salient, countable "
    "object. CLIP scored this image 0.148 against BLIP-2's 0.990 and VQAScore's 0.896. This "
    "matches CLIP's documented \u201cbag of words\u201d behavior: its global image-text "
    "similarity has little to latch onto when a passage describes a generic scene rather than "
    "naming specific, discriminable objects, whereas BLIP-2 and VQAScore, which reason over the "
    "full caption rather than a single embedding comparison, are not fooled in the same way."
)
add_para(
    "A third case runs in the opposite direction. For story_0412, a PixArt generation of two "
    "girls having a mock stick fight over a boy's attention, CLIP scored the image 0.271, a "
    "moderate to high score, while BLIP-2 scored it 0.0001, essentially zero. Here CLIP's "
    "coarse global similarity appears satisfied by the presence of plausible elements (two "
    "people, an outdoor setting, water nearby) while missing the specific narrative action the "
    "passage describes, a detail that BLIP-2's finer-grained matching head is built to catch. "
    "This is included specifically because it demonstrates that the disagreement between "
    "metrics is not simply CLIP being uniformly too harsh: the direction of the disagreement "
    "depends on the kind of error a given image contains."
)

add_h2("4.3 Mechanistic Metric Validation: Coverage")
add_para(
    "Before any mechanistic metric can be trusted, the extraction pipeline itself has to be "
    "shown to work correctly. The FLUX sixteen SAE grid was validated on a ten prompt sample "
    "and produced complete coverage: 320 of 320 expected rows (ten prompts, two seeds, sixteen "
    "block and timestep combinations), with zero missing cells. An earlier version of the "
    "pipeline had an off-by-one error in how the current denoising step was tracked relative to "
    "the transformer's forward pass, which silently dropped every row corresponding to the "
    "final denoising step across every block. This was caught by the coverage check on the ten "
    "prompt sample, root caused, and fixed prior to any of the results reported here."
)
add_para(
    "Across the validated ten prompt sample, concept concentration values fell in the range "
    "0.612 to 0.758, a moderate, non-degenerate range that indicates the trained SAEs are "
    "producing localized rather than uniform or collapsed spatial maps."
)

add_h2("4.4 Preliminary Correlation With Standard Metrics")
add_para(
    "For SD3, an initial correlation pass was run against the ten images already covered by "
    "the standard metric benchmark, joining the two tables on each image's identifier. Average "
    "SAE concentration, pooled across the five blocks and three timesteps evaluated, was "
    "compared to each standard metric using Spearman rank correlation. Table 2 reports the "
    "result."
)

table2 = doc.add_table(rows=4, cols=2)
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
table2.style = "Light Grid Accent 1"
set_cell_text(table2.rows[0].cells[0], "Standard metric", bold=True)
set_cell_text(table2.rows[0].cells[1], "Spearman correlation with mean SAE concentration", bold=True, align_right=True)
shade_cell(table2.rows[0].cells[0], "D9D9D9")
shade_cell(table2.rows[0].cells[1], "D9D9D9")
rows2 = [("CLIP Score", "0.333"), ("BLIP-2 Score", "0.103"), ("VQAScore", "0.576")]
for i, (name, val) in enumerate(rows2, start=1):
    set_cell_text(table2.rows[i].cells[0], name)
    set_cell_text(table2.rows[i].cells[1], val, align_right=True)
add_caption("Table 2. Spearman rank correlation, mean SAE concentration versus standard metrics, n = 10 images.")

add_para(
    "At this sample size the result should be read as directional rather than conclusive, but "
    "the pattern is notable: SAE concentration tracks VQAScore, the metric that reasons most "
    "explicitly about whether an image satisfies a prompt, considerably more closely than it "
    "tracks CLIP's coarse embedding similarity or BLIP-2's matching score. A broader run "
    "covering the full 600 image benchmark, using all five trained SD3 SAE blocks, is in "
    "progress at the time of writing, as is the corresponding sixteen SAE FLUX pass over its "
    "full 4,198 prompt training corpus."
)

# ================= 5 DISCUSSION =================
add_h1("5. Discussion and Limitations")
add_para(
    "The results above establish two things and leave one central question open. First, the "
    "standard metric benchmark is complete and internally consistent, and it surfaces concrete, "
    "explainable cases of disagreement between CLIP, BLIP-2, and VQAScore that match known "
    "properties of each metric rather than looking like noise. Second, the mechanistic "
    "extraction pipeline itself is validated: it produces complete, non-degenerate coverage "
    "across the full block and timestep grid for both FLUX and SD3, and an early correlation "
    "signal against the standard benchmark is present and directionally sensible."
)
add_para(
    "What remains open is the question the project was built to answer: whether the "
    "concept-level metrics explain variance in actual human quality judgment that the standard "
    "metrics leave unexplained. A correlation between a mechanistic metric and another automated "
    "metric, however sensible the direction, is not itself evidence that the mechanistic metric "
    "is measuring something a human would care about. That requires a genuine incremental "
    "validity test: fitting the standard metrics to real human ratings, then testing whether the "
    "concept-level metrics predict the residual variance the standard metrics fail to capture. "
    "The human evaluation tool described in Section 3.5 exists specifically to make this test "
    "possible, and running it at a scale sufficient to support a regression analysis is the "
    "immediate next step for this project."
)
add_para(
    "A second limitation is sample size in the correlation results reported in Section 4.4, "
    "which currently reflect ten prompts. Both the SD3 correlation run and the FLUX training "
    "run were scaled up to their full targets (600 prompts and 4,198 prompts respectively) at "
    "the time of writing, and the numbers in this section should be treated as an early "
    "checkpoint rather than a final result."
)

# ================= 6 CONCLUSION =================
add_h1("6. Conclusion and Future Work")
add_para(
    "This project has built and validated a complete pipeline for evaluating text-to-image "
    "diffusion models along two independent axes: a standard, four-metric perceptual benchmark, "
    "complete across all 1,800 generated images, and a mechanistic, sparse autoencoder based "
    "pipeline for decomposing a generated image into interpretable concepts and scoring their "
    "spatial reliability, consistency across seeds, and separation from one another. The "
    "standard benchmark alone already yields useful findings, including concrete, reproducible "
    "cases where CLIP's known blind spots on figurative language and generic scenes cause it to "
    "diverge sharply from BLIP-2 and VQAScore. The mechanistic pipeline is now producing "
    "complete, validated coverage for both FLUX and SD3, and an early correlation pass suggests "
    "its concept-level concentration metric tracks VQAScore more closely than it tracks CLIP."
)
add_para(
    "The immediate priorities going forward are, first, to complete the full scale runs of both "
    "the SD3 correlation analysis and the FLUX sixteen SAE training pass that are currently in "
    "progress, and second, to collect a substantial pool of human quality ratings through the "
    "deployed evaluation tool so that the concept-level metrics can be tested against human "
    "judgment directly through an incremental validity regression, which is the result that "
    "would actually establish whether this mechanistic approach adds anything the standard "
    "metrics do not already provide."
)

# ================= REFERENCES =================
add_h1("References")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
p.add_run(
    "Tinaz, B., Fabian, Z., and Soltanolkotabi, M. Emergence and Evolution of Interpretable "
    "Concepts in Diffusion Models. NeurIPS 2025. arXiv:2504.15473."
)

doc.save(r"C:\Users\Akshat Jha\akshat_ICLR\writeups\Mechanistic_Interpretability_of_Diffusion_Models_Full.docx")
print("saved")
