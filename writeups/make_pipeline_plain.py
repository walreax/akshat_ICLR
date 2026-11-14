"""Plain, ICLR-style MIRAGE pipeline figure (black/white, Times-like, vector PDF).

Sized for the ICLR text width (5.5 in). Writes writeups/mirage_pipeline_plain.{pdf,png}.
Run:  python writeups/make_pipeline_plain.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FW, FH = 5.5, 1.90
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0.17, 0.17 + FH)
ax.axis("off")

LW = 0.6
TITLE, BODY, MATH = 7.0, 5.7, 6.2

# column left edges / widths (inches)
widths = [0.78, 0.88, 0.98, 1.14, 0.86]
gaps = [0.16, 0.16, 0.24, 0.28]          # fork gap and merge gap are wider so the joins sit clear of the dashed group
lefts = []
x = (FW - (sum(widths) + sum(gaps))) / 2
for i, w in enumerate(widths):
    lefts.append(x)
    x += w + (gaps[i] if i < len(gaps) else 0)

yc = 1.12                 # centre line of the single boxes
H1, H2 = 0.98, 0.86       # single / stacked box heights
yA, yB = yc + 0.48, yc - 0.48   # centres of the two stacked boxes


def box(col, yc_, h, title, body, formula, formula_y=None):
    x0, w = lefts[col], widths[col]
    ax.add_patch(FancyBboxPatch((x0, yc_ - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec="black", lw=LW, zorder=3))
    cx = x0 + w / 2
    ax.text(cx, yc_ + h / 2 - 0.07, title, ha="center", va="top", fontsize=TITLE,
            fontweight="bold", linespacing=1.05, zorder=4)
    n_title = title.count("\n") + 1
    ytxt = yc_ + h / 2 - 0.07 - n_title * 0.115 - 0.05
    ax.text(cx, ytxt, body, ha="center", va="top", fontsize=BODY, linespacing=1.18, zorder=4)
    if formula:
        ax.text(cx, formula_y if formula_y is not None else yc_ - h / 2 + 0.17, formula,
                ha="center", va="center", fontsize=MATH, zorder=4)


box(0, yc, H1, "Creative\nprompt", "poem or story $P$\nwith salient\nconcepts $\\mathcal{C}(P)$", None)
box(1, yc, H1, "Diffusion\ntransformer", "$m$ stochastic seeds,\n$T$ denoising steps", r"$I_{S_r}=G(P,S_r)$")
box(2, yc, H1, "Sparse features", "block-wise SAE on\nimage tokens; dominant\nfeature $\\to$ spatial map",
    r"$Z^{(l,t)}=F_l(H^{(l,t)})$")
box(3, yA, H2, "Cross-seed consistency", "centroid dispersion\nacross $m$ seeds",
    r"$C_D=\mathbb{E}_{l,t}\!\left[1-D^{(l,t)}_{\mathrm{norm}}\right]$")
box(3, yB, H2, "Text–image attention", "concept tokens $\\leftrightarrow$\nimage regions",
    r"$A=\mathbb{E}_{c_i,l,t}\!\left[A_i^{(l,t)}\right]$")
box(4, yc, H1, "Harmonic\nmean", "reference-free score\nper generated image",
    r"$\mathrm{MIRAGE}=\dfrac{2\,C_D\,A}{C_D+A}$", formula_y=yc - H1 / 2 + 0.20)

# optional dashed group around the model-internal part (off by default)
SHOW_GROUP = False
if SHOW_GROUP:
    gx0 = lefts[2] - 0.07
    gx1 = lefts[3] + widths[3] + 0.07
    gy0, gy1 = yB - H2 / 2 - 0.07, yA + H2 / 2 + 0.07
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#F3F3F3", ec="#555555", lw=LW, ls=(0, (3, 2)), zorder=1))
    ax.text(gx0 + 0.04, gy1 + 0.04, "Model-internal signals (no reference image)", ha="left", va="bottom",
            fontsize=5.6, style="italic", color="#333333")


def arrow(xa, ya, xb, yb):
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", lw=LW, color="black", shrinkA=0, shrinkB=0, mutation_scale=6),
                zorder=2)


def line(xa, ya, xb, yb):
    ax.plot([xa, xb], [ya, yb], color="black", lw=LW, solid_capstyle="butt", zorder=2)


R = lambda c: lefts[c] + widths[c]
L = lambda c: lefts[c]
arrow(R(0), yc, L(1), yc)
arrow(R(1), yc, L(2), yc)
xm = (R(2) + L(3)) / 2          # fork 3 -> 4a / 4b
line(R(2), yc, xm, yc); line(xm, yB, xm, yA)
arrow(xm, yA, L(3), yA); arrow(xm, yB, L(3), yB)
xm2 = (R(3) + L(4)) / 2         # merge 4a / 4b -> 5
line(R(3), yA, xm2, yA); line(R(3), yB, xm2, yB); line(xm2, yB, xm2, yA)
arrow(xm2, yc, L(4), yc)

fig.savefig("writeups/mirage_pipeline_plain.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("writeups/mirage_pipeline_plain.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
print("wrote writeups/mirage_pipeline_plain.{pdf,png}")
