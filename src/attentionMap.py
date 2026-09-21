# ================================================================
# VISUALIZE FLUX ATTENTION MAPS
# ================================================================
#
# For every sample under flux_results/attention_maps/<slug>/, loads
# each concept's saved .npy spatial attention map, upsamples it to
# the generated image's resolution, and overlays it as a heatmap --
# one figure per sample, one panel per concept (plus the plain
# image). Saved to flux_results/graphs/<slug>_attention.png.
# ================================================================

import os
import sys
import glob

# Force a non-interactive backend BEFORE importing pyplot. On a
# headless SSH session, matplotlib can default to a GUI backend
# (Tk, Qt) that either errors or silently misbehaves with no
# display attached -- Agg just rasterizes to a file, which is all
# this script does anyway.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = "/DATA/swagata/akshat_ICLR"
RESULTS_DIR = os.path.join(PROJECT_DIR, "flux_results")
ATTENTION_DIR = os.path.join(RESULTS_DIR, "attention_maps")
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")


def visualize_sample(slug):

    sample_dir = os.path.join(ATTENTION_DIR, slug)
    npy_files = sorted(glob.glob(os.path.join(sample_dir, "*.npy")))

    if not npy_files:
        print(f"  [skip] no .npy files in {sample_dir}", flush=True)
        return

    image_path = os.path.join(IMAGE_DIR, f"{slug}.png")
    image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None

    if image is None:
        print(f"  [warn] no matching image at {image_path} -- plotting raw grids instead", flush=True)

    n_concepts = len(npy_files)
    n_cols = n_concepts + (1 if image is not None else 0)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.3))
    if n_cols == 1:
        axes = [axes]

    col = 0

    # Plain generated image first, if we have it.
    if image is not None:
        axes[col].imshow(image)
        axes[col].set_title("Generated Image", fontsize=11)
        axes[col].axis("off")
        col += 1

    for path in npy_files:

        concept = os.path.splitext(os.path.basename(path))[0]
        attention_map = np.load(path)

        ax = axes[col]

        if image is not None:

            # Upsample the (small, e.g. 32x32) attention grid to the
            # image's actual resolution before overlaying.
            heat = (attention_map / (attention_map.max() + 1e-12) * 255).astype(np.uint8)
            heat_resized = np.array(
                Image.fromarray(heat).resize(image.size, Image.BILINEAR)
            )

            ax.imshow(image)
            ax.imshow(heat_resized, cmap="magma", alpha=0.55)

        else:
            # No matching image on disk -- just show the raw grid.
            ax.imshow(attention_map, cmap="magma")

        ax.set_title(concept, fontsize=11)
        ax.axis("off")
        col += 1

    plt.suptitle(slug.replace("_", " ").title(), fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(GRAPH_DIR, f"{slug}_attention.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [ok] saved {out_path}", flush=True)


def main():

    # Print every path being used FIRST, before touching the
    # filesystem at all -- if nothing after this block ever prints,
    # the problem is upstream of this script (wrong conda env,
    # import hanging, etc.), not a silent failure inside it.
    print("=" * 70, flush=True)
    print("VISUALIZE FLUX ATTENTION MAPS", flush=True)
    print("=" * 70, flush=True)
    print("PROJECT_DIR   :", PROJECT_DIR, flush=True)
    print("ATTENTION_DIR :", ATTENTION_DIR, flush=True)
    print("IMAGE_DIR     :", IMAGE_DIR, flush=True)
    print("GRAPH_DIR     :", GRAPH_DIR, flush=True)
    print(flush=True)

    if not os.path.isdir(ATTENTION_DIR):
        print(f"ERROR: attention_maps directory does not exist: {ATTENTION_DIR}", flush=True)
        print("Check PROJECT_DIR above matches where flux.py actually wrote its output.", flush=True)
        sys.exit(1)

    os.makedirs(GRAPH_DIR, exist_ok=True)

    sample_slugs = sorted(
        d for d in os.listdir(ATTENTION_DIR)
        if os.path.isdir(os.path.join(ATTENTION_DIR, d))
    )

    print(f"Found {len(sample_slugs)} sample folders", flush=True)

    if not sample_slugs:
        print(f"Nothing to do -- {ATTENTION_DIR} has no subfolders.", flush=True)
        return

    for slug in sample_slugs:
        print(f"Processing {slug} ...", flush=True)
        visualize_sample(slug)

    print(flush=True)
    print("Done. Figures saved under:", GRAPH_DIR, flush=True)


if __name__ == "__main__":

    try:
        main()
    except Exception:
        # Make absolutely sure a failure is never silent.
        import traceback
        print("\nFATAL ERROR:", flush=True)
        traceback.print_exc()
        sys.exit(1)