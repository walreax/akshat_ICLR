"""
visualise_sae.py

Visualization companion for flux_sae.py.

Reads the saved SAE metrics AND SAE concept maps produced by flux_sae.py.
It never reruns FLUX.

Expected layout:
    ~/akshat_ICLR/
        flux_results/
            images/
                sample_01_*.png
                ...
            attention_maps/
                sample_01_*/
                    *.npy
            sae/
                metrics/
                    all_sae_metrics.json
                    sample_*.json
                concept_maps/
                    sample_01_*/
                        double_early_t1.0/
                            concept.npy
                        ...
                weights/
                sae_diagnostics.json

Outputs:
    flux_results/sae/visualizations/
        metric plots
        concept_maps/
            sample_XX/
                <block>_<timestep>/
                    <concept>_map.png
                    <concept>_overlay.png
                ...
        comparisons/
            sample_XX/
                <block>_<timestep>/
                    <concept>_attention_vs_sae.png
        montages/
            sample_XX/
                <block>_<timestep>_all_concepts.png
                <concept>_evolution.png
        concept_map_index.json
        sae_summary.json

Useful commands:
    python -u visualise_sae.py
    python -u visualise_sae.py --sample 1
    python -u visualise_sae.py --sample 1 --concept bird
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BUCKET_ORDER = ["t1.0", "t0.5", "t0.0"]
BUCKET_LABEL = {
    "t1.0": "Early (t=1.0)",
    "t0.5": "Middle (t=0.5)",
    "t0.0": "Final (t=0.0)",
}

BLOCK_ORDER = [
    "double_early",
    "double_mid",
    "double_late",
    "single_early",
    "single_mid",
    "single_late",
]

BLOCK_LABEL = {
    "double_early": "Double Early",
    "double_mid": "Double Mid",
    "double_late": "Double Late",
    "single_early": "Single Early",
    "single_mid": "Single Mid",
    "single_late": "Single Late",
}

METRIC_ORDER = [
    "centroid_deviation",
    "cross_concept_index",
    "attention_entropy",
    "attention_concentration_index",
]

METRIC_LABEL = {
    "centroid_deviation": "Centroid Deviation",
    "cross_concept_index": "Cross-Concept Index",
    "attention_entropy": "Entropy",
    "attention_concentration_index": "Concentration",
}


def slugify(text):
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def savefig(path):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def load_metrics(metrics_dir):
    combined = os.path.join(metrics_dir, "all_sae_metrics.json")
    if os.path.exists(combined):
        data = load_json(combined)
        if isinstance(data, list):
            return data

    records = []
    if not os.path.isdir(metrics_dir):
        return records

    for name in sorted(os.listdir(metrics_dir)):
        if not name.startswith("sample_") or not name.endswith(".json"):
            continue
        records.append(load_json(os.path.join(metrics_dir, name)))
    return records


def extract_sample_meta(record, fallback_id):
    exp = record.get("experiment", {}) if isinstance(record, dict) else {}
    return {
        "sample_id": exp.get("sample_id", fallback_id),
        "title": exp.get("title", f"Sample {fallback_id}"),
        "prompt": exp.get("prompt", ""),
        "source": exp.get("source", ""),
    }


def get_pair_metric(entry, metric_name):
    values = []
    obj = entry.get(metric_name, {})
    if not isinstance(obj, dict):
        return np.nan

    for pair_value in obj.values():
        if metric_name == "centroid_deviation":
            if isinstance(pair_value, dict):
                value = pair_value.get("distance")
            else:
                value = pair_value
        else:
            value = pair_value
        value = safe_float(value)
        if value is not None:
            values.append(value)

    return float(np.mean(values)) if values else np.nan


def get_config_rows(records):
    rows = []

    for idx, record in enumerate(records, start=1):
        meta = extract_sample_meta(record, idx)

        for key, entry in record.items():
            if key == "experiment" or not isinstance(entry, dict):
                continue
            if "__" not in key or "sae" not in entry:
                continue

            block, bucket = key.rsplit("__", 1)
            if bucket not in BUCKET_ORDER:
                continue

            concept_metrics = entry.get("concept_metrics", {})
            entropy_values = []
            concentration_values = []

            for cinfo in concept_metrics.values():
                if not isinstance(cinfo, dict):
                    continue
                e = safe_float(cinfo.get("attention_entropy"))
                a = safe_float(cinfo.get("attention_concentration_index"))
                if e is not None:
                    entropy_values.append(e)
                if a is not None:
                    concentration_values.append(a)

            cid_values = []
            assignments = entry.get("sae", {}).get("cid_assignments", {})
            for info in assignments.values():
                if isinstance(info, dict):
                    s = safe_float(info.get("cosine_similarity"))
                    if s is not None:
                        cid_values.append(s)

            rows.append({
                "sample_id": meta["sample_id"],
                "title": meta["title"],
                "source": meta["source"],
                "block": block,
                "bucket": bucket,
                "centroid_deviation": get_pair_metric(entry, "centroid_deviation"),
                "cross_concept_index": get_pair_metric(entry, "cross_concept_index"),
                "attention_entropy": (
                    float(np.mean(entropy_values))
                    if entropy_values else np.nan
                ),
                "attention_concentration_index": (
                    float(np.mean(concentration_values))
                    if concentration_values else np.nan
                ),
                "cid_cosine_similarity": (
                    float(np.mean(cid_values))
                    if cid_values else np.nan
                ),
            })

    return rows


def aggregate(rows, block=None, bucket=None, sample_id=None):
    selected = rows

    if block is not None:
        selected = [r for r in selected if r["block"] == block]
    if bucket is not None:
        selected = [r for r in selected if r["bucket"] == bucket]
    if sample_id is not None:
        selected = [
            r for r in selected
            if str(r["sample_id"]) == str(sample_id)
        ]

    return selected


def nanmean(values):
    values = [
        v for v in values
        if v is not None and np.isfinite(v)
    ]
    return float(np.mean(values)) if values else np.nan


# ================================================================
# METRIC VISUALIZATIONS
# ================================================================

def plot_sample_metric_heatmaps(rows, outdir):
    samples = sorted(
        {str(r["sample_id"]) for r in rows},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    configs = [
        f"{b}__{t}"
        for b in BLOCK_ORDER
        for t in BUCKET_ORDER
    ]

    for metric in METRIC_ORDER:
        matrix = np.full(
            (len(samples), len(configs)),
            np.nan,
        )

        for i, sid in enumerate(samples):
            for j, cfg in enumerate(configs):
                block, bucket = cfg.rsplit("__", 1)
                matches = [
                    r for r in rows
                    if str(r["sample_id"]) == sid
                    and r["block"] == block
                    and r["bucket"] == bucket
                ]
                if matches:
                    matrix[i, j] = nanmean(
                        [m[metric] for m in matches]
                    )

        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(
            f"SAE {METRIC_LABEL[metric]} — "
            "Sample × Block × Timestep"
        )
        ax.set_ylabel("Sample")
        ax.set_xlabel("SAE configuration")
        ax.set_yticks(range(len(samples)))
        ax.set_yticklabels(samples)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(
            [
                f"{BLOCK_LABEL[b]}\n{t}"
                for b, t in [
                    c.rsplit("__", 1)
                    for c in configs
                ]
            ],
            rotation=45,
            ha="right",
        )
        fig.colorbar(
            im,
            ax=ax,
            label=METRIC_LABEL[metric],
        )

        savefig(
            os.path.join(
                outdir,
                f"01_{slugify(metric)}_sample_heatmap.png",
            )
        )


def plot_mean_block_time(rows, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, metric in zip(axes.ravel(), METRIC_ORDER):
        matrix = np.full(
            (len(BLOCK_ORDER), len(BUCKET_ORDER)),
            np.nan,
        )

        for i, block in enumerate(BLOCK_ORDER):
            for j, bucket in enumerate(BUCKET_ORDER):
                selected = aggregate(
                    rows,
                    block=block,
                    bucket=bucket,
                )
                matrix[i, j] = nanmean(
                    [r[metric] for r in selected]
                )

        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(METRIC_LABEL[metric])
        ax.set_yticks(range(len(BLOCK_ORDER)))
        ax.set_yticklabels(
            [BLOCK_LABEL[b] for b in BLOCK_ORDER]
        )
        ax.set_xticks(range(len(BUCKET_ORDER)))
        ax.set_xticklabels(BUCKET_ORDER)
        ax.set_xlabel("Timestep")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                    )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.suptitle(
        "Mean SAE Metrics Across Samples",
        fontsize=16,
    )

    savefig(
        os.path.join(
            outdir,
            "02_mean_metric_by_block_time.png",
        )
    )


def plot_time_evolution(rows, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    x = np.arange(len(BUCKET_ORDER))

    for ax, metric in zip(axes.ravel(), METRIC_ORDER):
        for block in BLOCK_ORDER:
            means = []
            stds = []

            for bucket in BUCKET_ORDER:
                selected = aggregate(
                    rows,
                    block=block,
                    bucket=bucket,
                )
                vals = [
                    r[metric]
                    for r in selected
                    if np.isfinite(r[metric])
                ]

                means.append(
                    np.mean(vals) if vals else np.nan
                )
                stds.append(
                    np.std(vals) if vals else np.nan
                )

            ax.plot(
                x,
                means,
                marker="o",
                label=BLOCK_LABEL[block],
            )

            means = np.asarray(
                means,
                dtype=float,
            )
            stds = np.asarray(
                stds,
                dtype=float,
            )

            ax.fill_between(
                x,
                means - stds,
                means + stds,
                alpha=0.10,
            )

        ax.set_title(METRIC_LABEL[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [BUCKET_LABEL[b] for b in BUCKET_ORDER]
        )
        ax.set_xlabel("Diffusion stage")
        ax.set_ylabel("Metric value")
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(
        fontsize=8,
        ncol=2,
    )

    fig.suptitle(
        "SAE Metric Evolution Across Diffusion Time",
        fontsize=16,
    )

    savefig(
        os.path.join(
            outdir,
            "03_time_evolution.png",
        )
    )


def plot_block_comparison(rows, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(BLOCK_ORDER))

    for ax, metric in zip(axes.ravel(), METRIC_ORDER):
        width = 0.25

        for j, bucket in enumerate(BUCKET_ORDER):
            vals = []

            for block in BLOCK_ORDER:
                selected = aggregate(
                    rows,
                    block=block,
                    bucket=bucket,
                )
                vals.append(
                    nanmean(
                        [r[metric] for r in selected]
                    )
                )

            ax.bar(
                x + (j - 1) * width,
                vals,
                width=width,
                label=BUCKET_LABEL[bucket],
            )

        ax.set_title(METRIC_LABEL[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [BLOCK_LABEL[b] for b in BLOCK_ORDER],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("Mean value")
        ax.grid(
            axis="y",
            alpha=0.25,
        )

    axes[0, 0].legend(fontsize=8)

    fig.suptitle(
        "SAE Block Comparison",
        fontsize=16,
    )

    savefig(
        os.path.join(
            outdir,
            "04_block_comparison.png",
        )
    )


def plot_pairwise_heatmaps(records, outdir):
    pair_values = {
        m: defaultdict(list)
        for m in [
            "centroid_deviation",
            "cross_concept_index",
        ]
    }

    for record in records:
        for key, entry in record.items():
            if (
                key == "experiment"
                or not isinstance(entry, dict)
                or "__" not in key
            ):
                continue

            block, bucket = key.rsplit("__", 1)

            if bucket not in BUCKET_ORDER:
                continue

            for metric in pair_values:
                obj = entry.get(metric, {})

                for pair, value in obj.items():
                    if (
                        metric == "centroid_deviation"
                        and isinstance(value, dict)
                    ):
                        value = value.get("distance")

                    value = safe_float(value)

                    if value is not None:
                        pair_values[metric][pair].append(
                            value
                        )

    for metric, pairs in pair_values.items():
        if not pairs:
            continue

        names = sorted(pairs)
        vals = [
            np.mean(pairs[p])
            for p in names
        ]

        fig, ax = plt.subplots(
            figsize=(max(9, len(names) * 0.8), 5.5)
        )

        ax.bar(
            np.arange(len(names)),
            vals,
        )

        ax.set_title(
            f"Mean {METRIC_LABEL[metric]} — "
            "Concrete vs Metaphorical Pairs"
        )
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(
            names,
            rotation=45,
            ha="right",
        )
        ax.grid(
            axis="y",
            alpha=0.25,
        )

        savefig(
            os.path.join(
                outdir,
                f"05_pairwise_{slugify(metric)}.png",
            )
        )


def plot_cid_similarity(rows, outdir):
    matrix = np.full(
        (len(BLOCK_ORDER), len(BUCKET_ORDER)),
        np.nan,
    )

    for i, block in enumerate(BLOCK_ORDER):
        for j, bucket in enumerate(BUCKET_ORDER):
            selected = aggregate(
                rows,
                block=block,
                bucket=bucket,
            )
            matrix[i, j] = nanmean(
                [r["cid_cosine_similarity"] for r in selected]
            )

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        "Mean Concept → CID Cosine Similarity"
    )
    ax.set_yticks(range(len(BLOCK_ORDER)))
    ax.set_yticklabels(
        [BLOCK_LABEL[b] for b in BLOCK_ORDER]
    )
    ax.set_xticks(range(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_xlabel("Timestep")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        im,
        ax=ax,
        label="Cosine similarity",
    )

    savefig(
        os.path.join(
            outdir,
            "06_cid_cosine_similarity.png",
        )
    )


def plot_explained_variance(sae_results_dir, outdir):
    path = os.path.join(
        sae_results_dir,
        "sae_diagnostics.json",
    )

    if not os.path.exists(path):
        print(
            "Skipping explained variance:",
            path,
            "not found",
        )
        return

    diagnostics = load_json(path)
    labels = []
    values = []

    for key, info in diagnostics.items():
        ev = safe_float(
            info.get("explained_variance")
        )

        if ev is None:
            continue

        labels.append(key)
        values.append(100.0 * ev)

    if not values:
        return

    order = np.argsort(values)

    labels = [
        labels[i]
        for i in order
    ]

    values = [
        values[i]
        for i in order
    ]

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.barh(
        np.arange(len(labels)),
        values,
    )

    ax.set_yticks(
        np.arange(len(labels))
    )
    ax.set_yticklabels(
        labels,
        fontsize=8,
    )
    ax.set_xlabel(
        "Explained variance (%)"
    )
    ax.set_title(
        "SAE Reconstruction Quality"
    )
    ax.grid(
        axis="x",
        alpha=0.25,
    )

    savefig(
        os.path.join(
            outdir,
            "07_sae_explained_variance.png",
        )
    )


# ================================================================
# CONCEPT MAP VISUALIZATION
# ================================================================

def find_sample_image(project_dir, sample_id, sample_title=""):
    """
    flux.py saves images as:
        flux_results/images/<slug>.png

    Try the exact expected slug first, then fall back to a prefix
    search so this remains robust to title changes.
    """
    image_dir = os.path.join(
        project_dir,
        "flux_results",
        "images",
    )

    if not os.path.isdir(image_dir):
        return None

    expected_prefix = (
        f"sample_{int(sample_id):02d}_"
        if str(sample_id).isdigit()
        else f"sample_{sample_id}_"
    )

    candidates = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.startswith(expected_prefix)
        and name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if candidates:
        return sorted(candidates)[0]

    if sample_title:
        prefix = (
            f"sample_{int(sample_id):02d}_"
            f"{slugify(sample_title)}"
        )

        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = os.path.join(
                image_dir,
                prefix + ext,
            )
            if os.path.exists(candidate):
                return candidate

    return None


def find_attention_map(project_dir, sample_id, concept):
    """
    Locate the raw attention map produced by flux.py.
    """
    attention_dir = os.path.join(
        project_dir,
        "flux_results",
        "attention_maps",
    )

    if not os.path.isdir(attention_dir):
        return None

    sample_prefix = (
        f"sample_{int(sample_id):02d}_"
        if str(sample_id).isdigit()
        else f"sample_{sample_id}_"
    )

    sample_dirs = [
        os.path.join(attention_dir, name)
        for name in os.listdir(attention_dir)
        if name.startswith(sample_prefix)
        and os.path.isdir(os.path.join(attention_dir, name))
    ]

    if not sample_dirs:
        return None

    for sample_dir in sorted(sample_dirs):
        candidate = os.path.join(
            sample_dir,
            f"{concept}.npy",
        )
        if os.path.exists(candidate):
            return candidate

    return None


def load_concept_maps(sae_map_dir):
    """
    Returns:
        {
          sample_slug: {
            config_name: {
              concept: Path
            }
          }
        }
    """
    result = {}

    if not os.path.isdir(sae_map_dir):
        return result

    for sample_slug in sorted(os.listdir(sae_map_dir)):
        sample_dir = os.path.join(
            sae_map_dir,
            sample_slug,
        )

        if not os.path.isdir(sample_dir):
            continue

        result[sample_slug] = {}

        for config in sorted(os.listdir(sample_dir)):
            config_dir = os.path.join(
                sample_dir,
                config,
            )

            if not os.path.isdir(config_dir):
                continue

            concept_files = {}

            for name in sorted(os.listdir(config_dir)):
                if not name.endswith(".npy"):
                    continue

                concept = name[:-4]
                concept_files[concept] = os.path.join(
                    config_dir,
                    name,
                )

            if concept_files:
                result[sample_slug][config] = concept_files

    return result


def sample_slug_for_record(record):
    exp = record.get("experiment", {})
    sample_id = exp.get("sample_id")
    title = exp.get("title", "")

    if sample_id is None:
        return None

    return (
        f"sample_{int(sample_id):02d}_"
        f"{slugify(title)}"
    )


def normalise_map(amap):
    amap = np.asarray(amap, dtype=np.float32)
    amap = np.squeeze(amap)

    if amap.ndim != 2:
        raise ValueError(
            f"Expected 2D concept map, got shape {amap.shape}"
        )

    amap = np.nan_to_num(
        amap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = float(amap.min())
    hi = float(amap.max())

    if hi > lo:
        return (amap - lo) / (hi - lo)

    return np.zeros_like(amap)


def resize_map_to_image(amap, image_size):
    """
    image_size = (width, height)
    """
    image_w, image_h = image_size

    pil = Image.fromarray(
        np.uint8(
            np.clip(amap, 0.0, 1.0) * 255
        )
    )

    pil = pil.resize(
        (image_w, image_h),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(pil, dtype=np.float32) / 255.0


def plot_single_concept_map(
    amap_path,
    image_path,
    sample_slug,
    config,
    concept,
    cid_info,
    output_dir,
):
    amap = normalise_map(
        np.load(amap_path)
    )

    fig, axes = plt.subplots(
        1,
        3 if image_path else 1,
        figsize=(15 if image_path else 5, 5),
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_map = axes[0]

    ax_map.imshow(
        amap,
        cmap="magma",
        interpolation="nearest",
    )
    ax_map.set_title(
        f"SAE concept map\n{concept}"
    )
    ax_map.axis("off")

    if image_path:
        image = Image.open(image_path).convert("RGB")

        axes[1].imshow(image)
        axes[1].set_title("Generated image")
        axes[1].axis("off")

        resized = resize_map_to_image(
            amap,
            image.size,
        )

        axes[2].imshow(image)
        axes[2].imshow(
            resized,
            cmap="magma",
            alpha=0.50,
            interpolation="bilinear",
        )

        title = "SAE activation overlay"

        if cid_info:
            cid = cid_info.get("cid")
            sim = cid_info.get(
                "cosine_similarity"
            )

            if cid is not None:
                title += f"\nCID {cid}"

            if sim is not None:
                title += f" | cosine={sim:.3f}"

        axes[2].set_title(title)
        axes[2].axis("off")

    fig.suptitle(
        f"{sample_slug} | {config} | {concept}",
        fontsize=13,
    )

    out = os.path.join(
        output_dir,
        f"{slugify(concept)}_map.png",
    )

    savefig(out)

    return out


def plot_attention_vs_sae(
    project_dir,
    sample_id,
    concept,
    sae_map,
    config,
    output_dir,
):
    """
    Compare the original raw-attention map with the SAE-derived map.

    Raw attention is only used here for visualization; SAE metrics/maps
    remain exactly those saved by flux_sae.py.
    """
    attention_path = find_attention_map(
        project_dir,
        sample_id,
        concept,
    )

    if attention_path is None:
        return None

    try:
        attention = normalise_map(
            np.load(attention_path)
        )
    except Exception as exc:
        print(
            "Could not load attention map:",
            attention_path,
            exc,
        )
        return None

    sae_map = normalise_map(sae_map)

    if attention.shape != sae_map.shape:
        attention_img = Image.fromarray(
            np.uint8(attention * 255)
        ).resize(
            (sae_map.shape[1], sae_map.shape[0]),
            Image.Resampling.BILINEAR,
        )
        attention = (
            np.asarray(attention_img)
            .astype(np.float32)
            / 255.0
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        attention,
        cmap="magma",
    )
    axes[0].set_title(
        "Raw FLUX attention"
    )
    axes[0].axis("off")

    axes[1].imshow(
        sae_map,
        cmap="magma",
    )
    axes[1].set_title(
        "SAE concept map"
    )
    axes[1].axis("off")

    axes[2].imshow(
        attention,
        cmap="magma",
        alpha=0.50,
    )
    axes[2].imshow(
        sae_map,
        cmap="viridis",
        alpha=0.50,
    )
    axes[2].set_title(
        "Attention + SAE comparison"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Attention vs SAE | sample {sample_id} | "
        f"{config} | {concept}",
        fontsize=13,
    )

    out = os.path.join(
        output_dir,
        f"{slugify(concept)}_attention_vs_sae.png",
    )

    savefig(out)

    return out


def plot_config_concept_montage(
    concept_paths,
    image_path,
    config,
    output_path,
    max_concepts=12,
):
    concepts = sorted(concept_paths)[:max_concepts]

    if not concepts:
        return

    n = len(concepts)

    if image_path:
        cols = 4
        rows = int(np.ceil((n + 1) / cols))
    else:
        cols = 4
        rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    cursor = 0

    if image_path:
        image = Image.open(
            image_path
        ).convert("RGB")

        axes[cursor].imshow(image)
        axes[cursor].set_title(
            "Generated image"
        )
        axes[cursor].axis("off")
        cursor += 1

    for concept in concepts:
        amap = normalise_map(
            np.load(concept_paths[concept])
        )

        axes[cursor].imshow(
            amap,
            cmap="magma",
        )
        axes[cursor].set_title(
            concept,
            fontsize=9,
        )
        axes[cursor].axis("off")
        cursor += 1

    for ax in axes[cursor:]:
        ax.axis("off")

    fig.suptitle(
        f"{config} — SAE concept maps",
        fontsize=14,
    )

    savefig(output_path)


def plot_concept_evolution(
    concept,
    config_paths,
    image_path,
    output_path,
):
    """
    Show one concept across the three diffusion stages for each block.

    Layout:
        rows = blocks
        columns = t=1.0, t=0.5, t=0.0
    """
    available = []

    for block in BLOCK_ORDER:
        row = []

        for bucket in BUCKET_ORDER:
            config = f"{block}_{bucket}"

            if config in config_paths:
                path = config_paths[config].get(concept)
            else:
                path = None

            row.append(path)

        if any(path is not None for path in row):
            available.append(
                (block, row)
            )

    if not available:
        return

    fig, axes = plt.subplots(
        len(available),
        3,
        figsize=(12, 3.2 * len(available)),
        squeeze=False,
    )

    for i, (block, row) in enumerate(available):
        for j, path in enumerate(row):
            ax = axes[i, j]

            if path is None:
                ax.axis("off")
                continue

            amap = normalise_map(
                np.load(path)
            )

            ax.imshow(
                amap,
                cmap="magma",
            )

            ax.set_title(
                f"{BLOCK_LABEL.get(block, block)}\n"
                f"{BUCKET_LABEL[BUCKET_ORDER[j]]}",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle(
        f"SAE concept evolution: {concept}",
        fontsize=15,
    )

    savefig(output_path)


def visualise_concept_maps(
    project_dir,
    records,
    sae_results_dir,
    outdir,
    sample_filter=None,
    concept_filter=None,
):
    sae_map_dir = os.path.join(
        sae_results_dir,
        "concept_maps",
    )

    concept_data = load_concept_maps(
        sae_map_dir
    )

    if not concept_data:
        print(
            "No SAE concept maps found in:",
            sae_map_dir,
        )
        return

    print(
        f"Found {len(concept_data)} sample "
        "concept-map directories."
    )

    record_by_slug = {}

    for record in records:
        slug = sample_slug_for_record(record)

        if slug:
            record_by_slug[slug] = record

    map_root = os.path.join(
        outdir,
        "concept_maps",
    )
    comparison_root = os.path.join(
        outdir,
        "comparisons",
    )
    montage_root = os.path.join(
        outdir,
        "montages",
    )

    index = []

    for sample_slug, configs in concept_data.items():
        record = record_by_slug.get(
            sample_slug,
            {},
        )
        exp = record.get(
            "experiment",
            {},
        )

        sample_id = exp.get(
            "sample_id",
            sample_slug,
        )

        if (
            sample_filter is not None
            and str(sample_id) != str(sample_filter)
        ):
            continue

        title = exp.get(
            "title",
            sample_slug,
        )

        image_path = find_sample_image(
            project_dir,
            sample_id,
            title,
        )

        sample_map_root = os.path.join(
            map_root,
            sample_slug,
        )
        sample_comparison_root = os.path.join(
            comparison_root,
            sample_slug,
        )
        sample_montage_root = os.path.join(
            montage_root,
            sample_slug,
        )

        sample_entry = {
            "sample_id": sample_id,
            "sample_slug": sample_slug,
            "title": title,
            "image": image_path,
            "configs": {},
        }

        print()
        print(
            "=" * 70
        )
        print(
            f"VISUALIZING SAE CONCEPT MAPS: "
            f"{sample_slug}"
        )
        print(
            "=" * 70
        )

        # ------------------------------------------------------------
        # Individual maps + image overlays + attention comparison
        # ------------------------------------------------------------
        for config, concept_paths in configs.items():
            if concept_filter:
                concept_paths = {
                    c: p
                    for c, p in concept_paths.items()
                    if c == concept_filter
                }

            if not concept_paths:
                continue

            config_map_dir = os.path.join(
                sample_map_root,
                config,
            )
            config_compare_dir = os.path.join(
                sample_comparison_root,
                config,
            )

            os.makedirs(
                config_map_dir,
                exist_ok=True,
            )
            os.makedirs(
                config_compare_dir,
                exist_ok=True,
            )

            # Find the matching JSON entry to retrieve CID metadata.
            cid_info = {}

            entry = record.get(
                config.replace("_t", "__t")
            )

            # More robust lookup: config is e.g. double_early_t1.0.
            if "__" not in config:
                parts = config.rsplit("_t", 1)
                if len(parts) == 2:
                    block = parts[0]
                    bucket = "t" + parts[1]
                    entry = record.get(
                        f"{block}__{bucket}"
                    )

            if isinstance(entry, dict):
                cid_info = (
                    entry.get(
                        "sae",
                        {}
                    ).get(
                        "cid_assignments",
                        {},
                    )
                )

            sample_entry["configs"].setdefault(
                config,
                {},
            )

            for concept, path in concept_paths.items():
                try:
                    amap = normalise_map(
                        np.load(path)
                    )
                except Exception as exc:
                    print(
                        "Skipping invalid map:",
                        path,
                        exc,
                    )
                    continue

                concept_info = cid_info.get(
                    concept,
                    {},
                )

                map_png = plot_single_concept_map(
                    path,
                    image_path,
                    sample_slug,
                    config,
                    concept,
                    concept_info,
                    config_map_dir,
                )

                compare_png = plot_attention_vs_sae(
                    project_dir,
                    sample_id,
                    concept,
                    amap,
                    config,
                    config_compare_dir,
                )

                sample_entry["configs"][config][
                    concept
                ] = {
                    "npy": path,
                    "map_png": map_png,
                    "attention_vs_sae_png": compare_png,
                    "cid": concept_info.get("cid"),
                    "cosine_similarity": concept_info.get(
                        "cosine_similarity"
                    ),
                    "shape": list(amap.shape),
                    "min": float(amap.min()),
                    "max": float(amap.max()),
                    "mean": float(amap.mean()),
                }

            # --------------------------------------------------------
            # Montage for this block/timestep
            # --------------------------------------------------------
            montage_path = os.path.join(
                sample_montage_root,
                f"{config}_all_concepts.png",
            )

            plot_config_concept_montage(
                concept_paths,
                image_path,
                config,
                montage_path,
            )

        # ------------------------------------------------------------
        # Concept evolution across block/timestep
        # ------------------------------------------------------------
        all_concepts = sorted({
            concept
            for configs_for_sample in configs.values()
            for concept in configs_for_sample
        })

        if concept_filter:
            all_concepts = [
                c
                for c in all_concepts
                if c == concept_filter
            ]

        for concept in all_concepts:
            evolution_path = os.path.join(
                sample_montage_root,
                f"{slugify(concept)}_evolution.png",
            )

            plot_concept_evolution(
                concept,
                configs,
                image_path,
                evolution_path,
            )

        index.append(sample_entry)

    index_path = os.path.join(
        outdir,
        "concept_map_index.json",
    )

    with open(
        index_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            index,
            f,
            indent=4,
        )

    print(
        "Saved:",
        index_path,
    )


# ================================================================
# SUMMARY
# ================================================================

def write_summary(rows, sae_results_dir, outdir):
    summary = {
        "n_samples": len({
            str(r["sample_id"])
            for r in rows
        }),
        "n_configurations": len({
            (r["block"], r["bucket"])
            for r in rows
        }),
        "blocks": BLOCK_ORDER,
        "timesteps": BUCKET_ORDER,
        "metrics": {},
    }

    for metric in METRIC_ORDER + [
        "cid_cosine_similarity"
    ]:
        vals = [
            r[metric]
            for r in rows
            if np.isfinite(r[metric])
        ]

        summary["metrics"][metric] = {
            "mean": (
                float(np.mean(vals))
                if vals else None
            ),
            "std": (
                float(np.std(vals))
                if vals else None
            ),
            "min": (
                float(np.min(vals))
                if vals else None
            ),
            "max": (
                float(np.max(vals))
                if vals else None
            ),
            "n": len(vals),
        }

    diagnostics_path = os.path.join(
        sae_results_dir,
        "sae_diagnostics.json",
    )

    if os.path.exists(diagnostics_path):
        diagnostics = load_json(
            diagnostics_path
        )

        ev = [
            safe_float(
                info.get(
                    "explained_variance"
                )
            )
            for info in diagnostics.values()
            if safe_float(
                info.get(
                    "explained_variance"
                )
            ) is not None
        ]

        if ev:
            summary["sae_explained_variance"] = {
                "mean": float(np.mean(ev)),
                "std": float(np.std(ev)),
                "min": float(np.min(ev)),
                "max": float(np.max(ev)),
            }

    path = os.path.join(
        outdir,
        "sae_summary.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print(
        "Saved:",
        path,
    )


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved FLUX SAE metrics "
            "and concept maps."
        )
    )

    parser.add_argument(
        "--project-dir",
        default=DEFAULT_PROJECT_DIR,
        help="Project directory containing flux_results/",
    )

    parser.add_argument(
        "--sample",
        default=None,
        help="Optional sample id, e.g. 1 or 01",
    )

    parser.add_argument(
        "--concept",
        default=None,
        help="Optional exact concept name, e.g. bird",
    )

    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only make metric plots.",
    )

    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Only make concept-map visualizations.",
    )

    args = parser.parse_args()

    project_dir = os.path.abspath(
        args.project_dir
    )

    print("=" * 70)
    print("FLUX SAE VISUALIZATION")
    print("=" * 70)
    print(
        "Project directory:",
        project_dir,
    )

    sae_results_dir = os.path.join(
        project_dir,
        "flux_results",
        "sae",
    )

    metrics_dir = os.path.join(
        sae_results_dir,
        "metrics",
    )

    outdir = os.path.join(
        sae_results_dir,
        "visualizations",
    )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    print(
        "SAE results directory:",
        sae_results_dir,
    )
    print(
        "Metrics directory:",
        metrics_dir,
    )
    print(
        "Visualization directory:",
        outdir,
    )

    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(
            f"SAE metrics directory not found: "
            f"{metrics_dir}"
        )

    records = load_metrics(
        metrics_dir
    )

    if not records:
        raise RuntimeError(
            "No SAE metric JSON files found in "
            f"{metrics_dir}"
        )

    rows = get_config_rows(
        records
    )

    if args.sample is not None:
        rows = [
            r for r in rows
            if str(r["sample_id"])
            == str(args.sample)
        ]

        records = [
            r for r in records
            if str(
                r.get(
                    "experiment",
                    {}
                ).get(
                    "sample_id",
                    ""
                )
            ) == str(args.sample)
        ]

    if not rows and not args.maps_only:
        raise RuntimeError(
            "No SAE metric rows matched "
            "the requested sample."
        )

    print(
        f"Loaded {len(records)} sample records."
    )
    print(
        f"Flattened {len(rows)} "
        "block/timestep metric rows."
    )

    # ------------------------------------------------------------
    # Metric plots
    # ------------------------------------------------------------
    if not args.maps_only:
        print()
        print("=" * 70)
        print("BUILDING SAE METRIC VISUALIZATIONS")
        print("=" * 70)

        plot_sample_metric_heatmaps(
            rows,
            outdir,
        )

        plot_mean_block_time(
            rows,
            outdir,
        )

        plot_time_evolution(
            rows,
            outdir,
        )

        plot_block_comparison(
            rows,
            outdir,
        )

        plot_pairwise_heatmaps(
            records,
            outdir,
        )

        plot_cid_similarity(
            rows,
            outdir,
        )

        plot_explained_variance(
            sae_results_dir,
            outdir,
        )

        write_summary(
            rows,
            sae_results_dir,
            outdir,
        )

    # ------------------------------------------------------------
    # Concept maps
    # ------------------------------------------------------------
    if not args.metrics_only:
        print()
        print("=" * 70)
        print("BUILDING SAE CONCEPT-MAP VISUALIZATIONS")
        print("=" * 70)

        visualise_concept_maps(
            project_dir,
            records,
            sae_results_dir,
            outdir,
            sample_filter=args.sample,
            concept_filter=args.concept,
        )

    print()
    print("=" * 70)
    print("SAE VISUALIZATION COMPLETE")
    print("=" * 70)
    print(
        "Output:",
        outdir,
    )


if __name__ == "__main__":
    main()