import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic visual style
sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def parse_metrics_file(filepath):
    """Loads and parses the JSON metrics file into structured DataFrames."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find metrics file at: {filepath.resolve()}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)
        
    concept_records = []
    pair_records = []
    
    for sample in raw_json_data:
        exp = sample["experiment"]
        sample_id = exp["sample_id"]
        title = exp["title"]
        source = exp.get("source", "N/A")
        
        # 1. Parse individual concept metrics
        for concept, metrics in sample["concept_metrics"].items():
            concept_records.append({
                "sample_id": sample_id,
                "title": title,
                "source": source,
                "concept": concept,
                "centroid_x": metrics["centroid"][0],
                "centroid_y": metrics["centroid"][1],
                "entropy": metrics["attention_entropy"],
                "concentration_index": metrics["attention_concentration_index"]
            })
            
        # 2. Parse pairwise centroid deviations and cross-concept indices
        for pair_key, dev_info in sample["centroid_deviation"].items():
            concrete_concept, metaphor_concept = pair_key.split("__")
            cci = sample["cross_concept_index"].get(pair_key, np.nan)
            
            pair_records.append({
                "sample_id": sample_id,
                "title": title,
                "source": source,
                "pair": f"{concrete_concept} \u2192 {metaphor_concept}",
                "concrete": concrete_concept,
                "metaphor": metaphor_concept,
                "concrete_x": dev_info["concrete_centroid"][0],
                "concrete_y": dev_info["concrete_centroid"][1],
                "metaphor_x": dev_info["metaphorical_centroid"][0],
                "metaphor_y": dev_info["metaphorical_centroid"][1],
                "distance": dev_info["distance"],
                "cross_concept_index": cci
            })
            
    df_concepts = pd.DataFrame(concept_records)
    df_pairs = pd.DataFrame(pair_records)
    
    # Classify concepts as Concrete or Metaphorical based on pair roles
    concrete_set = set(df_pairs["concrete"])
    df_concepts["concept_type"] = df_concepts["concept"].apply(
        lambda c: "Concrete" if c in concrete_set else "Metaphorical"
    )
    
    return df_concepts, df_pairs


def generate_and_save_visualizations(df_concepts, df_pairs, output_dir="results/figures"):
    """Creates the 4-panel analysis dashboard and saves it to the target directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file = output_path / "flux_attention_metrics_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    palette = {"Concrete": "#1f77b4", "Metaphorical": "#d62728"}

    # -----------------------------------------------------------------
    # Panel 1: 2D Spatial Centroid Map with Concrete -> Metaphor Vectors
    # -----------------------------------------------------------------
    ax1 = axes[0, 0]
    for _, row in df_pairs.iterrows():
        ax1.annotate(
            "", 
            xy=(row["metaphor_x"], row["metaphor_y"]),
            xytext=(row["concrete_x"], row["concrete_y"]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.35, lw=1.2)
        )
    
    sns.scatterplot(
        data=df_concepts,
        x="centroid_x",
        y="centroid_y",
        hue="concept_type",
        palette=palette,
        s=80,
        edgecolor="black",
        linewidth=0.8,
        ax=ax1
    )
    ax1.axvline(0.5, color="black", linestyle="--", alpha=0.3)
    ax1.axhline(0.5, color="black", linestyle="--", alpha=0.3)
    ax1.set_title("A. Concept Centroids & Spatial Shift Vectors", fontweight="bold")
    ax1.set_xlabel("Centroid X (Normalized)")
    ax1.set_ylabel("Centroid Y (Normalized)")
    ax1.legend(title="Concept Type", loc="upper left")

    # -----------------------------------------------------------------
    # Panel 2: Cross-Concept Index vs. Centroid Distance
    # -----------------------------------------------------------------
    ax2 = axes[0, 1]
    sns.regplot(
        data=df_pairs,
        x="distance",
        y="cross_concept_index",
        scatter_kws={"alpha": 0.8, "color": "#2ca02c", "s": 60, "edgecolor": "k"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=ax2
    )
    corr = df_pairs["distance"].corr(df_pairs["cross_concept_index"])
    ax2.text(
        0.05, 0.1, f"Pearson r = {corr:.2f}",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax2.set_title("B. Cross-Concept Index vs. Centroid Distance", fontweight="bold")
    ax2.set_xlabel("Centroid Deviation (Euclidean Distance)")
    ax2.set_ylabel("Cross-Concept Index")

    # -----------------------------------------------------------------
    # Panel 3: Attention Concentration: Concrete vs. Metaphorical
    # -----------------------------------------------------------------
    ax3 = axes[1, 0]
    sns.boxplot(
        data=df_concepts,
        x="concept_type",
        y="concentration_index",
        palette=palette,
        width=0.4,
        ax=ax3,
        boxprops=dict(alpha=0.7)
    )
    sns.stripplot(
        data=df_concepts,
        x="concept_type",
        y="concentration_index",
        color="black",
        alpha=0.6,
        jitter=0.2,
        size=6,
        ax=ax3
    )
    ax3.set_title("C. Attention Concentration by Concept Type", fontweight="bold")
    ax3.set_xlabel("Concept Category")
    ax3.set_ylabel("Attention Concentration Index")

    # -----------------------------------------------------------------
    # Panel 4: Cross-Concept Alignment Across Experiment Samples
    # -----------------------------------------------------------------
    ax4 = axes[1, 1]
    pair_pivot = df_pairs.groupby(["title", "source"])["cross_concept_index"].mean().reset_index()
    pair_pivot = pair_pivot.sort_values(by="cross_concept_index", ascending=True)

    sns.barplot(
        data=pair_pivot,
        y="title",
        x="cross_concept_index",
        hue="source",
        palette="viridis",
        dodge=False,
        ax=ax4
    )
    ax4.set_xlim(0.6, 1.0)
    ax4.set_title("D. Mean Cross-Concept Alignment by Sample", fontweight="bold")
    ax4.set_xlabel("Mean Cross-Concept Index")
    ax4.set_ylabel("")
    ax4.legend(title="Benchmark Source", loc="lower right")

    plt.suptitle("FLUX Attention Analysis: Concrete vs. Metaphorical Grounding", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f" Visualization successfully generated and saved to:\n   --> {save_file.resolve()}")
    plt.close()


if __name__ == "__main__":
    # Relative path matching your project tree structure
    METRICS_FILE = "flux_results/metrics/all_metrics.json"
    OUTPUT_FOLDER = "results/figures"

    # Fallback to root or current directory if executing from another working dir
    if not os.path.exists(METRICS_FILE):
        if os.path.exists("all_metrics.json"):
            METRICS_FILE = "all_metrics.json"

    df_concepts, df_pairs = parse_metrics_file(METRICS_FILE)
    generate_and_save_visualizations(df_concepts, df_pairs, output_dir=OUTPUT_FOLDER)