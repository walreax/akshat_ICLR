"""
analyze_text_grounded.py
========================
Score every SAE-feature selection rule from text_grounded_sd3.py on the same human-rated set,
using the UNCHANGED feature code (scratch_pull_599/isoloria/replicate_mirage_features.py) and
the unchanged SA-7 composite (the attention-geometry half of MIRAGE v4):

    SA-7 = mean of within-model z-scores of
           centroid_drift, centroid_ts_path, conc_min, -conc_range,
           -dev_final, -centroid_depth_path, -conc_depth_path

Primary metric (fixed before looking at any result): Pearson r between the per-prompt SA-7 and the
human rating, with a 95% percentile bootstrap CI (prompt-level resampling). Secondary: Spearman,
and PAIRED bootstrap differences against the argmax rule (same prompts, same resamples).
Also reports how text-aligned each rule's chosen latent is (corr_attn) and how many distinct
latents it uses, plus the CLIP / BLIP-2 / VQAScore correlations on the same prompts.

Usage:
    python src/analyze_text_grounded.py --csv <tg_metrics.csv> --model SD3 --out <summary.csv>
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

warnings.filterwarnings("ignore")
ISO = os.environ.get("MIRAGE_FEATURE_DIR", "scratch_pull_599/isoloria")  # dir with replicate_mirage_features.py + raw CSVs (not in this repo)
sys.path.insert(0, ISO)
import replicate_mirage_features as R  # noqa: E402

KEYS = ["centroid_drift", "centroid_ts_path", "conc_min", "conc_range", "dev_final",
        "centroid_depth_path", "conc_depth_path"]
SIGN = [1, 1, 1, -1, -1, -1, -1]
REF = 31.5
MODEL_NAME = {"SD3": "SD3", "PixArt": "PixArt", "FLUX": "FLUX-schnell"}


def features_for(raw, model, titles):
    is_flux = model == "FLUX"
    blocks = R.FLUX_BLOCKS if is_flux else R.NONFLUX_BLOCKS
    ts = R.FLUX_TS if is_flux else R.NONFLUX_TS
    out = {}
    for t in titles:
        f = R.compute_features(raw, MODEL_NAME[model], t, is_flux)
        if f is None:
            continue
        cell, _ = R.per_cell_table(raw, MODEL_NAME[model], blocks, ts, t)
        cx, cy = cell[(blocks[-1], ts[-1])][:2]
        f["dev_final"] = float(np.sqrt((cx - REF) ** 2 + (cy - REF) ** 2))
        out[t] = f
    return pd.DataFrame(out).T


def sa7(feat):
    z = pd.DataFrame({k: sg * (feat[k] - feat[k].mean()) / feat[k].std(ddof=1) for k, sg in zip(KEYS, SIGN)})
    return z.mean(axis=1), z


def boot_ci(x, y, idx_list, fn):
    vals = np.array([fn(x[i], y[i]) for i in idx_list])
    return np.nanpercentile(vals, [2.5, 97.5])


def pr(a, b):
    return stats.pearsonr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="SD3")
    ap.add_argument("--human-set", required=True, help="CSV with columns pid, model, human (per prompt x model human rating)")
    ap.add_argument("--workbook", required=True, help="workbook with sheet '2. MIRAGE Setup' (CLIP/BLIP-2/VQAScore columns)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    tg = pd.read_csv(args.csv)
    tg["model"] = MODEL_NAME[args.model]
    p600 = pd.read_csv("data/prompts_600.csv")
    title2id = p600.set_index("name_of_work")["id"].to_dict()

    hs = pd.read_csv(args.human_set)
    hs = hs[hs.model == args.model].copy()
    hs["human"] = pd.to_numeric(hs["human"], errors="coerce")
    human = hs.set_index("pid")["human"]

    # workbook baselines on the same prompts
    wb = openpyxl.load_workbook(args.workbook, data_only=True, read_only=True)
    ws = wb["2. MIRAGE Setup"]
    base = {}
    for r in ws.iter_rows(min_row=12, max_row=543, min_col=1, max_col=7, values_only=True):
        if r[0] is not None and r[1] == args.model:
            base[r[0]] = dict(CLIP=r[4], BLIP2=r[5], VQAScore=r[6])
    base = pd.DataFrame(base).T.apply(pd.to_numeric, errors="coerce")

    rng = np.random.default_rng(0)
    rows = []
    feats_by_rule = {}
    for rule, g in tg.groupby("rule"):
        raw = g.rename(columns={})[["model", "concept_id", "block", "timestep_fraction",
                                    "centroid_x", "centroid_y", "concentration"]]
        titles = sorted(raw["concept_id"].unique())
        feat = features_for(raw, args.model, titles)
        feat.index = [title2id[t] for t in feat.index]
        feats_by_rule[rule] = feat

    common = sorted(set.intersection(*[set(f.index) for f in feats_by_rule.values()]) & set(human.dropna().index))
    print(f"prompts with all rules + human rating: {len(common)}")
    n = len(common)
    idx_list = [rng.integers(0, n, n) for _ in range(args.n_boot)]
    h = human.loc[common].values.astype(float)

    scores = {}
    for rule, feat in feats_by_rule.items():
        f = feat.loc[common]
        s, z = sa7(f)
        scores[rule] = s.values
    for name in ["CLIP", "BLIP2", "VQAScore"]:
        scores[name] = base.loc[common, name].values.astype(float)

    base_rule = "argmax_mean"
    for name, s in scores.items():
        r_p = pr(s, h)
        ci = boot_ci(s, h, idx_list, pr)
        r_s = stats.spearmanr(s, h)[0]
        row = dict(rule=name, n=n, pearson=r_p, ci_lo=ci[0], ci_hi=ci[1], spearman=r_s)
        if name in feats_by_rule and name != base_rule:
            d = np.array([pr(s[i], h[i]) - pr(scores[base_rule][i], h[i]) for i in idx_list])
            row.update(d_vs_argmax=r_p - pr(scores[base_rule], h), d_lo=np.nanpercentile(d, 2.5),
                       d_hi=np.nanpercentile(d, 97.5))
        if name in feats_by_rule:
            g = tg[tg.rule == name]
            row.update(mean_corr_attn=g["corr_attn"].mean(),
                       distinct_latents_per_cell=g[g.latent_idx >= 0].groupby(["block", "timestep_fraction"])["latent_idx"].nunique().mean()
                       if (g.latent_idx >= 0).any() else np.nan)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))

    # per-feature correlations (transparency: is any single feature carrying the composite?)
    print("\nper-feature Pearson r with human (signed so higher = better):")
    tab = {}
    for rule, feat in feats_by_rule.items():
        _, z = sa7(feat.loc[common])
        tab[rule] = {k: pr(z[k].values, h) for k in KEYS}
    print(pd.DataFrame(tab).round(3).to_string())
    if args.out:
        res.to_csv(args.out, index=False)
        pd.DataFrame(tab).to_csv(args.out.replace(".csv", "_features.csv"))
        print("saved", args.out)


if __name__ == "__main__":
    main()
