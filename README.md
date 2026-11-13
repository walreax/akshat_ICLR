# Mechanistic-Interpretability Image Quality Metrics for Diffusion Models

Internal-mechanism-based image-quality metrics for text-to-image diffusion
models (SD3, PixArt-alpha, FLUX), built from cross-attention centroid
geometry and SAE-based concept activations, validated against public
benchmarks (CLIPScore, BLIP-2, VQAScore) and real human ratings.

## Structure

| Path | Contents |
| --- | --- |
| `src/` | Core pipeline: per-model activation extractors (`sd_activation_extractor.py`, `pixart_activation_extractor.py`, `flux_activation_extractor.py`), SAE training (`train_sae.py`, `generate_sae.py`), run/eval wrappers (`run_extract*.py`, `run_sd3.py`, `run_pixart.py`), and analysis/visualization scripts. |
| `tests/` | Validation and smoke-test scripts (manual, GPU-dependent — not a CI suite). See `tests/README.md` for what each one checks and how to run it. |
| `data/` | The canonical 600-prompt evaluation set (`prompts_600.csv`), the calibration/test splits, concept definitions, historical training-set snapshots (`training_sets/`), and analysis-ready results (`results/`) — MIRAGE validation data, the baseline comparison, the concentration ablation, SAE top-concepts, and the anonymized human ratings (`results/rics.xlsx`). See `results/README.md`. |
| `human_eval_streamlit/` | The Streamlit app used to collect human ratings (prompt/image pairs, 4-axis scoring). |
| `human_eval_app/` | An unused Next.js prototype for the same task (correctly reads the canonical prompt set, but its rating storage is local-file-only and not yet backed by a real database). |
| `imageMetric/` | Public-benchmark scoring (CLIPScore, BLIP-2, VQAScore, Inception Score) and image generation scripts, self-contained. |
| `writeups/` | Pipeline/failure-case diagrams (`.drawio`), the metric spec, and paper drafts. |
| `sae_interpretation/` | Per-concept SAE activation visualizations by seed/timestep. |
| `flux_sae_training/`, `flux_sae_validation/` | FLUX-specific SAE training logs and validation output. |

## The metric

**MIRAGE** scores text-to-image alignment from cross-attention centroid
geometry alone — no reference image, no external model — combining how
far the attention centroid moves across transformer depth (`drift`) and
how erratically it moves across denoising time (`ts_path`):

```
raw = w0 + w_drift * drift + w_tspath * ts_path
MIRAGE = sigmoid((raw - midpoint) / temp)
```

On a 35-pair human-rated validation set it correlates far better with
human judgment (r = 0.495, LOOCV r = 0.377) than CLIPScore, BLIP-2, or
VQAScore (r = 0.050, -0.158, -0.039) on the same images. See
`writeups/metric_spec.html` for the metric definition and
`data/results/` for the full validation data, baseline comparison, and
human ratings (`mirage_validation_n35.csv`, `baseline_comparison.csv`,
`rics.xlsx`) — see `data/results/README.md` for a guided tour.

## Known limitations (tracked, not hidden)

- The shared CSV loader (`load_concept_dataset` in
  `src/sd_activation_extractor.py`) has a title-detection heuristic that
  merges one continuation row on the 600-prompt set (a lowercase poem
  title gets read as a continuation of the previous row), capping
  real coverage at 599/600 for SD3, PixArt, and FLUX-schnell.
- MIRAGE's weights are fit on the same 35 rows they're validated
  against; leave-one-out CV gives r &asymp; 0.38, the honest
  generalization estimate.
- Attention concentration (a separate entropy-based measure of how
  focused the attention map is) does **not** improve MIRAGE when added
  as a third term — tested seven ways, all negative or overfit. See
  `data/results/concentration_ablation_summary.md` for the full ablation.
