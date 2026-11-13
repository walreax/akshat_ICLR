# Results

Everything in this folder is a derived, analysis-ready output of the
pipelines in [`src/`](../../src). Nothing here identifies an annotator,
author, or institution -- see [Anonymity](#anonymity) below.

## MIRAGE metric

```
raw    = w0 + w_drift * drift + w_tspath * ts_path
MIRAGE = sigmoid((raw - midpoint) / temp)
```
`w0 = 0.1485, w_drift = 0.2923, w_tspath = -0.1028, midpoint = 0.35, temp = 0.20`

- **`mirage_validation_n35.csv`** -- the 35-item human-validated set MIRAGE's
  weights are fit and evaluated on. One row per (prompt, model): the two
  geometry features (`drift`, `ts_path`), the SAE attention-concentration
  feature tested and rejected in the ablation (see below), the resulting
  `MIRAGE_score`, three baseline scores (`CLIPScore`, `BLIP2_score`,
  `VQAScore`), and the ground-truth `human_overall` rating.
- **`baseline_comparison.csv`** -- Pearson r of each metric against
  `human_overall` on the same 35 items, plus MIRAGE's leave-one-out CV r
  (the honest generalization estimate, since the weights are fit on these
  same 35 rows). MIRAGE (r=0.495, LOOCV r=0.377) beats CLIPScore (r=0.050),
  BLIP2 (r=-0.158), and VQAScore (r=-0.039) by a wide margin -- all three
  baselines are essentially uncorrelated with human judgment on this set.
- **`concentration_ablation_summary.md`** -- full writeup of the 7 attempts
  to add SAE attention-concentration as a third MIRAGE feature. All 7
  failed (best case: overfits in-sample, reverses under LOOCV), which is
  why MIRAGE ships as the 2-feature model above.

## Human evaluation

- **`rics.xlsx`** -- anonymized crowd ratings from the human-eval app
  (`human_eval_streamlit/human_eval.py`). "Summary" sheet has per-model
  means; "Raw Data" has every individual rating. Current snapshot: **555
  valid ratings** from 5 anonymized annotators (`R1`-`R5`), across all
  three models (SD3, PixArt, FLUX) and both content types (story, poem),
  scored on 4 axes (Semantic Fidelity, Attribute Presence, Compositional
  Correctness, Narrative Fidelity) plus an overall score. 187 rows were
  excluded before this snapshot: 71 collected before a prompt/image
  pairing bug was fixed (invalid -- rated against the wrong prompt text)
  and 116 from one confirmed bot session (rate-limited afterward; see the
  app's rate-limit fix in `human_eval_streamlit/human_eval.py`).

## SAE concept analysis

- **`top_concepts_full.csv`** / **`top_concepts_top32.csv`** -- for each
  (model, transformer block, denoising timestep), every SAE latent that
  was ever selected as the dominant concept for some (concept, seed) pair
  in the benchmark run, ranked by selection frequency (`top32` is the
  same table filtered to the 32 most-recurring latents per group; `full`
  keeps every latent so nothing is silently dropped). **Coverage is
  currently partial** -- SD3 (186 rows) is from the completed benchmark
  run, PixArt (17 rows) reflects that model's SAE evaluation still being
  mid-run at the time this was generated. Regenerate with
  `src/top_concepts.py` once both runs finish.

## Reproducing / regenerating

All CSVs here are derived from raw pipeline output that is not checked in
(multi-GB attention maps, activations, and per-seed images). To
regenerate:
- `mirage_validation_n35.csv` / `baseline_comparison.csv` -- rerun the
  centroid-geometry extraction (`src/sd_activation_extractor.py` /
  `src/pixart_activation_extractor.py` with `--load-sae`) against the
  validation subset, then fit MIRAGE's weights against `rics.xlsx`.
- `top_concepts_*.csv` -- `src/top_concepts.py` against each model's
  `concept_metrics.csv`.
- `rics.xlsx` -- pull `human_evaluation_results.csv` from the eval
  server and rebuild with the exclusion rules documented in the sheet.

## Anonymity

- No real names, emails, usernames, or institution names appear in any
  file in this folder. Annotator IDs are sequential (`R1`, `R2`, ...),
  remapped from session identifiers that never contained personal
  information to begin with (either a numeric test ID or an
  auto-generated `guest_<random>` string).
- `rics.xlsx`'s document metadata (`docProps/core.xml`) has `creator` and
  `lastModifiedBy` blanked.
- Literary work titles/authors (e.g. "To God by Ivor Gurney") are public
  domain source text used as generation prompts, not annotator or author
  identities.
