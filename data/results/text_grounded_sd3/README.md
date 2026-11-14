# Text-grounded SAE-feature selection vs. the argmax rule (SD3, human-rated set)

Reviewer question: why should the dominant SAE feature (argmax of mean activation) correspond to
prompt concepts? Compare text-grounded alternatives against the argmax rule on the same human set.

* Set: the 171 SD3 prompts with human ratings (same prompts as the MIRAGE workbook), SD3-medium,
  CPU generator seed 42, 28 steps, CFG 7.0, blocks 4/8/12/16/20 x timesteps 0.5/0.75/1.0.
  The argmax rule reproduces the stored `sd3_raw.csv` latents / centroids / concentrations exactly
  (checked on the smoke test: 30/30 cells, max abs diff 0).
* Rules (src/text_grounded_sd3.py): `argmax_mean` (current), `attn_corr` and `text_token`
  (text-grounded), `attn_map` (concept-token attention itself, no SAE), `random_active` and
  `fixed_generic` (nulls; the latter is one fixed latent per cell, independent of the prompt).
* Concept tokens: the prompt's top-8 salient object nouns (src/build_concept_spans.py).
* Scoring (src/analyze_text_grounded.py): the unchanged SA-7 attention-geometry composite per rule,
  Pearson r with the human rating, 95% bootstrap CIs (5000 prompt-level resamples), paired
  bootstrap differences vs argmax.

Files: `tg_metrics.csv` (per prompt x cell x rule rows), `summary_by_rule.csv`,
`per_feature_correlations.csv`, `argmax_latent_diversity_all_models.csv` (how many distinct latents
argmax picks per cell, all three models, 599 prompts).

Caveats: SD3 only (PixArt and FLUX not run); n = 171 and wide CIs; SA-7's seven features were
developed on argmax maps from this same data, so they favour argmax-like maps.

Inputs not included in this repository (so the analysis step cannot be re-run from the repo alone):
`replicate_mirage_features.py` and the raw per-cell CSVs it reads (`sd3_raw.csv` etc.; point
`MIRAGE_FEATURE_DIR` at them), the workbook with the CLIP / BLIP-2 / VQAScore columns, and the
per-prompt human-rating table (`--human-set`). `tg_metrics.csv` here is the full output of the
GPU run, so the per-rule results can be recomputed from it given those inputs.
