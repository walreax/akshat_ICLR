# Validation & smoke-test scripts

These are **manual, GPU-dependent validation scripts**, not an automated
CI test suite — they need a live GPU, real model weights, and (mostly)
real extraction output already sitting on disk, so they aren't runnable
in a CI container. What each one checks:

| Script | Checks |
| --- | --- |
| `smoke_test_ablation.py` | End-to-end causal check that `SAEFeaturePatcher` actually changes generation: ablate/amplify one real SAE latent and compare the output image against baseline, overall and inside a mask centered on the latent's own centroid. One row, no statistics — "does the mechanism work at all" before running a full ablation study. |
| `validate_sae.py` | Loads a trained SAE checkpoint against FLUX and confirms it reconstructs real activations at the expected fidelity. |
| `check_datasets.py` | Loads a concept dataset through the shared CSV loader and reports how many rows were parsed vs. how many exist in the source file — this is what originally caught the CSV-parser bug that silently merges a continuation row (see `src/sd_activation_extractor.py`'s `load_concept_dataset`). |
| `check_latents.py` | Sanity-checks saved `.npz` latent dumps (shape, key coverage) from a `--save-latents` extraction run. |
| `check_progress.py` / `check_progress_evalset.py` | Real progress on an extraction run, via `pandas` grouping on `(concept_id, seed)` pairs — never `wc -l` / naive line counts, since prompt text can contain embedded newlines inside valid CSV quoting. |
| `diag.sh` / `diag2.py` | Independent duplicate-row checks on `concept_metrics.csv` (bash and pandas versions of the same check) — this is what caught the PixArt silent-failure bug (a callback API mismatch that made every extracted row get silently dropped for hours while the GPU stayed busy generating real images). |

## Running them

`smoke_test_ablation.py` and `check_datasets.py` import sibling modules
that now live in `../src/` (`sd_activation_extractor.py`,
`pixart_activation_extractor.py`). Run them with `src/` on the path:

```bash
PYTHONPATH=../src python smoke_test_ablation.py
PYTHONPATH=../src python check_datasets.py
```

The rest have no cross-module imports and can be run directly, from
whatever directory the corresponding extraction output actually lives in
(the paths inside each script point at the specific run directory it
checks — edit the constant at the top rather than passing a CLI flag,
matching this project's other `run_*.py` scripts).
