# Does attention-concentration help MIRAGE? No.

MIRAGE's linear core uses two cross-attention-geometry features, `drift`
(centroid deviation from the un-attended baseline) and `ts_path`
(cumulative centroid path length across denoising timesteps):

```
raw    = w0 + w_drift * drift + w_tspath * ts_path
MIRAGE = sigmoid((raw - midpoint) / temp)
```

We tested seven ways of adding a third feature, SAE attention
*concentration* (`A`, mean peak-activation concentration of the argmax
concept per block/timestep), to see whether it improves the fit against
the n=35 human-rated validation set. All seven were negative or neutral;
none is used in the reported MIRAGE formula.

| # | Attempt | In-sample r | LOOCV r | Verdict |
|---|---------|:-:|:-:|---|
| 1 | Concentration alone | 0.036 | -0.676 | Far worse than either drift or ts_path alone |
| 2 | `M = α(1-D̄) + βĀ` (deviation reward + concentration, no intercept) | 0.196 (D̄ only) / 0.071 (Ā only) as single terms | -- | Both component correlations weak |
| 3 | Same, with intercept | -- | -- | Intercept doesn't rescue it |
| 4 | Mean-centering concentration before combining | (unchanged) | (unchanged) | Pearson r is shift-invariant -- mean-centering cannot change a Pearson correlation; verified analytically, not just empirically |
| 5 | 3-term regression: `drift + ts_path + concentration` | 0.498 (vs. 0.495 for 2-term) | **0.341** (vs. **0.377** for 2-term) | In-sample gain is illusory -- LOOCV drops, classic overfitting from the extra free parameter on n=35 |
| 6 | Raw per-block / per-timestep concentration disaggregation (not the aggregate mean) | best single slice: 0.226 (`conc_min`) | -- | Still weaker than the 2-term baseline |
| 7 | 5-feature model combining several disaggregated concentration slices | -- | **-0.051** | Worse than predicting the mean every time |

**Baseline for comparison** -- the reported 2-feature model (`drift + ts_path`,
no concentration):

| | In-sample r | LOOCV r |
|---|:-:|:-:|
| MIRAGE (drift + ts_path) | 0.495 | 0.377 |

**Takeaway**: attention concentration, in every form we tried (raw,
centered, disaggregated by block/timestep, combined linearly or as a
third regression term), does not add real signal on top of the two
centroid-geometry features. The apparent in-sample improvement from
adding it (attempt 5) is overfitting -- it reverses under leave-one-out
cross-validation, which is why MIRAGE ships as the 2-term model.

See `mirage_validation_n35.csv` for the underlying per-item `drift`,
`ts_path`, and `attention_concentration` values these numbers were
computed from.
