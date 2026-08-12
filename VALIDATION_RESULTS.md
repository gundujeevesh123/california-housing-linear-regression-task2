# Task 2 — Second Verification Pass Results

**Stage:** Actual execution, reproducibility, statistical validation
**Auditor role:** Senior ML Execution Engineer · Reproducibility Engineer · Statistical Validator
**Date:** 2026-08-12

Every number in this document was produced by scikit-learn from the raw
California Housing dataset in the audit environment — nothing is copied,
remembered, or assumed. Every figure was written by matplotlib to disk from
model predictions in the same run.

---

## 1. Execution status

| Step | Result |
|---|---|
| Read `TASK2_AUDIT.md` before executing | Yes — audit passed, only cosmetic findings F-1 / F-2 remain (see §11). |
| Notebook clean-kernel execution #1 (`jupyter nbconvert --execute`) | Success — 15 code cells, monotonic `execution_count` 1 → 15, zero errors, 406 660 bytes written. |
| Notebook clean-kernel execution #2 | Success — deterministic numerical outputs bit-identical to run #1. |
| Extended validation runner (`src/run_validation.py`) run #1 | Success — 8 artifacts written into `results/` and `figures/`. |
| Extended validation runner (`src/run_validation.py`) run #2 | Success — every one of those 8 artifacts is byte-identical to run #1 (SHA-256 match). |
| Execution errors encountered | None. |

## 2. Dataset actually used (runtime values)

| Property | Runtime value |
|---|---|
| Source | `sklearn.datasets.fetch_california_housing(as_frame=True)` (Task 1 CSV scan attempted first, returned no match) |
| Shape | `(20 640, 9)` — 8 features + `HousePrice` |
| Columns | `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, HousePrice` |
| Dtypes | all `float64` |
| Missing values | `0` |
| Duplicate rows | `0` |
| Target (`HousePrice`) | min `0.14999`, max `5.00001`, mean `2.0686`, median `1.7970`, std `1.1540` |
| Target units | Units of $100,000 (correctly documented in the report) |

## 3. Train / test split

| Property | Value |
|---|---|
| `train_test_split` args | `test_size=0.2, random_state=42` |
| Train rows | `16 512` |
| Test rows | `4 128` |

## 4. Preprocessing verification

`StandardScaler` fit on **training data only**, then applied to both splits. Runtime verification of standardisation on `X_train_scaled`:

| Statistic | Value |
|---|---|
| Per-feature mean | `[-0. -0. -0. -0. -0. 0. 0. 0.]` |
| Per-feature std | `[1. 1. 1. 1. 1. 1. 1. 1.]` |
| All mean magnitudes < 1 × 10⁻¹⁰? | `True` |
| All std magnitudes within 1 × 10⁻¹⁰ of 1? | `True` |

No leakage: `X_test` uses `scaler.transform` (not `fit_transform`). No `scaler.fit(X)` or `scaler.fit_transform(X)` on the full frame anywhere in the source.

## 5. Model metrics — actual runtime results

Per-model **train and test** metrics, MAE, and the train-test R² gap. All values from a single executed run and written to `results/model_comparison.csv`.

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Test MAE | R² gap | Rank |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Linear Regression | 0.7197 | 0.7456 | 0.6126 | 0.5758 | 0.5332 | +0.0368 | 3 |
| Ridge Regression | 0.7197 | 0.7456 | 0.6126 | 0.5758 | 0.5332 | +0.0367 | 2 |
| **Decision Tree (max_depth=5)** | **0.6959** | **0.7242** | **0.6377** | **0.5997** | **0.5223** | **+0.0379** | **1** |

Full-precision values are in `results/model_comparison.csv` (six decimals). Ranks use the assignment's selection rule: primary = min test RMSE, tiebreak = max test R². All three R² gaps are small (≤ 0.038); no overfitting signal beyond what a depth-5 tree naturally shows.

## 6. Programmatically selected winner

Selected by `min(rows, key=lambda r: (r["Test RMSE"], -r["Test R2"]))` — no hard-coded name.

- **Winner:** `Decision Tree`
- **Test RMSE:** `0.7242`
- **Test R²:** `0.5997`
- **Test MAE:** `0.5223`

The same winner is announced in the executed notebook's stdout: `→ Selected model: Decision Tree`, and appears in every artifact (comparison CSV, Excel workbook, saved joblib, report PDF, all figures).

## 7. Cross-validation (supporting evidence, not a replacement)

5-fold CV on the **training** data only, refitting the scaler inside a `Pipeline` per fold to avoid leakage. `KFold(n_splits=5, shuffle=True, random_state=42)`.

| Model | CV RMSE mean | CV RMSE std | CV R² mean | CV R² std |
|---|---:|---:|---:|---:|
| Linear Regression | 0.7205 | 0.0155 | 0.6115 | 0.0138 |
| Ridge Regression | 0.7205 | 0.0155 | 0.6115 | 0.0138 |
| Decision Tree (max_depth=5) | 0.7227 | 0.0115 | 0.6092 | 0.0041 |

**Interpretation.** On the fixed hold-out split, Decision Tree beats the linear models by ≈0.02 RMSE. Under 5-fold CV, the three models' CV-mean RMSEs are within ±1σ of each other — the tree's held-out advantage is real for this split but not overwhelming across CV folds. Because the assignment specifies a single 80/20 hold-out as the evaluation protocol, the winner is decided by test-set RMSE (Decision Tree). CV is reported here only as supporting evidence and does not override that rule.

## 8. Feature importance (Decision Tree)

Extracted from `dt.feature_importances_` after the same fit used for the reported metrics. Written to `results/feature_importance.csv`.

| Feature | Importance |
|---|---:|
| MedInc | 0.7712 |
| AveOccup | 0.1284 |
| HouseAge | 0.0416 |
| AveRooms | 0.0313 |
| Latitude | 0.0220 |
| Population | 0.0025 |
| Longitude | 0.0021 |
| AveBedrms | 0.0009 |

Consistent with domain intuition: median block income (`MedInc`) is the dominant driver, followed by average household size (`AveOccup`). Location features (`Latitude`, `Longitude`) collectively contribute ~2%.

## 9. Predictions.csv (auditable numerical foundation)

Written to `results/predictions.csv` — long-format, 12 384 rows (4 128 test rows × 3 models):

| Column | Description |
|---|---|
| `model` | `Linear Regression` / `Ridge Regression` / `Decision Tree` |
| `actual` | ground-truth `y_test[i]` value |
| `predicted` | model's test prediction on scaled features |
| `residual` | `actual − predicted` |
| `absolute_error` | `|actual − predicted|` |

Every point in every regenerated figure can be traced back to a specific row of this file.

## 10. Regenerated figures (from code, never edited)

| Path | Source of pixels |
|---|---|
| `figures/actual_vs_predicted_best_model.png` | matplotlib scatter of `y_test` vs `predictions[best_name]`, red `y = x` line drawn from `[lo, hi]` to `[lo, hi]`, title auto-generated from RMSE/R² of the winner |
| `figures/residuals_best_model.png` | matplotlib residual scatter + histogram from `y_test − predictions[best_name]` |
| `outputs/figures/actual_vs_predicted_decision_tree.png` | notebook cell 12 (same underlying data) |
| `outputs/figures/residuals_decision_tree.png` | notebook cell 13 |
| `outputs/figures/rmse_r2_comparison.png` | notebook cell 14 |
| `outputs/excel_charts/Chart*.png` | LibreOffice rendering of native Excel charts bound to `Task2_Predictions.xlsx` cell ranges |

No pixel-level edits. If a plot needed changing, only the Python code was modified and the figure regenerated.

## 11. Reproducibility check

**Deterministic outputs across two independent clean-kernel runs of the notebook:**

| Artifact | SHA-256 match run #1 vs run #2 |
|---|---|
| `outputs/metrics.json` | Identical |
| `outputs/model_comparison.csv` | Identical |
| `models/best_model.joblib` | Identical |
| `outputs/figures/actual_vs_predicted_decision_tree.png` | Identical |
| `outputs/figures/residuals_decision_tree.png` | Identical |
| `outputs/figures/rmse_r2_comparison.png` | Identical |
| Notebook file itself | Differs (notebook metadata has timestamps); substantive cell outputs identical. |

**Deterministic outputs across two runs of `src/run_validation.py`:**

| Artifact | SHA-256 match run #1 vs run #2 |
|---|---|
| `results/model_comparison.csv` | Identical |
| `results/predictions.csv` | Identical |
| `results/cv_scores.csv` | Identical |
| `results/feature_importance.csv` | Identical |
| `results/summary.json` | Identical |
| `results/environment.json` | Identical |
| `figures/actual_vs_predicted_best_model.png` | Identical |
| `figures/residuals_best_model.png` | Identical |

Perfect reproducibility for every deterministic artifact.

## 12. Environment (from `results/environment.json`)

| Component | Version / value |
|---|---|
| Python | 3.10.12 |
| Platform | Linux (audit sandbox) |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.7 |
| joblib | 1.5.2 |
| `random_state` | 42 (fixed everywhere) |

## 13. Errors found and corrected

None. Executed cleanly on both runs; the audit-flagged items (F-1: missing PLANNING/VALIDATION_NOTES from earlier stages; F-2: Task 1 lookup path in the current sandbox mount) were classified in the audit as documentation/environment cosmetics — this stage does not introduce any code changes. The Task 1 CSV scan continues to return `None` in this sandbox and the sklearn fallback provides the correct dataset either way.

## 14. Files created or updated by this validation pass

New files:

```
results/
├── model_comparison.csv    (with Train RMSE, Test RMSE, Train R², Test R², Test MAE, R² gap, Rank)
├── predictions.csv         (long format, 12,384 rows — auditable foundation for every figure)
├── cv_scores.csv           (5-fold CV mean/std of RMSE and R² per model)
├── feature_importance.csv  (Decision Tree feature importances)
├── environment.json        (Python / library versions + run configuration)
└── summary.json            (single-file snapshot of everything in this document)

figures/
├── actual_vs_predicted_best_model.png   (winner, spec-mandated filename)
└── residuals_best_model.png             (companion residual diagnostic)

src/
└── run_validation.py       (extended validation runner — reproducible entry point)
```

Existing notebook, `outputs/*`, `reports/*`, `models/*`, and the Excel workbook were re-executed but the deterministic parts are byte-identical to the prior state.

## 15. Sign-off

- Notebook executes cleanly from a clean kernel on two independent runs: ✓
- Every deterministic artifact reproduces bit-identically across runs: ✓
- All reported metrics are runtime outputs of scikit-learn on the actual dataset — no hand-typed values anywhere: ✓
- Winner (Decision Tree) selected by the assignment's programmatic rule; consistent across all artifacts: ✓
- Actual vs Predicted plot regenerated directly from `y_test` and `y_pred_best`; red y = x reference line present: ✓
- Cross-validation, feature importance, MAE, and train-vs-test comparison added as supporting evidence without altering the required protocol: ✓
- Ready for the finalization stage (polished report and consolidated release).

*End of second verification pass. No further changes should be made until the finalization stage produces the polished report.*
