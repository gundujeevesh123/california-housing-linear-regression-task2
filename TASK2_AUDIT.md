# Task 2 — First Verification Pass Audit

**Audit date:** 2026-08-12
**Auditor role:** Senior ML Code Auditor · Dataset Verification Engineer · Assignment Compliance Reviewer
**Authoritative source of requirements:** `Task_2/🤖 Artificial Intelligence & Machine Learning– Task 2.pdf`

**Overall verdict:** The project as-executed satisfies every mandatory requirement of the assignment PDF and every numerical claim in every artifact matches an independent recomputation to floating-point precision. Five minor findings (Sections 9–10) do not affect assignment compliance but should be noted before finalization.

---

## 1. Assignment Requirements Matrix

| # | Requirement | M / O | Current implementation | Correct? | Required action |
|---|---|---|---|---|---|
| 1 | California Housing Dataset | M | `fetch_california_housing(as_frame=True)` inside `notebooks/AI_ML_Task2_Model_Comparison.ipynb` cell 2 (with a schema-validated Task 1 CSV scan attempted first) | Yes | None |
| 2 | Target = Median House Value, exposed as `HousePrice` | M | Renamed via `data.target.rename("HousePrice")` in cell 2 | Yes | None |
| 3 | Feature preparation (X / y separation) | M | Cell 4 `X = df.drop(columns=[TARGET_COLUMN]); y = df[TARGET_COLUMN]` with an assertion that `X.shape[1] == 8` and that the target has not leaked into X | Yes | None |
| 4 | Feature scaling with StandardScaler | M | Cell 6 `StandardScaler()` fit on `X_train`, transform applied to both splits | Yes | None |
| 5 | Train/test split (`test_size=0.2`, `random_state=42`) | M | Cell 5 uses exactly those arguments; observed split = 16 512 / 4 128 rows | Yes | None |
| 6 | `LinearRegression()` | M | Cell 8 dict entry `"Linear Regression": LinearRegression()` | Yes | None |
| 7 | `Ridge(alpha=1.0)` | M | Cell 8 `"Ridge Regression": Ridge(alpha=1.0)` | Yes | None |
| 8 | `DecisionTreeRegressor(max_depth=5)` | M | Cell 8 `DecisionTreeRegressor(max_depth=5, random_state=42)` — `random_state` added for reproducibility, `max_depth=5` unchanged | Yes | None |
| 9 | RMSE and R² on test set | M | Cell 9 iterates over `models`, fits on `X_train_scaled`, predicts on `X_test_scaled`, computes RMSE + R² via version-safe `root_mean_squared_error` (`squared=False` fallback) and `r2_score` | Yes | None |
| 10 | Model performance comparison table | M | Cell 9 produces `results_df`; cell 10 writes `outputs/model_comparison.csv` and `outputs/metrics.json` with an added `Rank` column | Yes | None |
| 11 | Best-model selection | M | Cell 11 `ranked = results_df.sort_values(by=["RMSE", "R2 Score"], ascending=[True, False])`; `best_name = ranked.index[0]` — no hard-coded winner in the source | Yes | None |
| 12 | Actual vs Predicted plot with red y=x line | M | Cell 12 uses `y_pred_best = predictions[best_name]`, scatter of `y_test` vs `y_pred_best`, red diagonal via `ax.plot(..., color="red", label="Perfect prediction (y = x)")` | Yes | None |
| 13 | Jupyter notebook (`AI_ML_Task2_Model_Comparison.ipynb`) | M | Present at exact required filename; 31 cells (16 markdown, 15 code); all executed clean top-to-bottom | Yes | None |
| 14 | 1–2 page PDF report | M | `reports/Task2_Report.pdf` — 2 pages, includes Introduction, Methodology, Feature Scaling, Ridge, Models, Results, Visualisation, Conclusion, Limitations | Yes | None |
| 15 | Optional: save best model via joblib | O | `models/best_model.joblib` — `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor)])` | Yes | None |
| 16 | Optional: extra charts | O | Residual plot + RMSE/R² bar chart + Excel workbook with 3 native charts + 3 chart PNGs + rendered PDF | Yes | None |

Mandatory items: 14 / 14 satisfied. Optional items: 2 / 2 satisfied.

---

## 2. Workspace Inventory (currently present)

```
Task_2/
├── HOW_TO_VERIFY.md                                 8 KB
├── README.md                                        4 KB
├── 🤖 Artificial Intelligence & Machine Learning– Task 2.pdf   372 KB   ← assignment
├── notebooks/
│   └── AI_ML_Task2_Model_Comparison.ipynb           407 KB   ← Deliverable 1
├── models/
│   └── best_model.joblib                              7 KB   ← optional
├── outputs/
│   ├── metrics.json                                  494 B   ← full-precision truth
│   ├── model_comparison.csv                          126 B   ← Deliverable 2
│   ├── Task2_Predictions.xlsx                       424 KB   ← Excel + 3 charts
│   ├── figures/
│   │   ├── actual_vs_predicted_decision_tree.png    191 KB   (matplotlib, notebook)
│   │   ├── residuals_decision_tree.png              141 KB   (matplotlib, notebook)
│   │   └── rmse_r2_comparison.png                    56 KB   (matplotlib, notebook)
│   └── excel_charts/
│       ├── Chart1_ActualVsPredicted_DecisionTree.png 90 KB   (Excel-rendered)
│       ├── Chart2_Residuals_DecisionTree.png         83 KB   (Excel-rendered)
│       ├── Chart3_RMSE_R2_Comparison.png             76 KB   (Excel-rendered)
│       └── Task2_Predictions_rendered.pdf           336 KB   (4-page charts PDF)
├── reports/
│   ├── Task2_Report.pdf                             211 KB   ← Deliverable 3
│   └── Task2_Report.md                                8 KB   (human-readable mirror)
└── src/
    ├── build_notebook.py                             24 KB   (regen notebook)
    ├── generate_report.py                            15 KB   (regen PDF report)
    ├── build_excel_workbook.py                       17 KB   (regen xlsx)
    ├── add_excel_charts.py                            8 KB   (adds native charts)
    └── build_charts_pdf.py                            9 KB   (regen 4-page PDF)
```

**Missing (from earlier stages):** `PLANNING.md`, `VALIDATION_NOTES.md`. Also missing from this session's start — they were removed at some point in a prior clean-up. Not a compliance issue, but the paper-trail from stages 1 and 3 no longer exists in the workspace.

**Task 1 local dataset lookup:** `Task_1/` contains no California Housing CSV (only `california_counties.geojson` and `pace_barry_replication.csv`, neither of which matches the required schema). The notebook's `_try_local_task1_dataset()` correctly returns `None` and falls back to the sklearn loader. Behaviour is compliant.

---

## 3. Dataset Verification (Instructions 3 – 5)

Verified computationally against a fresh `fetch_california_housing(as_frame=True)` call:

| Property | Observed | Expected reference | Match |
|---|---|---|---|
| Rows | 20 640 | ~20 640 | ✓ |
| Feature columns | 8 (`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`) | 8 numeric | ✓ |
| Target column | `HousePrice` (renamed from sklearn's `MedHouseVal`) | Median House Value | ✓ |
| Missing values | 0 | 0 | ✓ |
| Duplicate rows | 0 | 0 | ✓ |
| Target range | min 0.14999, max 5.00001, mean 2.0686 | 0.15 – 5.0 range, in units of $100 000 | ✓ |

**Target-scale documentation check:**
- Report page 1 correctly identifies the target as *"median house value, in units of $100,000"*.
- Report Results section explicitly writes *"RMSE is in the same units as HousePrice ($100k), so 0.72 corresponds to a typical error of about $72,000."* — mathematically correct.
- No artifact in the project describes the target as raw dollars, avoiding the common ×100 000 error.

---

## 4. Preprocessing / Leakage Audit (Instruction 6)

Notebook cell 7:

```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Executed only *after* the split in cell 6. Grep sweep across all source files confirms:

| Pattern | Occurrences | Meaning |
|---|---|---|
| `scaler.fit_transform(X_train)` | 1 (in the notebook) | Training-only fit ✓ |
| `scaler.transform(X_test)` | 1 (in the notebook) | Correct test transform ✓ |
| `scaler.fit_transform(X)` | 0 | No full-X fit ✓ |
| `scaler.fit(X)` | 0 | No full-X fit ✓ |

The `build_notebook.py` markdown cell "6. Feature scaling — leakage-safe" states explicitly *"Fitting the scaler on the full X before splitting would silently leak the test set's mean / std into training and inflate the reported metrics"* — an intentional deviation from the assignment PDF's sample code, which is called out and justified.

The joblib deliverable bundles the fitted scaler with the estimator as a `Pipeline`, so no downstream user can accidentally feed unscaled data to the tree.

---

## 5. Model Audit (Instruction 7)

| Model | Source-code declaration | PDF requirement | Match |
|---|---|---|---|
| Linear Regression | `LinearRegression()` | `LinearRegression()` | ✓ |
| Ridge Regression | `Ridge(alpha=1.0)` | `Ridge(alpha=1.0)` | ✓ |
| Decision Tree | `DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)` where `RANDOM_STATE = 42` | `DecisionTreeRegressor(max_depth=5)` — `random_state` not required | ✓ (with `random_state=42` added for reproducibility; `max_depth=5` unchanged) |

The `random_state=42` addition is defensible: it does not alter the required hyperparameter, and it is required to make the notebook's numbers reproducible on every re-execution.

No fourth model is trained or referenced anywhere. No hyperparameter search is performed.

---

## 6. Metric Audit (Instruction 8)

**Source-code trace:** cell 9 iterates over `models.items()`, calls `estimator.fit(X_train_scaled, y_train)`, then `estimator.predict(X_test_scaled)`, then computes RMSE + R² on `(y_test, y_pred)`. No training-set prediction call exists anywhere (`estimator.predict(X_train_scaled)` = 0 hits).

**Independent recomputation** (fresh sklearn pipeline built in a separate Python process, comparing to full-precision values in `outputs/metrics.json`):

| Model | Independent RMSE | metrics.json RMSE | |Δ| | Independent R² | metrics.json R² | |Δ| |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.7455813830 | 0.7455813830 | 0 | 0.5757877060 | 0.5757877060 | 0 |
| Ridge Regression | 0.7455567443 | 0.7455567443 | 0 | 0.5758157429 | 0.5758157429 | 0 |
| Decision Tree | 0.7242338143 | 0.7242338143 | 0 | 0.5997321244 | 0.5997321244 | 0 |

Excel workbook `3_Model_Comparison` sheet computes the same values by formula, and reads back within 4.4 × 10⁻¹⁶ (floating-point noise, no drift).

`RMSE` and `R²` implementations are:
- `sklearn.metrics.root_mean_squared_error` (preferred on scikit-learn ≥ 1.4) with `mean_squared_error(..., squared=False)` fallback for older versions — mathematically identical.
- `sklearn.metrics.r2_score` — standard sklearn implementation.

**No metric anywhere is hand-typed.** Markdown grep for `RMSE = <number>` / `R² = <number>` patterns returned 0 hits.

---

## 7. Model-Selection Audit (Instruction 9)

**Selection logic** (cell 11 of the notebook):

```
ranked = results_df.sort_values(by=["RMSE", "R2 Score"], ascending=[True, False])
best_name = ranked.index[0]
best_model = fitted[best_name]
```

- Primary criterion: lowest test RMSE (`ascending=True`).
- Tiebreak: highest test R² (`ascending=False`).
- **No hard-coded winner assignment exists in any source file.** Grep for `best_(model|name) = (LinearRegression\(|Ridge\(|DecisionTree|"Linear|"Ridge|"Decision)` returned 0 hits.

**Data-derived winner** (from truth ranking): **Decision Tree** (RMSE 0.7242, R² 0.5997).

Executed notebook stdout on cell 11 contains the literal string `→ Selected model: Decision Tree` — matches.

**Ranks (from full-precision metrics, not display-rounded):**

| Model | Test RMSE (full) | Test R² (full) | Rank | Displayed as |
|---|---:|---:|:---:|:---:|
| Linear Regression | 0.7455813830 | 0.5757877060 | 3 | 0.7456 / 0.5758 / 3 |
| Ridge Regression | 0.7455567443 | 0.5758157429 | 2 | 0.7456 / 0.5758 / 2 |
| **Decision Tree** | 0.7242338143 | 0.5997321244 | **1** | 0.7242 / 0.5997 / 1 |

At 4-dp display precision Linear and Ridge appear tied; at full precision Ridge is strictly better (both metrics). Rank correctly reflects the full-precision ordering.

The Actual-vs-Predicted plot, residual plot, saved figure filenames (`actual_vs_predicted_decision_tree.png`, `residuals_decision_tree.png`), and the persisted joblib pipeline **all point to the same programmatically-selected winner**.

---

## 8. Visualization Audit (Instruction 10 — trace-only, no edits)

Every image in the project was traced back to executable Python that produced it:

| Figure | Source of pixels | Data source |
|---|---|---|
| `outputs/figures/actual_vs_predicted_decision_tree.png` | matplotlib inside notebook cell 12 (`ax.scatter(y_test, y_pred_best, ...)`) | `y_test` and `predictions[best_name]` from cell 9 |
| `outputs/figures/residuals_decision_tree.png` | matplotlib inside notebook cell 13 (`axes[0].scatter(y_pred_best, residuals, ...)`) | `y_test - y_pred_best` |
| `outputs/figures/rmse_r2_comparison.png` | matplotlib inside notebook cell 14 (bar plots over `results_df["RMSE"]` and `["R2 Score"]`) | `results_df` from cell 9 |
| `outputs/excel_charts/Chart1_*.png` | LibreOffice rendering of the native `ScatterChart` embedded in `Task2_Predictions.xlsx` sheet `1_Actual_vs_Predicted` | Excel cell references: `Predictions!$A$2:$A$4129` × `$B$2:$B$4129` (Actual × Decision-Tree Predicted) |
| `outputs/excel_charts/Chart2_*.png` | Same, sheet `2_Residuals` | Excel cell references, with column B a formula `=Sheet1!A_i − A_i` |
| `outputs/excel_charts/Chart3_*.png` | Same, sheet `3_Model_Comparison` | Excel cell references — RMSE and R² are Excel `SQRT(SUMPRODUCT(...))` and `1 − SS_res/SS_tot` formulas |
| `outputs/excel_charts/Task2_Predictions_rendered.pdf` | LibreOffice PDF rendering of the same workbook | Same |

**Red y = x reference line** is present in both Chart 1 (matplotlib) and Chart 1 (Excel) — grep for `color="red"` and `"Perfect prediction (y = x)"` returns hits in both `build_notebook.py` and `add_excel_charts.py`.

**Excel-formula integrity** (data_only=False inspection of the workbook):
- `3_Model_Comparison!B2` = `=SQRT(SUMPRODUCT((_Predictions_Full!$A$2:$A$4129-_Predictions_Full!$B$2:$B$4129)^2)/4128)`
- `3_Model_Comparison!D2` = `=RANK(B2,$B$2:$B$4,1)`
- `2_Residuals!B2` = `='1_Actual_vs_Predicted'!A2-A2`

All formulas recalculated cleanly (recalc reported `total_errors: 0` over 4 137 formulas). Computed values match independent recomputation to floating-point precision.

No image is a screenshot, mock-up, or manually retouched raster. No pixel-editing has occurred.

---

## 9. Fabrication Search (Instruction 11)

| Signal | Result |
|---|---|
| Hard-coded winner assignment in any `.py` or `.ipynb` | **None found** |
| Hand-typed RMSE/R² numeric literals in notebook markdown | **None found** (grep for `RMSE\s*[=:]\s*[0-9]`) |
| Metric numbers in `Task2_Report.md` that don't match `metrics.json` | None (all four rounded tokens present and consistent) |
| Metric numbers in `Task2_Report.pdf` that don't match `metrics.json` | None |
| Hard-coded residual diagnostics in the report vs a fresh recompute | Report says "mean −0.0019, std 0.724"; recompute gives mean −0.001916, std 0.724231 → **matches at the stated precision** |
| Predictions in `Task2_Predictions.xlsx` vs a fresh sklearn recompute | Matches within 3 × 10⁻¹⁶ (floating-point noise) |
| Stale image referring to a non-winner model | None (both `_decision_tree` figures exist; no `_linear_regression` or `_ridge_regression` figures) |
| Notebook markdown vs executed code | Every claim traceable to the immediately-following code cell |
| Fake citations / invented references | None in any report or README |

**One nuance — residual mean sign glyph.** The report uses the Unicode minus sign `−` (U+2212), so a naive grep for `-0.0019` (ASCII hyphen) returns no hits. Grep with the Unicode minus `−0.0019` returns the hit. Not a defect; noting for future automated verification.

---

## 10. Detected Problems (findings that should be addressed before finalization)

| # | Severity | Finding | Action |
|---|---|---|---|
| F-1 | Low (documentation) | `PLANNING.md` and `VALIDATION_NOTES.md` from earlier project stages are absent from the workspace. `HOW_TO_VERIFY.md`, `README.md`, and this `TASK2_AUDIT.md` remain, but the stage-1 requirements/risk log and the stage-3 independent-audit notes are gone. | Regenerate the two files, or explicitly declare they are superseded by `HOW_TO_VERIFY.md` + `TASK2_AUDIT.md`. |
| F-2 | Very low (environment) | `_try_local_task1_dataset()` searches `Task_2/../Task_1/**/*.csv`. In this session's mount, `Task_2` sits at `/sessions/…/mnt/Task_2/` and no `Task_1` folder is a sibling, so the local lookup returns `None` silently and the sklearn fallback runs. On the user's real disk (`C:\Users\bharadwaj\Downloads\MainCrafts\`), `Task_1` and `Task_2` **are** siblings, so the lookup would function correctly. Behaviour is compliant either way; the sklearn dataset is authoritative. | Optional: expand the search to also look under `../MainCrafts/Task_1/**/*.csv` so the traversal works in this sandbox mount too. |
| F-3 | Very low (cosmetic) | `Task2_Report.md` writes the winner rank as `**1**` (bold markdown) rather than `1`. A naive grep for the exact string `"| 1 |"` misses it. Actual rendered PDF is unaffected. | Optional: also emit a plain `1` cell if you plan to grep the markdown mirror. |
| F-4 | Very low (mount duplication) | `/sessions/…/mnt/Task_2/` and `/sessions/…/mnt/MainCrafts/Task_2/` refer to the same inode. Same files, two mounts. | No action — this is a workspace-mount setup, not a real duplication. |
| F-5 | Informational (rendering artefact) | LibreOffice's default print of large sheets clips embedded charts if the print area is not set. `build_excel_workbook.py` now sets `print_area = "A1:R32"` with landscape + fit-to-page on all three chart sheets, so both viewing (Excel or LibreOffice Calc) and printing produce the full chart. Verified visually in a fresh render. | None (already fixed in the current source). |

Everything else audited is correct.

---

## 11. Optional Enhancement Classifier (Instruction 12)

| Enhancement | Assessment | Recommendation |
|---|---|---|
| Train-vs-test performance comparison | Useful | Not required by the assignment. Could add one cell reporting `train_score - test_score` per model as an overfitting sanity check. Low cost. |
| Residual analysis | Already present | Cell 13 produces residual scatter + histogram + prints residual mean/std. Nothing to add. |
| Cross-validation (5-fold) | **Unnecessary for this assignment** | Assignment specifies a single 80/20 hold-out; adding CV would exceed scope. Mentioned as a next-step in the Limitations section — appropriate treatment. |
| MAE | **Unnecessary for this assignment** | Assignment enumerates only RMSE + R². Mentioned in Limitations as a future extension. |
| Feature importances for the Decision Tree | Useful (interpretability) | Not required by the assignment; adding a small horizontal bar chart of `dt.feature_importances_` would strengthen the "why the tree wins" narrative. Optional. |
| Reproducibility check | Already present | Notebook cell 2 prints the reproducibility banner (Python, scikit-learn, pandas, numpy, matplotlib, joblib versions). Executed monotonically 1 → 15. Nothing to add. |

None of the optional items are gating for assignment compliance.

---

## 12. Verification Checklist (must all be true before finalization)

- [x] `AI_ML_Task2_Model_Comparison.ipynb` exists at the required filename and executes top-to-bottom with monotonic `execution_count` starting at 1 and no error outputs.
- [x] Dataset loaded via `fetch_california_housing(as_frame=True)`; target renamed to `HousePrice`.
- [x] 20 640 rows, 8 features, 0 missing values, 0 duplicates.
- [x] `X` and `y` separated; `X.shape[1] == 8` asserted.
- [x] `train_test_split(..., test_size=0.2, random_state=42)` → 16 512 / 4 128 rows.
- [x] `StandardScaler` fit on `X_train` only, applied to both splits (leakage-safe).
- [x] Exactly the three required models with the required hyperparameters (`random_state=42` added to the Decision Tree for reproducibility only).
- [x] RMSE and R² computed on the test set only.
- [x] Comparison table (`outputs/model_comparison.csv`) contains all three models with Rank.
- [x] All metrics in every artifact match an independent recomputation to floating-point precision.
- [x] No hard-coded winner in any source file; winner selected by `argmin(RMSE)` tiebroken by `argmax(R²)`.
- [x] Actual vs Predicted plot uses the programmatic winner (`predictions[best_name]`).
- [x] Red y = x reference line is drawn from actual Python code (matplotlib) and Excel cell references (ScatterChart series).
- [x] Residual plot exists, uses `y_test − y_pred_best`.
- [x] Joblib artifact is a `Pipeline(StandardScaler → DecisionTreeRegressor)`; reload predicts identically to a fresh tree on raw inputs.
- [x] Report is 2 pages, contains all required sections plus Limitations.
- [x] Report residual claims (mean −0.0019, std 0.724) match a fresh recompute (−0.001916, 0.724231).
- [x] Excel workbook has `Model_Comparison!B2:C4` and `RANK` cells implemented as **formulas**, not typed values; recalculation produces zero errors over 4 137 formulas.
- [x] No prohibited phrases (Claude, Gemini, "In today's rapidly evolving", TODO, FIXME) in any report or notebook.
- [ ] Regenerate `PLANNING.md` and `VALIDATION_NOTES.md`, **OR** explicitly state in `README.md` that they have been superseded by `HOW_TO_VERIFY.md` + `TASK2_AUDIT.md`. **(F-1 above)**

Once item 20 is resolved, the project is release-ready.

---

## 13. Recommended Corrections Before Final Release

1. **F-1** — Restore or explicitly supersede `PLANNING.md` and `VALIDATION_NOTES.md`. Suggested minimal action: add one line to `README.md` pointing to `TASK2_AUDIT.md` as the authoritative audit trail.
2. **F-2** — *Optional*. Broaden the Task 1 CSV scan to also traverse `Task_2/../MainCrafts/Task_1/`. Cosmetic; sklearn fallback already delivers the correct dataset.
3. All other findings (F-3, F-4, F-5) are cosmetic or environmental and do not require action.

No changes are required to the notebook code, the metric computations, the model selection, the visualization pipeline, the joblib artifact, or the report — every one of those passed both the source-level audit and the numerical recomputation.

---

*End of first-pass audit. No modifications have been made to any submission artifact during this audit.*
