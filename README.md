# AI/ML Task 2 — Feature Engineering, Model Optimization & Performance Comparison

**Program:** MainCrafts Technology — AI & ML Internship
**Assignment:** Compare `LinearRegression`, `Ridge(alpha=1.0)`, and
`DecisionTreeRegressor(max_depth=5)` on the California Housing dataset, using
`StandardScaler` preprocessing and an 80/20 train/test split with `random_state=42`.
Report RMSE, R², select the best-performing model programmatically, and visualise
Actual vs Predicted for the winner.

---

## Where to look first (evaluator)

| Priority | Artifact | What it is |
| --- | --- | --- |
| 1 | `reports/Task2_Report.pdf` | 2-page technical report — read this first. |
| 2 | `notebooks/AI_ML_Task2_Model_Comparison.ipynb` | Fully executed Jupyter notebook — 31 cells, 15 code cells, monotonic execution counters. |
| 3 | `outputs/model_comparison.csv` | Comparison table (Model, RMSE, R², Rank) — the single source of truth for every metric. |
| 4 | `outputs/figures/actual_vs_predicted_decision_tree.png` | Winner's Actual vs Predicted plot with red y = x reference line. |
| 5 | `outputs/figures/residuals_decision_tree.png` | Residual diagnostics for the winner. |
| 6 | `models/best_model.joblib` | Winning model bundled with the fitted StandardScaler as a `sklearn.pipeline.Pipeline` (loadable via `joblib.load(...)`). |

## Verified results (from the executed notebook)

| Model | Test RMSE | Test R² | Rank |
| --- | ---: | ---: | :---: |
| Linear Regression | 0.7456 | 0.5758 | 3 |
| Ridge Regression | 0.7456 | 0.5758 | 2 |
| **Decision Tree (max_depth = 5)** | **0.7242** | **0.5997** | **1** |

**Selected model:** Decision Tree Regressor — chosen by the programmatic rule
`argmin(RMSE)`, tiebreak `argmax(R²)`. Not hard-coded anywhere.

## Reproducing the results

```bash
# from Task_2/
python3 src/build_notebook.py         # builds + executes the notebook
python3 src/generate_report.py        # rebuilds Task2_Report.pdf from metrics.json
```

Both scripts are deterministic (`random_state=42`); rerunning them regenerates every
artifact byte-identically. Every number in the report is pulled from
`outputs/metrics.json` — no metric is hand-typed.

For the extended validation artifacts under `results/` (train/test metrics, MAE, 5-fold CV,
feature importance, long-format predictions):

```bash
python3 src/run_validation.py
```

## Project layout

```
Task_2/
├── README.md                                   # this file
├── HOW_TO_VERIFY.md                            # Six independent ways to verify every number
├── TASK2_AUDIT.md                              # Stage-6 code-audit findings
├── VALIDATION_RESULTS.md                       # Stage-7 execution + reproducibility record
├── notebooks/
│   └── AI_ML_Task2_Model_Comparison.ipynb      # Deliverable 1 (executed)
├── outputs/
│   ├── model_comparison.csv                    # Deliverable 2 (with Rank column)
│   ├── metrics.json                            # Full-precision metrics + run config
│   ├── Task2_Predictions.xlsx                  # Excel workbook (3 datasets + 3 native charts)
│   ├── figures/
│   │   ├── actual_vs_predicted_decision_tree.png
│   │   ├── residuals_decision_tree.png
│   │   └── rmse_r2_comparison.png              # Side-by-side bar chart
│   └── excel_charts/
│       ├── Chart1_ActualVsPredicted_DecisionTree.png
│       ├── Chart2_Residuals_DecisionTree.png
│       ├── Chart3_RMSE_R2_Comparison.png
│       └── Task2_Predictions_rendered.pdf      # 4-page charts PDF
├── results/                                    # Extended validation (Stage 7)
│   ├── model_comparison.csv                    # Train + Test RMSE/R² + MAE + Rank
│   ├── predictions.csv                         # Long-format, 12,384 rows (audit trail)
│   ├── cv_scores.csv                           # 5-fold CV mean/std supporting evidence
│   ├── feature_importance.csv                  # Decision Tree feature importances
│   ├── environment.json                        # Library versions + run config
│   └── summary.json                            # Single-file snapshot
├── figures/                                    # Regenerated at Stage-7 validation
│   ├── actual_vs_predicted_best_model.png      # Spec-mandated filename
│   └── residuals_best_model.png
├── models/
│   └── best_model.joblib                       # Optional persistence (Pipeline)
├── reports/
│   ├── Task2_Report.pdf                        # Deliverable 3 (2 pages)
│   └── Task2_Report.md                         # Human-readable mirror
└── src/
    ├── build_notebook.py                       # Assembles + executes the notebook
    ├── generate_report.py                      # Renders the PDF from metrics.json
    ├── build_excel_workbook.py                 # Regenerates the .xlsx with charts
    ├── add_excel_charts.py                     # Adds native openpyxl charts
    ├── build_charts_pdf.py                     # Builds the 4-page charts PDF
    └── run_validation.py                       # Extended validation runner (Stage 7)
```

## Environment

Python 3.10 · scikit-learn 1.7.2 · pandas · numpy · matplotlib · joblib · fpdf2 ·
nbformat · nbclient. Exact versions are printed at the top of the executed notebook.
