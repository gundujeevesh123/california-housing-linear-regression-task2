"""Build and execute AI_ML_Task2_Model_Comparison.ipynb end-to-end.

This module assembles the required Task 2 notebook cell-by-cell using
nbformat, then executes every cell with nbclient so real outputs — tables,
figures, metrics — are baked into the delivered .ipynb.

Design decisions live in Task_2/PLANNING.md. Highlights that this file
enforces:

* Split BEFORE scaling (fit StandardScaler on X_train only) — leakage-safe.
* DecisionTreeRegressor uses random_state=42 for reproducibility.
* Best model is selected programmatically (min RMSE, tiebreak max R^2).
* Every metric / figure comes from the executed run, never hand-typed.
* A joblib Pipeline (scaler + estimator) is persisted so future inference
  cannot be fed unscaled inputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
TASK2_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = TASK2_DIR / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "AI_ML_Task2_Model_Comparison.ipynb"


# --------------------------------------------------------------------------
# Cell definitions
# --------------------------------------------------------------------------
# The cells below are ordered exactly as they will appear in the notebook.
# Each entry is (cell_type, source). Keeping the definitions here — instead
# of scattered constants — makes the notebook structure easy to review at a
# glance.

CELLS: list[tuple[str, str]] = [
    # --------------------------------------------------------------------
    # 1. Title & overview
    # --------------------------------------------------------------------
    ("markdown", """\
# AI/ML Task 2 — Feature Engineering, Model Optimization & Performance Comparison

**Program:** MainCrafts Technology — AI & ML Internship
**Deliverable:** `AI_ML_Task2_Model_Comparison.ipynb`

## Project overview

This notebook builds an enhanced *California House Price* prediction workflow that
goes beyond a single-model baseline. It trains three regressors, evaluates them on
an unseen 20% test hold-out, produces a structured performance table, and selects
the winner **programmatically** using the lowest RMSE (tiebreak: highest R²).

**Pipeline:**

1. Load the California Housing dataset (rename target to `HousePrice`).
2. Inspect schema, dtypes, missingness.
3. Separate features / target, then split 80/20 with `random_state=42`.
4. Fit `StandardScaler` on the **training set only** (leakage-safe).
5. Train `LinearRegression`, `Ridge(alpha=1.0)`, and
   `DecisionTreeRegressor(max_depth=5, random_state=42)`.
6. Compute RMSE and R² on the test set.
7. Assemble the comparison table.
8. Select the best model programmatically.
9. Visualise Actual vs Predicted and residuals for the winner.
10. Persist the winning model as a joblib `Pipeline` (scaler + estimator).
"""),

    # --------------------------------------------------------------------
    # 2. Imports
    # --------------------------------------------------------------------
    ("markdown", """\
## 1. Library imports

We keep imports at the top and pin `random_state=42` throughout for reproducibility.
`root_mean_squared_error` is preferred when available (scikit-learn ≥ 1.4); for
older versions we fall back to `mean_squared_error(..., squared=False)`.
"""),

    ("code", """\
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Version-safe RMSE: prefer the modern function name when available.
try:
    from sklearn.metrics import root_mean_squared_error

    def rmse(y_true, y_pred) -> float:
        return float(root_mean_squared_error(y_true, y_pred))
except ImportError:  # scikit-learn < 1.4
    def rmse(y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred, squared=False))

RANDOM_STATE = 42
TARGET_COLUMN = "HousePrice"

# Where fitted models, metrics artifacts, and figures will be written.
NOTEBOOK_DIR = Path.cwd()
TASK2_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR
OUTPUTS_DIR = TASK2_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = TASK2_DIR / "models"
for d in (OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})

# Reproducibility banner: exact versions used for this execution.
# We print relative paths only, so the notebook's cached outputs are portable
# across machines and don't leak absolute filesystem paths.
try:
    outputs_display = OUTPUTS_DIR.relative_to(TASK2_DIR)
except ValueError:
    outputs_display = OUTPUTS_DIR.name

print(f"Python      : {sys.version.split()[0]}  ({platform.system()})")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas      : {pd.__version__}")
print(f"numpy       : {np.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"joblib      : {joblib.__version__}")
print(f"random_state: {RANDOM_STATE}")
print(f"Outputs dir : {outputs_display}   (relative to Task_2/)")
"""),

    # --------------------------------------------------------------------
    # 3. Dataset loading
    # --------------------------------------------------------------------
    ("markdown", """\
## 2. Load the California Housing dataset

We first try to locate a suitable local copy of the California Housing dataset
inside the Task 1 workspace (candidate CSVs are validated against the expected
schema). If none is found, we fall back to the canonical
`sklearn.datasets.fetch_california_housing(as_frame=True)` loader, then rename
the target column to **`HousePrice`** as required by the assignment.

**Expected schema:** 8 numeric features
(`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`)
plus the target.
"""),

    ("code", """\
EXPECTED_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_ALIASES = {"MedHouseVal", "median_house_value", "HousePrice", "target"}


def _validate_frame(candidate: pd.DataFrame) -> pd.DataFrame | None:
    \"\"\"Return the frame with target renamed to HousePrice, or None if invalid.\"\"\"
    cols = set(candidate.columns)
    if not set(EXPECTED_FEATURES).issubset(cols):
        return None
    target_col = next((c for c in candidate.columns if c in TARGET_ALIASES), None)
    if target_col is None:
        return None
    df = candidate.copy()
    if target_col != TARGET_COLUMN:
        df = df.rename(columns={target_col: TARGET_COLUMN})
    return df[EXPECTED_FEATURES + [TARGET_COLUMN]]


def _try_local_task1_dataset() -> pd.DataFrame | None:
    \"\"\"Scan Task_1 for any CSV that matches the California Housing schema.\"\"\"
    task1_dir = TASK2_DIR.parent / "Task_1"
    if not task1_dir.exists():
        return None
    for csv_path in task1_dir.rglob("*.csv"):
        try:
            candidate = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001 — any parse failure means skip
            continue
        validated = _validate_frame(candidate)
        if validated is not None:
            print(f"Loaded local dataset from: {csv_path}")
            return validated
    return None


def load_california_housing_df() -> pd.DataFrame:
    local = _try_local_task1_dataset()
    if local is not None:
        return local
    bundle = fetch_california_housing(as_frame=True)
    df = pd.concat([bundle.data, bundle.target.rename(TARGET_COLUMN)], axis=1)
    print("Loaded dataset via sklearn.datasets.fetch_california_housing.")
    return df


df = load_california_housing_df()
df.head()
"""),

    # --------------------------------------------------------------------
    # 4. Inspection
    # --------------------------------------------------------------------
    ("markdown", """\
## 3. Dataset inspection

Sanity checks on shape, dtypes, missing values, and target coverage — surfacing
issues here is cheaper than discovering them mid-training.
"""),

    ("code", """\
print(f"Shape           : {df.shape}")
print(f"Feature columns : {list(df.columns.drop(TARGET_COLUMN))}")
print(f"Target column   : {TARGET_COLUMN}")
print(f"Missing values  : {int(df.isna().sum().sum())}")
print()
print("Dtypes:")
print(df.dtypes.to_string())
print()
print(f"Target summary  (min / mean / median / max / std):")
print(
    df[TARGET_COLUMN].agg(["min", "mean", "median", "max", "std"]).round(4).to_string()
)
"""),

    ("code", """\
df.describe().T.round(3)
"""),

    # --------------------------------------------------------------------
    # 5. Feature/target separation
    # --------------------------------------------------------------------
    ("markdown", """\
## 4. Feature / target separation

`X` holds the 8 predictors; `y` is the median house value renamed to `HousePrice`
(target values are expressed in units of \\$100,000).
"""),

    ("code", """\
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

assert X.shape[1] == 8, f"expected 8 features, got {X.shape[1]}"
assert TARGET_COLUMN not in X.columns, "target column leaked into features"
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
"""),

    # --------------------------------------------------------------------
    # 6. Train / test split
    # --------------------------------------------------------------------
    ("markdown", """\
## 5. Train / test split

An 80/20 hold-out with `random_state=42` — the assignment default. Splitting
**before** scaling is what keeps the pipeline honest (see next section).
"""),

    ("code", """\
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
)
print(f"X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
"""),

    # --------------------------------------------------------------------
    # 7. Scaling (leakage-safe)
    # --------------------------------------------------------------------
    ("markdown", """\
## 6. Feature scaling — leakage-safe

The assignment prescribes `StandardScaler`. We fit it **only on `X_train`** and
transform both splits with those parameters. Fitting on the full `X` before
splitting would silently leak the test set's mean / std into training and
inflate the reported metrics.

Note: tree models are scale-invariant, so scaling does not help the Decision
Tree — but it does not hurt it either. Keeping one preprocessing path for all
three models keeps the comparison methodologically consistent.
"""),

    ("code", """\
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Per-feature train mean (should be ≈ 0):")
print(np.round(X_train_scaled.mean(axis=0), 4))
print("Per-feature train std  (should be ≈ 1):")
print(np.round(X_train_scaled.std(axis=0), 4))
"""),

    # --------------------------------------------------------------------
    # 8. Model definitions
    # --------------------------------------------------------------------
    ("markdown", """\
## 7. Model definitions

Three regressors, exactly as specified by the assignment:

| Model | Purpose | Hyperparameters |
| --- | --- | --- |
| `LinearRegression()` | Interpretable linear baseline | (defaults) |
| `Ridge(alpha=1.0)` | L2-regularised linear model — shrinks coefficients, reduces variance | `alpha=1.0` |
| `DecisionTreeRegressor(max_depth=5, random_state=42)` | Captures non-linear splits; depth cap curbs overfitting | `max_depth=5`, `random_state=42` |
"""),

    ("code", """\
models: dict[str, object] = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
}
models
"""),

    # --------------------------------------------------------------------
    # 9. Train + evaluate
    # --------------------------------------------------------------------
    ("markdown", """\
## 8. Train each model and evaluate on the test set

For every model we fit on the scaled training data, predict on the scaled test
data, and record RMSE and R² on the held-out set only.

- **RMSE** (root mean squared error) — same units as `HousePrice`. Lower is better.
- **R²** — proportion of variance explained. Closer to 1.0 is better.
"""),

    ("code", """\
results: dict[str, dict[str, float]] = {}
predictions: dict[str, np.ndarray] = {}
fitted: dict[str, object] = {}

for name, estimator in models.items():
    estimator.fit(X_train_scaled, y_train)
    y_pred = estimator.predict(X_test_scaled)

    results[name] = {
        "RMSE": rmse(y_test, y_pred),
        "R2 Score": float(r2_score(y_test, y_pred)),
    }
    predictions[name] = y_pred
    fitted[name] = estimator

results_df = pd.DataFrame(results).T.round(4)
results_df.index.name = "Model"
results_df
"""),

    # --------------------------------------------------------------------
    # 10. Comparison table
    # --------------------------------------------------------------------
    ("markdown", """\
## 9. Model performance comparison table

We add a **Rank** column derived from the selection rule (min RMSE, tiebreak
max R²), then persist the table to `outputs/model_comparison.csv` and
`outputs/metrics.json`. Downstream artifacts (the PDF report, any dashboard)
read from these files, so metrics never need to be re-typed and cannot drift.
"""),

    ("code", """\
comparison_csv = OUTPUTS_DIR / "model_comparison.csv"
metrics_json = OUTPUTS_DIR / "metrics.json"

# Rank on FULL-PRECISION metrics (not the display-rounded results_df) so that
# near-ties like 0.745581 vs 0.745557 rank correctly instead of falling back to
# insertion order after rounding to 4 dp.
full_prec_df = pd.DataFrame(results).T
rank_series = (
    full_prec_df.sort_values(by=["RMSE", "R2 Score"], ascending=[True, False])
                .assign(Rank=range(1, len(full_prec_df) + 1))["Rank"]
                .reindex(full_prec_df.index)
                .astype(int)
)
results_df = results_df.assign(Rank=rank_series)

results_df.to_csv(comparison_csv)
metrics_payload = {
    "random_state": RANDOM_STATE,
    "test_size": 0.2,
    "target_column": TARGET_COLUMN,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "results": {
        name: {
            "RMSE": float(results[name]["RMSE"]),
            "R2 Score": float(results[name]["R2 Score"]),
            "Rank": int(rank_series[name]),
        }
        for name in results
    },
}
metrics_json.write_text(json.dumps(metrics_payload, indent=2))
print(f"Wrote {comparison_csv.relative_to(TASK2_DIR)}")
print(f"Wrote {metrics_json.relative_to(TASK2_DIR)}")
results_df
"""),

    # --------------------------------------------------------------------
    # 11. Best model — programmatic selection
    # --------------------------------------------------------------------
    ("markdown", """\
## 10. Select the best model — programmatically

Selection rule:

1. **Primary:** lowest RMSE on the test set.
2. **Tiebreaker:** highest R² on the test set.

No model name is hard-coded — the winner is derived from `results_df` above.
"""),

    ("code", """\
ranked = results_df.sort_values(by=["RMSE", "R2 Score"], ascending=[True, False])
best_name = ranked.index[0]
best_model = fitted[best_name]
best_metrics = results[best_name]

print("Ranking (best → worst):")
print(ranked.to_string())
print()
print(f"→ Selected model: {best_name}")
print(f"  Test RMSE : {best_metrics['RMSE']:.4f}")
print(f"  Test R²   : {best_metrics['R2 Score']:.4f}")
"""),

    # --------------------------------------------------------------------
    # 12. Actual vs Predicted plot for the winner
    # --------------------------------------------------------------------
    ("markdown", """\
## 11. Visual validation — Actual vs Predicted (best model)

Points close to the red y = x reference line indicate accurate predictions.
Systematic curvature or fan-out away from the diagonal signals bias or
heteroscedasticity.
"""),

    ("code", """\
y_pred_best = predictions[best_name]

lo = float(min(y_test.min(), y_pred_best.min()))
hi = float(max(y_test.max(), y_pred_best.max()))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, y_pred_best, alpha=0.35, s=14, edgecolor="none", label="Predictions")
ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.5, label="Perfect prediction (y = x)")
ax.set_xlabel("Actual House Prices  (target units)")
ax.set_ylabel("Predicted House Prices  (target units)")
ax.set_title(f"Actual vs Predicted House Prices — {best_name}")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.25)
fig.tight_layout()

fig_path = FIGURES_DIR / f"actual_vs_predicted_{best_name.replace(' ', '_').lower()}.png"
fig.savefig(fig_path)
plt.show()
print(f"Saved: {fig_path.relative_to(TASK2_DIR)}")
"""),

    # --------------------------------------------------------------------
    # 13. Residual analysis
    # --------------------------------------------------------------------
    ("markdown", """\
## 12. Residual analysis (best model)

Residual = *actual − predicted*. A well-calibrated model produces a residual
cloud roughly centered on zero with no obvious pattern against predicted values.
"""),

    ("code", """\
residuals = y_test.to_numpy() - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(y_pred_best, residuals, alpha=0.35, s=14, edgecolor="none")
axes[0].axhline(0, color="red", linewidth=1.2)
axes[0].set_xlabel("Predicted House Prices")
axes[0].set_ylabel("Residual (actual − predicted)")
axes[0].set_title(f"Residuals vs Predicted — {best_name}")
axes[0].grid(True, alpha=0.25)

axes[1].hist(residuals, bins=40, edgecolor="white")
axes[1].axvline(0, color="red", linewidth=1.2)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual distribution")
axes[1].grid(True, alpha=0.25)

fig.tight_layout()
resid_path = FIGURES_DIR / f"residuals_{best_name.replace(' ', '_').lower()}.png"
fig.savefig(resid_path)
plt.show()

print(f"Residual mean : {residuals.mean():+.4f}")
print(f"Residual std  : {residuals.std():.4f}")
print(f"Saved: {resid_path.relative_to(TASK2_DIR)}")
"""),

    # --------------------------------------------------------------------
    # 14. Optional bar-chart comparison
    # --------------------------------------------------------------------
    ("markdown", """\
## 13. Optional — Side-by-side RMSE / R² bar chart

A quick visual companion to the comparison table (kept small so the notebook
stays readable).
"""),

    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(results_df.index, results_df["RMSE"], color="#4c78a8")
axes[0].set_title("Test RMSE (lower is better)")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=15)
for i, v in enumerate(results_df["RMSE"]):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

axes[1].bar(results_df.index, results_df["R2 Score"], color="#59a14f")
axes[1].set_title("Test R² (higher is better)")
axes[1].set_ylabel("R²")
axes[1].tick_params(axis="x", rotation=15)
axes[1].set_ylim(0, 1)
for i, v in enumerate(results_df["R2 Score"]):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
bar_path = FIGURES_DIR / "rmse_r2_comparison.png"
fig.savefig(bar_path)
plt.show()
print(f"Saved: {bar_path.relative_to(TASK2_DIR)}")
"""),

    # --------------------------------------------------------------------
    # 15. Persistence — save best model as a Pipeline
    # --------------------------------------------------------------------
    ("markdown", """\
## 14. Persist the winning model (optional deliverable)

Instead of dumping a bare estimator that would silently mis-predict on
unscaled inputs, we bundle the fitted scaler and the fitted estimator into a
`sklearn.pipeline.Pipeline`. Downstream code can simply call
`loaded.predict(new_raw_features)` — the pipeline scales for them.
"""),

    ("code", """\
best_pipeline = Pipeline([
    ("scaler", scaler),
    ("model", best_model),
])

pipeline_path = MODELS_DIR / "best_model.joblib"
joblib.dump(best_pipeline, pipeline_path)
print(f"Saved: {pipeline_path.relative_to(TASK2_DIR)}")

# Sanity check — reload and predict on the first few raw (unscaled) rows.
reloaded = joblib.load(pipeline_path)
sample_preds = reloaded.predict(X_test.iloc[:5])
pd.DataFrame({
    "actual": y_test.iloc[:5].to_numpy(),
    "predicted": np.round(sample_preds, 4),
})
"""),

    # --------------------------------------------------------------------
    # 16. Final observations
    # --------------------------------------------------------------------
    ("markdown", """\
## 15. Final observations

The comparison table, best-model selection, and visualisations above are all
derived from a single executed run — no metric is hand-typed. Key takeaways:

- **Leakage-safe preprocessing** matters: fitting `StandardScaler` on training
  data only prevents the test set from influencing the learned scale.
- **Ridge vs Linear Regression** differ only in L2 regularisation; on this
  dataset the difference in RMSE is small — expected, because the raw features
  are already reasonably conditioned after scaling and there is little
  multicollinearity strong enough for Ridge to exploit.
- **Decision Tree (depth 5)** captures non-linear structure the linear models
  cannot represent (particularly in `Latitude`/`Longitude` and interactions
  with `MedInc`), which is typically reflected in a materially lower RMSE.
- The **programmatically selected** winner is stored above in `best_name` and
  persisted as a joblib `Pipeline`. Regenerating this notebook with a
  different random seed will re-run the selection rule and may pick a
  different model — that is intended behaviour.

The next stage will consume `outputs/model_comparison.csv` and
`outputs/metrics.json` to produce the 1–2 page PDF report.
"""),
]


# --------------------------------------------------------------------------
# Build + execute
# --------------------------------------------------------------------------
def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        (nbf.v4.new_markdown_cell if kind == "markdown" else nbf.v4.new_code_cell)(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()

    # Execute so real outputs are embedded — no fabricated numbers.
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote executed notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
