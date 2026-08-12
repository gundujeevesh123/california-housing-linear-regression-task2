"""Second verification pass — extended validation runner.

Produces:
  results/model_comparison.csv     Train + Test RMSE, R², MAE and Rank per model.
  results/predictions.csv          Long-format: model, actual, predicted, residual,
                                    absolute_error (all three models, 12,384 rows).
  results/cv_scores.csv            5-fold CV mean/std of RMSE & R² per model.
  results/feature_importance.csv   Decision-Tree feature importances.
  figures/actual_vs_predicted_best_model.png   Direct from y_test / y_pred_best.

Every number is produced by scikit-learn from raw data — no notebook artifact
is read as input. This lets the validation stand independently of the notebook.
"""

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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Version-safe RMSE
try:
    from sklearn.metrics import root_mean_squared_error
    def rmse(a, b) -> float:
        return float(root_mean_squared_error(a, b))
except ImportError:
    from sklearn.metrics import mean_squared_error
    def rmse(a, b) -> float:
        return float(mean_squared_error(a, b, squared=False))


TASK2 = Path(__file__).resolve().parents[1]
RESULTS = TASK2 / "results"
FIGURES = TASK2 / "figures"
for d in (RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "HousePrice"

# --------------------------------------------------------------------------
# 1. Dataset — actual runtime values, printed for the record
# --------------------------------------------------------------------------
print("=" * 60)
print("DATASET — actual runtime values")
print("=" * 60)
bundle = fetch_california_housing(as_frame=True)
df = pd.concat([bundle.data, bundle.target.rename(TARGET)], axis=1)
print(f"Shape           : {df.shape}")
print(f"Columns         : {list(df.columns)}")
print(f"Dtypes          : {df.dtypes.to_dict()}")
print(f"Missing values  : {int(df.isna().sum().sum())}")
print(f"Duplicate rows  : {int(df.duplicated().sum())}")
print(f"Target stats    : "
      f"min={df[TARGET].min():.5f}  max={df[TARGET].max():.5f}  "
      f"mean={df[TARGET].mean():.4f}  median={df[TARGET].median():.4f}  "
      f"std={df[TARGET].std():.4f}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# --------------------------------------------------------------------------
# 2. Split + scale
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
)
print(f"Train shape     : {X_train.shape}")
print(f"Test shape      : {X_test.shape}")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Runtime confirmation that scaling is centered/unit-variance on train
tr_mean = X_train_s.mean(axis=0)
tr_std = X_train_s.std(axis=0)
print(f"X_train_scaled mean per feature  : {np.round(tr_mean, 4)}   "
      f"(all ≈ 0? {np.all(np.abs(tr_mean) < 1e-10)})")
print(f"X_train_scaled std  per feature  : {np.round(tr_std, 4)}   "
      f"(all ≈ 1? {np.all(np.abs(tr_std - 1) < 1e-10)})")

# --------------------------------------------------------------------------
# 3. Train / test metrics per model
# --------------------------------------------------------------------------
print()
print("=" * 60)
print("METRICS — computed at runtime from actual predictions")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
}

rows = []
predictions = {}  # store test predictions for each model
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    train_rmse = rmse(y_train, y_pred_train)
    test_rmse  = rmse(y_test,  y_pred_test)
    train_r2   = float(r2_score(y_train, y_pred_train))
    test_r2    = float(r2_score(y_test,  y_pred_test))
    test_mae   = float(mean_absolute_error(y_test, y_pred_test))

    rows.append({
        "Model":       name,
        "Train RMSE":  train_rmse,
        "Test RMSE":   test_rmse,
        "Train R2":    train_r2,
        "Test R2":     test_r2,
        "Test MAE":    test_mae,
        "R2 Gap":      train_r2 - test_r2,
    })
    predictions[name] = y_pred_test
    print(f"  {name:<20}  train RMSE={train_rmse:.4f}  test RMSE={test_rmse:.4f}  "
          f"train R²={train_r2:.4f}  test R²={test_r2:.4f}  "
          f"MAE={test_mae:.4f}  R² gap={train_r2 - test_r2:+.4f}")

# --------------------------------------------------------------------------
# 4. Programmatic ranking (primary = min test RMSE, tiebreak = max test R²)
# --------------------------------------------------------------------------
rows.sort(key=lambda r: (r["Test RMSE"], -r["Test R2"]))
for i, r in enumerate(rows, 1):
    r["Rank"] = i

results_df = (
    pd.DataFrame(rows)
      .set_index("Model")
      .reindex(list(models.keys()))       # restore Linear→Ridge→DT display order
)
results_df.to_csv(RESULTS / "model_comparison.csv",
                  float_format="%.6f")
print()
print("Ranked comparison table (written to results/model_comparison.csv):")
print(results_df.round(6).to_string())

# --------------------------------------------------------------------------
# 5. Determine the winner programmatically, print + record
# --------------------------------------------------------------------------
best_name = min(rows, key=lambda r: (r["Test RMSE"], -r["Test R2"]))["Model"]
best_row = next(r for r in rows if r["Model"] == best_name)
print()
print(f"Programmatically selected winner: {best_name}")
print(f"  Test RMSE = {best_row['Test RMSE']:.4f}")
print(f"  Test R²   = {best_row['Test R2']:.4f}")
print(f"  Test MAE  = {best_row['Test MAE']:.4f}")

# --------------------------------------------------------------------------
# 6. Long-format predictions.csv (auditable numerical foundation)
# --------------------------------------------------------------------------
long_rows = []
for name, yhat in predictions.items():
    residual = y_test.to_numpy() - yhat
    long_rows.append(pd.DataFrame({
        "model":          name,
        "actual":         y_test.to_numpy(),
        "predicted":      yhat,
        "residual":       residual,
        "absolute_error": np.abs(residual),
    }))
pred_df = pd.concat(long_rows, ignore_index=True)
pred_df.to_csv(RESULTS / "predictions.csv", index=False, float_format="%.6f")
print(f"\nWrote {RESULTS / 'predictions.csv'} — {len(pred_df):,} rows "
      f"({len(pred_df) // len(models)} test rows × {len(models)} models)")

# --------------------------------------------------------------------------
# 7. 5-fold Cross-Validation on the TRAINING set only (never on test)
#    Each fold refits the scaler inside a Pipeline to avoid leakage.
# --------------------------------------------------------------------------
print()
print("=" * 60)
print("CROSS-VALIDATION (5-fold, on training data)")
print("=" * 60)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model.__class__(**model.get_params()))])
    r2_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    # For RMSE, use neg_root_mean_squared_error (available since sklearn 1.1)
    try:
        rmse_scores = -cross_val_score(pipe, X_train, y_train, cv=kf,
                                       scoring="neg_root_mean_squared_error",
                                       n_jobs=-1)
    except ValueError:
        rmse_scores = np.sqrt(-cross_val_score(pipe, X_train, y_train, cv=kf,
                                               scoring="neg_mean_squared_error",
                                               n_jobs=-1))
    cv_rows.append({
        "Model":        name,
        "CV RMSE mean": float(rmse_scores.mean()),
        "CV RMSE std":  float(rmse_scores.std(ddof=1)),
        "CV R2 mean":   float(r2_scores.mean()),
        "CV R2 std":    float(r2_scores.std(ddof=1)),
    })
    print(f"  {name:<20}  CV RMSE = {rmse_scores.mean():.4f} ± {rmse_scores.std(ddof=1):.4f}   "
          f"CV R² = {r2_scores.mean():.4f} ± {r2_scores.std(ddof=1):.4f}")

cv_df = pd.DataFrame(cv_rows).set_index("Model")
cv_df.to_csv(RESULTS / "cv_scores.csv", float_format="%.6f")

# --------------------------------------------------------------------------
# 8. Decision Tree feature importance (winner-specific interpretability)
# --------------------------------------------------------------------------
dt = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE).fit(X_train_s, y_train)
fi = pd.DataFrame({
    "feature":    X.columns,
    "importance": dt.feature_importances_,
}).sort_values("importance", ascending=False)
fi.to_csv(RESULTS / "feature_importance.csv", index=False, float_format="%.6f")
print()
print("Decision-Tree feature importance:")
print(fi.to_string(index=False))

# --------------------------------------------------------------------------
# 9. Actual vs Predicted plot for the actual winner — regenerated directly
#    from y_test and predictions[best_name]
# --------------------------------------------------------------------------
y_pred_best = predictions[best_name]
lo = float(min(y_test.min(), y_pred_best.min()))
hi = float(max(y_test.max(), y_pred_best.max()))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.35, s=14, edgecolor="none",
           color="#4c78a8", label="Test-set predictions")
ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.6,
        label="Perfect prediction (y = x)")
ax.set_xlabel("Actual House Prices  (units of $100,000)")
ax.set_ylabel("Predicted House Prices  (units of $100,000)")
ax.set_title(f"Actual vs Predicted House Prices — {best_name}\n"
             f"(test set, n={len(y_test)}, RMSE={best_row['Test RMSE']:.4f}, R²={best_row['Test R2']:.4f})")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = FIGURES / "actual_vs_predicted_best_model.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nWrote {out}")

# --------------------------------------------------------------------------
# 10. Also generate residual and comparison companion figures (from real data)
# --------------------------------------------------------------------------
residual = y_test.to_numpy() - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(y_pred_best, residual, alpha=0.35, s=14, edgecolor="none", color="#6baed6")
axes[0].axhline(0, color="red", linewidth=1.2)
axes[0].set_xlabel("Predicted House Prices")
axes[0].set_ylabel("Residual (actual − predicted)")
axes[0].set_title(f"Residuals vs Predicted — {best_name}")
axes[0].grid(True, alpha=0.3)
axes[1].hist(residual, bins=40, edgecolor="white")
axes[1].axvline(0, color="red", linewidth=1.2)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Residual distribution  (mean={residual.mean():+.4f}, std={residual.std():.4f})")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "residuals_best_model.png", dpi=150)
plt.close(fig)
print(f"Wrote {FIGURES / 'residuals_best_model.png'}")

# --------------------------------------------------------------------------
# 11. Environment/version record
# --------------------------------------------------------------------------
env = {
    "python":       sys.version.split()[0],
    "platform":     platform.platform(),
    "scikit-learn": sklearn.__version__,
    "pandas":       pd.__version__,
    "numpy":        np.__version__,
    "matplotlib":   matplotlib.__version__,
    "joblib":       joblib.__version__,
    "random_state": RANDOM_STATE,
    "test_size":    0.2,
    "n_train":      len(X_train),
    "n_test":       len(X_test),
    "winner":       best_name,
}
(RESULTS / "environment.json").write_text(json.dumps(env, indent=2))
print(f"Wrote {RESULTS / 'environment.json'}")

# --------------------------------------------------------------------------
# 12. Final summary snapshot for VALIDATION_RESULTS.md
# --------------------------------------------------------------------------
summary = {
    "dataset": {
        "source":  "sklearn.datasets.fetch_california_housing(as_frame=True)",
        "rows":    int(df.shape[0]),
        "columns": list(df.columns),
        "missing": int(df.isna().sum().sum()),
        "target":  TARGET,
        "target_min":    float(df[TARGET].min()),
        "target_max":    float(df[TARGET].max()),
        "target_mean":   float(df[TARGET].mean()),
        "target_median": float(df[TARGET].median()),
        "target_std":    float(df[TARGET].std()),
    },
    "split": {"test_size": 0.2, "random_state": RANDOM_STATE,
              "n_train": len(X_train), "n_test": len(X_test)},
    "results": rows,
    "cv":      cv_rows,
    "winner":  best_name,
    "environment": env,
    "feature_importance": fi.to_dict(orient="records"),
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"Wrote {RESULTS / 'summary.json'}")
