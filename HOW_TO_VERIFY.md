# How to verify every output in this Task 2 submission

You do not need to trust that any number in this project is real — every metric
can be recomputed, every chart can be redrawn from raw data, and every claim
in the report ties back to a file you can open. This guide walks through six
independent verification paths, ordered from quickest to most rigorous.

---

## 1. One-minute visual sanity check (no tools needed)

Open these five files and confirm they show the same three numbers.

| Open | Look at | Should show |
| --- | --- | --- |
| `outputs\model_comparison.csv` | Any text editor / Excel | 4 rows: header + Linear (0.7456, 0.5758, rank 3), Ridge (0.7456, 0.5758, rank 2), Decision Tree (0.7242, 0.5997, rank 1) |
| `outputs\metrics.json` | Any text editor | Same numbers, at full precision |
| `reports\Task2_Report.pdf` | Any PDF reader | Same numbers on page 1 table; page count = 2 |
| `reports\Task2_Report.md` | Any Markdown viewer | Same table |
| `outputs\excel_charts\Task2_Predictions_rendered.pdf` | Any PDF reader | Same table on page 1; charts on pages 2, 3, 4 |

If those five files disagree on any number, something has drifted — flag it.

---

## 2. Prove the Excel metrics are formulas, not hard-coded values

Open `outputs\Task2_Predictions.xlsx` in Excel.

Go to the **Comparison** sheet and click each of the following cells. The
Excel formula bar will show a **formula**, not a typed number:

| Cell | Formula you should see |
| --- | --- |
| `B2` | `=SQRT(SUMPRODUCT((Predictions!$B$2:$B$4129-Predictions!$C$2:$C$4129)^2)/4128)` |
| `B3` | `=SQRT(SUMPRODUCT((Predictions!$B$2:$B$4129-Predictions!$D$2:$D$4129)^2)/4128)` |
| `B4` | `=SQRT(SUMPRODUCT((Predictions!$B$2:$B$4129-Predictions!$E$2:$E$4129)^2)/4128)` |
| `C2` | `=1-SUMPRODUCT((Predictions!$B$2:$B$4129-Predictions!$C$2:$C$4129)^2)/SUMPRODUCT((Predictions!$B$2:$B$4129-AVERAGE(Predictions!$B$2:$B$4129))^2)` |
| `D2` | `=RANK(B2,$B$2:$B$4,1)` |

If you can see those formulas, then the displayed RMSE / R² / Rank came from
Excel's calculator on your machine, not from anything I typed.

**Bonus check** — go to the **Predictions** sheet, click cell `F2`. You should
see `=B2-E2`. That is the residual (Actual − Decision Tree prediction)
computed as a formula for every row.

---

## 3. Prove the charts are bound to live data, not embedded images

Still in `Task2_Predictions.xlsx`:

1. Click the tab **Chart_1_Actual_vs_Predicted**.
2. Right-click anywhere on the scatter plot → **Select Data** (or, in newer
   Excel, click the chart → the Ribbon shows Chart Design → **Select Data**).
3. You will see two data series. Click the first one ("Test-set predictions")
   → **Edit**. The X-values point to `Predictions!$B$2:$B$4129` (Actual) and
   the Y-values point to `Predictions!$E$2:$E$4129` (Decision Tree
   predictions). The red diagonal series points to
   `YX_ReferenceLine!$B$2:$C$3`.
4. Change any cell in `Predictions!E2:E4129` and the corresponding chart dot
   moves. That is impossible for an embedded image; only a live chart behaves
   this way.

Repeat for **Chart_2_Residuals** (references `Predictions!E:E` and
`Predictions!F:F`) and **Chart_3_RMSE_R2_Comparison** (references
`Comparison!A1:C4`).

---

## 4. Re-execute the Jupyter notebook end-to-end

Requirements: Python 3.9+ and the packages listed in
`Task_1\requirements.txt` (pandas, numpy, scikit-learn, matplotlib, joblib,
jupyter, nbclient). If you do not already have them installed:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib joblib jupyter nbclient nbformat nbconvert
```

Then, from the `Task_2` folder, run either of these:

```powershell
# Option A — regenerate + execute the notebook via nbclient
python src\build_notebook.py

# Option B — execute the existing notebook as an independent user would
jupyter nbconvert --to notebook --execute notebooks\AI_ML_Task2_Model_Comparison.ipynb --output AI_ML_Task2_Model_Comparison.ipynb
```

What to check afterwards:

1. Command exits with no errors.
2. `outputs\metrics.json` has just been rewritten (timestamp updates).
3. Open the executed notebook and scroll to **cell 10 (Model performance
   comparison table)**. The displayed table must equal the CSV.
4. Cell 11 must print `→ Selected model: Decision Tree`.
5. The Actual vs Predicted figure (cell 12) must match Chart 1.

If any of that fails, the notebook is broken — regardless of what any other
document claims.

---

## 5. Recompute every metric from scratch, in Python, in ~15 lines

Paste this into any Python prompt (needs only scikit-learn). It rebuilds the
whole pipeline from raw sklearn data and prints the metrics — no files from
this project are used:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

d = fetch_california_housing(as_frame=True)
df = pd.concat([d.data, d.target.rename("HousePrice")], axis=1)
X, y = df.drop(columns=["HousePrice"]), df["HousePrice"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(alpha=1.0)),
    ("Decision Tree",    DecisionTreeRegressor(max_depth=5, random_state=42)),
]:
    model.fit(Xtr_s, ytr)
    p = model.predict(Xte_s)
    print(f"{name:<20} RMSE={root_mean_squared_error(yte, p):.4f}  R²={r2_score(yte, p):.4f}")
```

Expected output (independent of every artifact in this project):

```
Linear Regression    RMSE=0.7456  R²=0.5758
Ridge Regression     RMSE=0.7456  R²=0.5758
Decision Tree        RMSE=0.7242  R²=0.5997
```

If those numbers match the CSV, the metrics are correct.

*(If your `scikit-learn` is older than 1.4 replace
`from sklearn.metrics import root_mean_squared_error` with
`from sklearn.metrics import mean_squared_error` and use
`mean_squared_error(yte, p, squared=False)`.)*

---

## 6. Prove the saved model is genuine

Requirements: same Python environment as above.

```python
import joblib, pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the pipeline exactly as an evaluator would
pipeline = joblib.load(r"models\best_model.joblib")
print("Loaded:", pipeline)

# Feed it RAW (unscaled) inputs — the scaler is inside the pipeline
raw = fetch_california_housing(as_frame=True).data.head(5)
print(pd.DataFrame({"raw_input_row": range(5),
                    "prediction": pipeline.predict(raw)}))
```

Two things to notice:

1. The loaded object prints as `Pipeline(steps=[('scaler', StandardScaler()),
   ('model', DecisionTreeRegressor(max_depth=5, random_state=42))])` — that
   is a real sklearn Pipeline, not a placeholder.
2. It accepts raw (un-scaled) rows and returns 5 predictions — proving the
   preprocessing is bundled in and future inference cannot silently receive
   the wrong scale.

---

## What each check protects against

| Check | Catches |
| --- | --- |
| 1 (visual) | A file being out of date after a partial rebuild |
| 2 (Excel formulas) | Someone typing hard-coded metrics into cells |
| 3 (chart bindings) | Someone pasting a screenshot of a chart instead of a real chart |
| 4 (nbconvert) | A notebook that no longer executes or whose outputs were manually edited |
| 5 (independent recompute) | The whole project sharing a common bug |
| 6 (joblib load) | A saved model that would not actually work in production |

If all six pass, the deliverables are self-consistent and reproducible.

---

## Files referenced in this guide

```
Task_2\
├── outputs\
│   ├── model_comparison.csv                       ← check 1, 4
│   ├── metrics.json                               ← check 1, 4
│   ├── Task2_Predictions.xlsx                     ← check 2, 3
│   └── excel_charts\
│       ├── Task2_Predictions_rendered.pdf         ← check 1
│       ├── Chart1_ActualVsPredicted_DecisionTree.png
│       ├── Chart2_Residuals_DecisionTree.png
│       └── Chart3_RMSE_R2_Comparison.png
├── notebooks\
│   └── AI_ML_Task2_Model_Comparison.ipynb         ← check 4
├── models\
│   └── best_model.joblib                          ← check 6
├── reports\
│   ├── Task2_Report.pdf                           ← check 1
│   └── Task2_Report.md                            ← check 1
└── src\                                           (regeneration scripts — read-only reference)
```
