# AI/ML Task 2 — Feature Engineering, Model Optimization & Performance Comparison

**Program:** MainCrafts Technology — AI & ML Internship
**Dataset:** California Housing (`sklearn.datasets.fetch_california_housing`)
**Target:** `HousePrice` (median house value, in units of $100,000)

---

## Introduction

The California Housing dataset contains 20,640 census-block observations of median housing
characteristics across California. Each row provides eight numeric predictors —
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`,
`Longitude` — together with the block's **median house value**, which is renamed here to
`HousePrice` per the assignment. The task is a supervised regression problem: predict
`HousePrice` from the eight features and compare several algorithms on the same held-out
data.

## Methodology

Data acquisition first attempts to load a suitable California Housing CSV from the Task 1
workspace using a schema-validated scan (`_try_local_task1_dataset()`); no such CSV is
present in this workspace, so the notebook falls back to
`fetch_california_housing(as_frame=True)` and renames the target series to `HousePrice`.
The full frame has no missing values.

Features are separated from the target (`X = df.drop("HousePrice", axis=1)`,
`y = df["HousePrice"]`) and split with `train_test_split(..., test_size=0.2,
random_state=42)`, yielding 16,512 training rows and 4,128 test rows. A `StandardScaler`
is then fit on `X_train` only and used to transform both `X_train` and `X_test`. Fitting
the scaler on the full `X` before splitting would leak the test distribution's mean and
standard deviation into training and inflate the reported metrics; splitting first avoids
that.

Three regressors are trained on the scaled training data, and RMSE and R² are computed
on the scaled test data. All three metrics come from a single executed run — no
numerical result in this document was typed by hand.

### Feature scaling

`StandardScaler` centres and rescales each feature to zero mean and unit variance:

$$z = \frac{x - \mu}{\sigma}$$

where μ and σ are estimated from the training set. This matters most for the linear
models: `LinearRegression` and `Ridge` are affected by feature magnitudes because their
loss is written in the original coordinate system, and Ridge's L2 penalty in particular
is fair only when features share a comparable scale. Decision trees, by contrast, split
on feature thresholds and are essentially invariant to monotonic rescalings, so scaling
neither helps nor harms them; a shared preprocessing path is retained for methodological
consistency across the three models.

### Ridge regression

Ridge minimises the least-squares loss augmented with an L2 penalty on the coefficients:

$$\min_{\beta}\; \lVert y - X\beta \rVert_2^2 \;+\; \alpha \lVert \beta \rVert_2^2$$

With `alpha=1.0`, the penalty shrinks coefficients toward zero and typically reduces
variance at the cost of a small bias increase. It is a variance-reduction tool, not a
universal overfitting cure — its benefit is largest when features are correlated or when
`n` is small relative to the number of predictors.

### Models compared

- **Linear Regression** — ordinary least squares; interpretable linear baseline.
- **Ridge Regression** — the same linear model with L2 regularisation (`alpha=1.0`).
- **Decision Tree Regressor** — a nonlinear, threshold-based partitioner constrained to
  `max_depth=5`, with `random_state=42` for reproducibility.

## Results

The comparison table below is read directly from `outputs/model_comparison.csv`, which is
written by the notebook at run time.

| Model | Test RMSE | Test R² | Rank |
|---|---:|---:|:---:|
| Linear Regression | 0.7456 | 0.5758 | 3 |
| Ridge Regression | 0.7456 | 0.5758 | 2 |
| **Decision Tree (max_depth=5)** | **0.7242** | **0.5997** | **1** |

RMSE is expressed in the same units as `HousePrice` — hundreds of thousands of dollars —
so a value of 0.72 corresponds to a typical error of about $72,000. R² is the share of
variance in `HousePrice` explained by the model on the test set. Larger R² and smaller
RMSE both indicate better fit.

The two linear models are practically indistinguishable: at full precision they differ
only in the fifth decimal (RMSE 0.745581 vs 0.745557). This is expected — the eight
California Housing features are only weakly collinear after standardisation, so L2
shrinkage has little room to help. The Decision Tree is the strongest of the three by a
modest but consistent margin: about a 2.9% RMSE reduction and a 2.4-percentage-point R²
gain over Linear Regression, coming from a handful of nonlinear splits (notably on
`MedInc`, `Latitude`, and `Longitude`) that the linear models cannot represent.

Model selection is performed programmatically inside the notebook by sorting the results
by RMSE ascending, then R² descending; the winner is `ranked.index[0]`. No model name is
hard-coded.

## Visualisation

The Actual vs Predicted plot (`outputs/figures/actual_vs_predicted_decision_tree.png`)
uses the winning model on the test set. A depth-5 tree produces at most 32 distinct
predicted values (2⁵ leaves), so the scatter shows clear horizontal banding —
predictions collapse onto the 32 leaf means, while the actual values vary continuously
along each band. The main systematic deviation is at the upper end: the target is capped
at 5.0 ($500,000), and the tree systematically **under-predicts** for that ceiling
group, so points at the top-right of the plot sit below the red y = x reference line.
Near the median of the target range, points cluster closer to the diagonal.

The residual plot (`outputs/figures/residuals_decision_tree.png`) shows residuals
(*actual − predicted*) with mean −0.0019 and standard deviation 0.724 on the test set.
The residual cloud is roughly centred on zero for low-to-mid predictions and fans out
for higher predictions, reflecting both the leaf discretisation and the ceiling effect.
There is no strong evidence of a systematic sign bias, but the visible fan-shape
indicates heteroscedastic errors that a deeper model or an ensemble would likely reduce.

## Conclusion

Among the three evaluated models under the specified experimental setup — an 80/20
hold-out with `random_state=42`, `StandardScaler` fit on training data only, and
hyperparameters fixed to the assignment values — the **Decision Tree Regressor
(`max_depth=5`, `random_state=42`)** was selected as the best-performing model, with a
test RMSE of **0.7242** and a test R² of **0.5997**. It was chosen by the programmatic
selection rule (minimum RMSE; R² as tiebreaker) applied to the executed results table.

The comparison illustrates a familiar pattern: on this dataset, additional linear
regularisation contributes almost nothing over ordinary least squares, whereas even a
shallow nonlinear model captures enough of the geographic and income structure to move
both metrics meaningfully. The result should not be read as a universal claim that
decision trees dominate linear models — only that, at these hyperparameters and on this
split, the tree provides the best fit of the three.

## Limitations and next steps

Evaluation relies on a single 80/20 hold-out rather than cross-validation, and the
three algorithms are all evaluated at fixed hyperparameters with no search. Only RMSE
and R² are reported; a full study would add MAE and calibration diagnostics. Natural
extensions: relax `max_depth`, add an ensemble (Random Forest or Gradient Boosting), or
introduce domain-motivated engineered features such as distance to major coastal
cities.

---

**Artifacts produced by the executed notebook**

- `outputs/model_comparison.csv` — the comparison table above (machine-readable).
- `outputs/metrics.json` — full-precision metrics, split sizes, and configuration.
- `outputs/figures/actual_vs_predicted_decision_tree.png` — winner's calibration plot.
- `outputs/figures/residuals_decision_tree.png` — winner's residual diagnostics.
- `outputs/figures/rmse_r2_comparison.png` — side-by-side RMSE and R² bars.
- `models/best_model.joblib` — winning model bundled with the fitted scaler as a
  `sklearn.pipeline.Pipeline`, so downstream inference cannot receive unscaled inputs.
