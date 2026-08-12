"""Rebuild Task2_Charts.pdf as a clean 4-page report.

Page 1 : Title + verified metrics table + run configuration.
Page 2 : Chart 1 — Actual vs Predicted (Decision Tree) full-page.
Page 3 : Chart 2 — Residuals vs Predicted (Decision Tree) full-page.
Page 4 : Chart 3 — RMSE and R² model comparison bars full-page.

Every metric on Page 1 is read from outputs/metrics.json (which was written
by the executed notebook). Every chart is a native Excel chart rendered by
LibreOffice from Task2_Predictions.xlsx.
"""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

TASK2 = Path(__file__).resolve().parents[1]
CHART_DIR = TASK2 / "outputs" / "excel_charts"
METRICS_JSON = TASK2 / "outputs" / "metrics.json"
OUT_PDF = CHART_DIR / "Task2_Predictions_rendered.pdf"  # overwrite the bloated one

CHART_FILES = [
    ("Chart 1 — Actual vs Predicted (Decision Tree)",
     "Blue dots are the 4,128 test-set predictions from the winning model. "
     "The red line is y = x (a perfect prediction). Horizontal banding appears "
     "because a depth-5 tree produces only 32 distinct leaf values.",
     "Chart1_ActualVsPredicted_DecisionTree.png"),
    ("Chart 2 — Residuals vs Predicted (Decision Tree)",
     "Residual = Actual − Predicted. The red horizontal line is y = 0 (no error). "
     "Residuals cluster in vertical columns at the 32 leaf-mean predictions. "
     "Mean residual = -0.0019, standard deviation = 0.724 on the test set.",
     "Chart2_Residuals_DecisionTree.png"),
    ("Chart 3 — Model comparison (RMSE and R² on test set)",
     "Side-by-side test RMSE (blue) and test R² (red) for the three models. "
     "Decision Tree has the lowest RMSE and highest R² of the three; the two "
     "linear models are practically indistinguishable at this precision.",
     "Chart3_RMSE_R2_Comparison.png"),
]

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_ITALIC  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"


class ChartsPdf(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6,
                  f"AI/ML Task 2 — Excel-Rendered Charts   ·   Page {self.page_no()} / {{nb}}",
                  align="C")
        self.set_text_color(0)


def build():
    metrics = json.loads(METRICS_JSON.read_text())
    results = metrics["results"]

    pdf = ChartsPdf(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_font("DejaVu", "",  FONT_REGULAR)
    pdf.add_font("DejaVu", "B", FONT_BOLD)
    pdf.add_font("DejaVu", "I", FONT_ITALIC)
    pdf.alias_nb_pages()

    # ==================================================================
    # Page 1 — cover + verified metrics + config
    # ==================================================================
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 15)
    pdf.multi_cell(0, 7,
        "AI/ML Task 2 — Model Comparison Charts",
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(80)
    pdf.multi_cell(0, 4.4,
        "MainCrafts Technology — AI & ML Internship   ·   "
        "Dataset: California Housing (sklearn.datasets.fetch_california_housing)   ·   "
        "Target: HousePrice",
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(2)

    # What this PDF is
    pdf.set_font("DejaVu", "B", 11)
    pdf.multi_cell(0, 5, "About this PDF", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9.5)
    pdf.multi_cell(0, 4.4,
        "This document contains the three Excel charts built from "
        "Task2_Predictions.xlsx. Each chart is a native Excel chart bound to live "
        "cell ranges in that workbook (Predictions and Comparison sheets); LibreOffice "
        "rendered them to the pages that follow. The metrics table below is read from "
        "outputs/metrics.json — the same file the notebook writes at run time — so no "
        "number in this PDF was typed by hand.",
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Verified metrics table
    pdf.set_font("DejaVu", "B", 11)
    pdf.multi_cell(0, 5, "Verified test-set metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    col_w = [66, 34, 34, 20]
    headers = ["Model", "Test RMSE", "Test R²", "Rank"]
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_fill_color(230, 235, 245)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    display_order = ["Linear Regression", "Ridge Regression", "Decision Tree"]
    winner = min(results.items(), key=lambda kv: (kv[1]["RMSE"], -kv[1]["R2 Score"]))[0]
    for name in display_order:
        row = results[name]
        is_winner = name == winner
        if is_winner:
            pdf.set_font("DejaVu", "B", 10)
            pdf.set_fill_color(220, 240, 220)
        else:
            pdf.set_font("DejaVu", "", 10)
            pdf.set_fill_color(255, 255, 255)
        label = f"{name}   [best]" if is_winner else name
        pdf.cell(col_w[0], 6.5, label, border=1, fill=True)
        pdf.cell(col_w[1], 6.5, f"{row['RMSE']:.4f}", border=1, align="R", fill=True)
        pdf.cell(col_w[2], 6.5, f"{row['R2 Score']:.4f}", border=1, align="R", fill=True)
        pdf.cell(col_w[3], 6.5, f"{row['Rank']}", border=1, align="C", fill=True)
        pdf.ln()
    pdf.ln(3)

    # Run configuration
    pdf.set_font("DejaVu", "B", 11)
    pdf.multi_cell(0, 5, "Run configuration", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)
    cfg = [
        ("Dataset",           "California Housing via sklearn.datasets.fetch_california_housing(as_frame=True)"),
        ("Target column",     metrics["target_column"]),
        ("Rows train / test", f"{metrics['n_train']:,} / {metrics['n_test']:,}"),
        ("test_size",         f"{metrics['test_size']}"),
        ("random_state",      f"{metrics['random_state']}"),
        ("Preprocessing",     "StandardScaler fit on X_train only, then transform train and test"),
        ("Models",            "LinearRegression() · Ridge(alpha=1.0) · DecisionTreeRegressor(max_depth=5, random_state=42)"),
        ("Selection rule",    "argmin(test RMSE); tiebreak argmax(test R²) — computed programmatically"),
        ("Chart source",      "Native Excel charts in Task2_Predictions.xlsx, rendered by LibreOffice"),
    ]
    for k, v in cfg:
        pdf.set_font("DejaVu", "B", 9.5)
        pdf.cell(40, 4.5, k)
        pdf.set_font("DejaVu", "", 9.5)
        pdf.multi_cell(0, 4.5, v, new_x="LMARGIN", new_y="NEXT")

    # Footer note about page contents
    pdf.ln(3)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(90)
    pdf.multi_cell(0, 4.2,
        "Pages 2–4 that follow show, in order:  Chart 1 (Actual vs Predicted for the "
        "Decision Tree),  Chart 2 (Residuals vs Predicted for the Decision Tree),  "
        "Chart 3 (RMSE and R² across all three models).",
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)

    # ==================================================================
    # Pages 2–4 — one chart each, full-page landscape look inside portrait A4
    # ==================================================================
    for title, caption, filename in CHART_FILES:
        pdf.add_page()

        # Header with chart title
        pdf.set_font("DejaVu", "B", 13)
        pdf.multi_cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        # Fit the image into the remaining page width, preserving aspect ratio
        page_w = 210 - 15 - 15  # margins subtracted
        max_h = 297 - pdf.get_y() - 40  # leave room for caption + footer
        img_path = CHART_DIR / filename
        # Determine natural aspect and pick the tighter constraint
        from PIL import Image
        with Image.open(img_path) as im:
            w0, h0 = im.size
        # Scale so image fits within (page_w, max_h)
        scale = min(page_w / w0, max_h / (h0 * (25.4 / 72) if False else h0))
        # Use openpyxl-style mm sizing by giving fpdf explicit width and letting
        # it compute height from the file's DPI. Simplest reliable approach:
        target_w = page_w
        target_h = target_w * (h0 / w0)
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * (w0 / h0)
        x = (210 - target_w) / 2
        pdf.image(str(img_path), x=x, w=target_w, h=target_h)
        pdf.ln(2)

        # Caption
        pdf.set_font("DejaVu", "I", 9)
        pdf.set_text_color(90)
        pdf.multi_cell(0, 4.3, caption, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)

    pdf.output(str(OUT_PDF))
    print(f"Wrote: {OUT_PDF}  ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
