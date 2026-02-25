"""evaluate.py

Evaluate the saved best model on the processed test set.
Must compute:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Must save artifacts:
- confusion matrix image
- ROC curve image
- metrics.json

This is a template. Replace TODO sections with your implementation.
"""

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

# TODO: import sklearn metrics, matplotlib for plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Processed test CSV path")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--outdir", required=True, help="Output directory for metrics and plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in processed test.csv")

    # TODO: X_test, y_test
    # y_test = (df["Churn"] == "Yes").astype(int)
    # X_test = df.drop(columns=["Churn"])

    model = joblib.load(args.model)

    # TODO: get predictions + probabilities
    # y_pred = model.predict(X_test)
    # y_proba = model.predict_proba(X_test)[:, 1]

    # TODO: compute metrics dict
    metrics = {
        "TODO": "Replace with real metrics",
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # TODO: save confusion_matrix.png and roc_curve.png to outdir
    print(f"✅ Saved metrics to: {outdir / 'metrics.json'}")
    print("⚠ NOTE: This is a template. Implement evaluation + plots.")

if __name__ == "__main__":
    main()
