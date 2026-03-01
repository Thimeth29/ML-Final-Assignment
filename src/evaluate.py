import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="data/processed/test.csv")
    parser.add_argument("--model", required=True, help="models/best_model.pkl")
    parser.add_argument("--outdir", required=True, help="reports/")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test)
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' in processed test.csv")

    y_true = df["Churn"].astype(int)
    X_test = df.drop(columns=["Churn"])

    model = joblib.load(args.model)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    print("Saved reports/metrics.json")
    print("Saved reports/confusion_matrix.png")
    print("Saved reports/roc_curve.png")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()