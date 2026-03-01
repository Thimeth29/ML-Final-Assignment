import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_confusion_matrix(y_true, y_pred, out_path: Path):
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
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_proba, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="data/processed/train.csv")
    parser.add_argument("--model_out", required=True, help="models/best_model.pkl")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column in processed train.csv")

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # ✅ Validation split (so evaluation is meaningful)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y
    )

    candidates = [
        ("logreg", LogisticRegression(max_iter=2000, random_state=args.random_state)),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=args.random_state)),
        ("gb", GradientBoostingClassifier(random_state=args.random_state)),
        # If you want XGBoost later: add XGBClassifier here
    ]

    mlflow.set_experiment(args.experiment)

    best_name, best_model, best_auc, best_run_id = None, None, -1.0, None

    artifacts_dir = Path("reports/mlflow_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, model in candidates:
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

            m = compute_metrics(y_val, y_pred, y_proba)

            # ✅ Log parameters
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("val_size", args.val_size)
            if name == "rf":
                mlflow.log_param("n_estimators", 400)

            # ✅ Log metrics
            for k, v in m.items():
                mlflow.log_metric(k, v)

            # ✅ Save & log artifacts (confusion matrix + ROC curve)
            cm_path = artifacts_dir / f"{name}_confusion_matrix.png"
            roc_path = artifacts_dir / f"{name}_roc_curve.png"
            save_confusion_matrix(y_val, y_pred, cm_path)
            save_roc_curve(y_val, y_proba, roc_path)

            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            # ✅ Log model artifact
            mlflow.sklearn.log_model(model, name="model")

            # Track best
            if m["roc_auc"] > best_auc:
                best_auc = m["roc_auc"]
                best_name = name
                best_model = model
                best_run_id = run.info.run_id

            print(f"✅ {name} VAL ROC-AUC: {m['roc_auc']:.4f}")

    # Save best model locally
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)

    Path("models").mkdir(exist_ok=True)
    Path("models/best_run.txt").write_text(best_run_id or "", encoding="utf-8")
    Path("models/best_model_name.txt").write_text(best_name or "", encoding="utf-8")
    Path("models/best_val_roc_auc.txt").write_text(str(best_auc), encoding="utf-8")

    print(f"\n🏆 Best model: {best_name} | VAL ROC-AUC: {best_auc:.4f}")
    print(f"✅ Saved best model to: {out_path}")
    print("✅ Logged params, metrics, model, confusion matrix, ROC curve to MLflow")


if __name__ == "__main__":
    main()