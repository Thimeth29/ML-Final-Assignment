import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import mlflow
import mlflow.sklearn


def metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="data/processed/train.csv")
    parser.add_argument("--model_out", required=True, help="models/best_model.pkl")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' in processed train.csv")

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # Train 3 models
    candidates = [
        ("logreg", LogisticRegression(max_iter=2000, random_state=args.random_state)),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=args.random_state)),
        ("gb", GradientBoostingClassifier(random_state=args.random_state)),
    ]

    mlflow.set_experiment(args.experiment)

    best_name, best_model, best_auc, best_run = None, None, -1.0, None

    for name, model in candidates:
        with mlflow.start_run(run_name=name) as run:
            model.fit(X, y)

            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

            m = metrics(y, y_pred, y_proba)

            # Log params
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", args.random_state)
            if name == "rf":
                mlflow.log_param("n_estimators", 400)

            # Log metrics
            for k, v in m.items():
                mlflow.log_metric(k, v)

            # Log model artifact
            mlflow.sklearn.log_model(model, "model")

            print(f"✅ {name} ROC-AUC: {m['roc_auc']:.4f}")

            if m["roc_auc"] > best_auc:
                best_auc = m["roc_auc"]
                best_name = name
                best_model = model
                best_run = run.info.run_id

    # Save best model
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)

    Path("models").mkdir(exist_ok=True)
    Path("models/best_run.txt").write_text(best_run or "", encoding="utf-8")

    print(f"\n🏆 Best model: {best_name} | ROC-AUC: {best_auc:.4f}")
    print(f"✅ Saved best model to: {out_path}")
    print("✅ Saved models/best_run.txt")


if __name__ == "__main__":
    main()