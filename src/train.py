"""train.py

PART 2 – Model Development:
Train at least 3 models:
- Logistic Regression
- Random Forest
- XGBoost (or Gradient Boosting)

Log all experiments to MLflow:
- parameters
- metrics
- confusion matrix / ROC curve artifacts (you can log from evaluate.py too)
- model artifacts

Output:
- best model saved to models/best_model.pkl

This is a template. Replace TODO sections with your implementation.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd

# TODO: import sklearn models, xgboost, pipelines, encoders, scalers, etc.
# TODO: import mlflow and mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Processed train CSV path")
    parser.add_argument("--model_out", required=True, help="Output model path (.pkl)")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    args = parser.parse_args()

    train_path = Path(args.train)
    df = pd.read_csv(train_path)

    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in processed train.csv")

    # TODO 1: Prepare X, y (convert y to 0/1)
    # y = (df["Churn"] == "Yes").astype(int)
    # X = df.drop(columns=["Churn"])

    # TODO 2: Build 3 models + training pipeline(s)
    # TODO 3: Train + evaluate (on a validation split or cross-val)
    # TODO 4: MLflow logging for each model run
    # TODO 5: Choose best model based on ROC-AUC (recommended)

    # Placeholder: save nothing until implemented
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: replace with actual best_model
    best_model = {"TODO": "Replace with trained model"}
    joblib.dump(best_model, out_path)

    print(f"✅ Saved best model to: {out_path}")
    print("⚠ NOTE: This is a template. Implement training + MLflow logging.")

if __name__ == "__main__":
    main()
