import argparse
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_and_log_model(model, name, X_train, y_train, experiment_id):
    with mlflow.start_run(run_name=name, experiment_id=experiment_id, nested=True):
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
        
        # Log training ROC-AUC (simple proxy for model health)
        y_proba = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_proba)
        mlflow.log_metric("train_roc_auc", auc)
        
        # Log model
        if "XGB" in name:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
            
        print(f"{name} trained. Train ROC-AUC: {auc:.4f}")
        return auc

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

    y_train = df["Churn"].astype(int)
    X_train = df.drop(columns=["Churn"])

    # Setup MLflow
    mlflow.set_experiment(args.experiment)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_score = -1
    best_model = None
    best_name = ""

    with mlflow.start_run(run_name="ModelSelection"):
        for name, model in models.items():
            score = train_and_log_model(model, name, X_train, y_train, experiment.experiment_id)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        # Log selection info
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_metric("best_train_auc", best_score)
        
        # Final save
        out_path = Path(args.model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, out_path)
        
        print(f"\nBest Model: {best_name} with Train ROC-AUC: {best_score:.4f}")
        print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
