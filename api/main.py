from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Churn Prediction API", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

model = None
preprocessor = None


class PredictRequest(BaseModel):
    # send raw fields as key/value pairs
    features: Dict[str, Any]


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    # Create a 1-row DataFrame from raw input
    X_raw = pd.DataFrame([req.features])

    # Transform using saved preprocessor
    X = preprocessor.transform(X_raw)

    # Predict probability of churn = 1
    proba = float(model.predict_proba(X)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"

    return {
        "churn_probability": round(proba, 4),
        "prediction": prediction,
    }