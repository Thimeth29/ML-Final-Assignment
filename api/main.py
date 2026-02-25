"""FastAPI prediction service (template)

Requirements:
- POST /predict
- Return:
  {
    "churn_probability": 0.82,
    "prediction": "Yes"
  }

TODO:
- Define the input schema based on your processed features
- Load the trained model from models/best_model.pkl
- Run prediction and return response
"""

from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
model = None

class PredictRequest(BaseModel):
    # TODO: Replace with real feature fields.
    # Example:
    # tenure: int
    # MonthlyCharges: float
    placeholder: float = 0.0

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not found. Train the model and ensure models/best_model.pkl exists."}

    # TODO: Convert req -> dataframe/array with correct feature order
    # TODO: proba = model.predict_proba(X)[0, 1]
    # TODO: pred = "Yes" if proba >= 0.5 else "No"
    return {"churn_probability": 0.0, "prediction": "No", "note": "Template - implement feature mapping + prediction"}
