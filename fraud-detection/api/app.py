from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from pathlib import Path
import joblib

# fraud-detection/api/app.py
BASE_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = BASE_DIR / "rf_fraud_model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection Inference API")

# ---------- Schemas ----------
class PredictionRequest(BaseModel):
    features: List[List[float]]   # batch input
    threshold: float = 0.5

class PredictionResponse(BaseModel):
    fraud_probability: List[float]
    prediction: List[int]

# ---------- Health check ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Inference ----------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    X = np.array(req.features)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= req.threshold).astype(int)

    return {
        "fraud_probability": probs.tolist(),
        "prediction": preds.tolist()
    }
