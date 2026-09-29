"""Serving : expose le modèle via une API REST."""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import FEATURES
from src.train import MODEL_PATH

app = FastAPI(title="Churn API")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Modèle introuvable : lancez d'abord l'entraînement")
        _model = joblib.load(MODEL_PATH)
    return _model


class Client(BaseModel):
    age: int = Field(ge=18, le=120)
    anciennete_mois: int = Field(ge=0)
    nb_produits: int = Field(ge=0)
    solde_moyen: float
    nb_reclamations: int = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(client: Client):
    model = get_model()
    X = pd.DataFrame([client.model_dump()])[FEATURES]
    proba = float(model.predict_proba(X)[0][1])
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}
