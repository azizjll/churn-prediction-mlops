"""Pipeline d'entraînement : validation -> entraînement -> évaluation -> porte de qualité."""
import json
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET, generate_data, validate_data

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

PARAMS = {"n_estimators": 200, "max_depth": 8, "random_state": 42}


def train_and_evaluate(df):
    """Entraîne le modèle et retourne (modèle, métriques)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    return model, metrics


def get_production_f1() -> float:
    """F1 du modèle actuellement en production (0 s'il n'y en a pas encore)."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())["f1"]
    return 0.0


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")

    # 1. Ingestion + 2. Validation
    df = generate_data()
    validate_data(df)
    print(f"[1-2] Données OK : {len(df)} lignes")

    with mlflow.start_run():
        # 3. Entraînement + 4. Évaluation
        model, metrics = train_and_evaluate(df)
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        print(f"[3-4] Nouveau modèle : {metrics}")

        # 5. Porte de qualité : on ne remplace que si c'est meilleur
        prod_f1 = get_production_f1()
        print(f"[5]   F1 production : {prod_f1}")
        if metrics["f1"] > prod_f1:
            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            METRICS_PATH.write_text(json.dumps(metrics))
            mlflow.set_tag("decision", "promoted")
            print("[5]   Nouveau modèle ACCEPTÉ et enregistré")
        else:
            mlflow.set_tag("decision", "rejected")
            print("[5]   Nouveau modèle REFUSÉ, l'ancien est conservé")


if __name__ == "__main__":
    main()
