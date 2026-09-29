# Mini-projet MLOps : prédiction de churn

## Lancer en local
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v                 # tests
python -m src.train              # pipeline d'entraînement (relancez-le 2 fois !)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000 -> historique des essais
uvicorn src.serve:app --reload   # API sur http://localhost:8000/docs
```

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Test de l'API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":45,"anciennete_mois":6,"nb_produits":1,"solde_moyen":500,"nb_reclamations":4}'
```
