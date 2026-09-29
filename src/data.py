"""Étape 1 & 2 : ingestion (données simulées) et validation des données."""
import numpy as np
import pandas as pd

FEATURES = ["age", "anciennete_mois", "nb_produits", "solde_moyen", "nb_reclamations"]
TARGET = "churn"


def generate_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Simule des clients d'une banque. En vrai : lecture SQL / fichier / API."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "anciennete_mois": rng.integers(1, 120, n),
        "nb_produits": rng.integers(1, 6, n),
        "solde_moyen": rng.normal(3000, 1500, n).round(2),
        "nb_reclamations": rng.poisson(1, n),
    })
    score = (
        0.8 * df["nb_reclamations"]
        - 0.02 * df["anciennete_mois"]
        - 0.3 * df["nb_produits"]
        - 0.0003 * df["solde_moyen"]
        + rng.normal(0, 0.5, n)
    )
    df[TARGET] = (score > np.percentile(score, 75)).astype(int)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Lève une erreur si les données sont invalides -> le pipeline s'arrête."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if df.isnull().any().any():
        raise ValueError("Valeurs manquantes détectées")
    if len(df) < 500:
        raise ValueError(f"Pas assez de données : {len(df)} lignes")
    if not df["age"].between(18, 120).all():
        raise ValueError("Âges aberrants détectés")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("La cible doit être 0 ou 1")
