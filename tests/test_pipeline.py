import pytest

from src.data import generate_data, validate_data
from src.train import train_and_evaluate


def test_donnees_valides():
    validate_data(generate_data())


def test_colonne_manquante():
    df = generate_data().drop(columns=["age"])
    with pytest.raises(ValueError):
        validate_data(df)


def test_valeurs_manquantes():
    df = generate_data()
    df.loc[0, "age"] = None
    with pytest.raises(ValueError):
        validate_data(df)


def test_modele_assez_performant():
    _, metrics = train_and_evaluate(generate_data())
    assert metrics["f1"] > 0.6, f"F1 trop faible : {metrics}"
