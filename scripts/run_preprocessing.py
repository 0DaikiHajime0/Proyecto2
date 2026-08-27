"""
Script de preprocesamiento
---------------------------
Limpia, aplica ingenieria de features y divide el dataset de Spotify en
train/test estratificado por la categoria de popularidad.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocess import (
    load_data,
    clean_data,
    split_data,
)
from src.features.build_features import build_features

RAW_PATH = PROJECT_DIR / "data" / "raw" / "spotify_data_processed.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Cargando dataset crudo...")
    df = load_data(str(RAW_PATH))
    print(f"  Registros crudos: {len(df)}")

    df = clean_data(df)
    print(f"  Tras limpieza: {len(df)}")

    df = build_features(df)
    print("  Features derivadas anadidas.")

    X_train, X_test, y_train, y_test = split_data(df)
    train = X_train.copy()
    train["popularity_category"] = y_train.values
    test = X_test.copy()
    test["popularity_category"] = y_test.values

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Guardado train.csv ({train.shape}) y test.csv ({test.shape}) en {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
