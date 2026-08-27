"""
Script de prediccion y demostracion del agente
------------------------------------------------
Carga el mejor modelo, genera predicciones sobre el conjunto de prueba y
muestra una explicacion/recomendacion del agente RAG para un track de ejemplo.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.models.predict_model import load_model, predict
from src.agentic.rag_engine import SpotifyRAGEngine
from src.agentic.agent import SpotifyPopularityAgent
from src.data.preprocess import ALL_FEATURES

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_PATH = PROJECT_DIR / "data" / "raw" / "spotify_data_processed.csv"


def main():
    model, name = load_model()
    print(f"Modelo cargado: {name}")

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X = test[ALL_FEATURES]
    y_pred, y_proba = predict(model, X)
    print(f"Predicciones generadas para {len(X)} tracks.")
    print("\nDistribucion de predicciones:")
    print(y_pred.value_counts(normalize=True).round(3))

    # Demostracion del agente RAG
    rag = SpotifyRAGEngine(str(RAW_PATH))
    agent = SpotifyPopularityAgent(model=model, rag_engine=rag)

    sample_idx = 0
    track = X.iloc[sample_idx].to_dict()
    pred = y_pred.iloc[sample_idx]
    prob = y_proba.iloc[sample_idx].max()

    print("\n===== EXPLICACION DEL AGENTE =====")
    print(agent.explain_prediction(track, pred, prob))
    print("\n===== RECOMENDACION DEL AGENTE =====")
    print(agent.generate_recommendation(track, pred, prob))
    print("\n===== CONSULTA RAG DE EJEMPLO =====")
    print(agent.query("Cuales generos tienen mayor popularidad?"))


if __name__ == "__main__":
    main()
