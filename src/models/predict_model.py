"""
Modulo de prediccion
---------------------
Carga el modelo entrenado y genera predicciones para nuevos tracks,
devolviendo tambien las probabilidades por clase.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"
CLASS_LABELS = ["Low", "Medium", "High"]


def load_model(path: str = None):
    """Carga el modelo guardado en models/best_model.pkl."""
    path = Path(path) if path else MODEL_PATH
    bundle = joblib.load(path)
    return bundle["model"], bundle["name"]


def predict(model, X: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Genera predicciones y probabilidades por clase."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    proba_df = pd.DataFrame(y_proba, columns=model.classes_, index=X.index)
    return pd.Series(y_pred, index=X.index, name="predicted_popularity"), proba_df
