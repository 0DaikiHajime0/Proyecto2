"""
Modulo de predicciones
----------------------
Carga modelo entrenado y genera predicciones.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def load_model(model_path: str):
    """Carga un modelo serializado."""
    return joblib.load(model_path)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Genera predicciones de clase."""
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Genera probabilidades de prediccion."""
    return model.predict_proba(X)[:, 1]


def create_submission(predictions: np.ndarray, customer_ids: pd.Series) -> pd.DataFrame:
    """Crea DataFrame de submission."""
    return pd.DataFrame({
        'customerID': customer_ids,
        'Churn_Prediction': predictions
    })
