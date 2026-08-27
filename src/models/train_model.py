"""
Modulo de entrenamiento de modelos
-----------------------------------
Entrena y evalua modelos de clasificacion multiclase para predecir la
categoria de popularidad de los tracks de Spotify. Registra experimentos
en MLflow (local o DagsHub si se configuran las variables de entorno).
"""
from __future__ import annotations

import os
import pickle
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import mlflow
import mlflow.sklearn

from src.data.preprocess import make_pipeline, ALL_FEATURES

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Configuracion de MLflow: por defecto local; si hay credenciales de DagsHub
# se usa el tracking remoto.
DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "mle-project2-spotify-popularity")

if DAGSHUB_TOKEN and DAGSHUB_USER:
    import dagshub

    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, token=DAGSHUB_TOKEN)
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    )
else:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR.as_posix()}/mlflow.db")

EXPERIMENT_NAME = "spotify_popularity_classification"
mlflow.set_experiment(EXPERIMENT_NAME)

CLASS_LABELS = ["Low", "Medium", "High"]


def get_models() -> dict:
    """Retorna diccionario con los modelos a evaluar."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=50,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evalua un modelo y retorna metricas multiclase."""
    y_pred = model.predict(X_test)
    y_proba_raw = model.predict_proba(X_test)
    y_test_str = np.asarray(y_test, dtype=str)

    # Alinear columnas de probabilidad al orden canonico de clases.
    classes = list(model.classes_)
    order = [classes.index(c) for c in CLASS_LABELS]
    y_proba = y_proba_raw[:, order]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    try:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_test_str, classes=CLASS_LABELS)
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_bin, y_proba, multi_class="ovr", average="macro"
        )
    except Exception as e:
        metrics["roc_auc_ovr"] = float("nan")

    return metrics, y_pred, y_proba


def train_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """Entrena todos los modelos y retorna resultados + mejores metricas."""
    models = get_models()
    results = {}
    best_name = None
    best_f1 = -1.0

    # Linea base (predictor mayoritario) para contextualizar las metricas.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    d_pred = dummy.predict(X_test)
    d_proba = np.full((len(y_test), len(CLASS_LABELS)), 1 / len(CLASS_LABELS))
    results["baseline_majority"] = {
        "model": dummy,
        "metrics": {
            "accuracy": accuracy_score(y_test, d_pred),
            "precision_macro": precision_score(y_test, d_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, d_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, d_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_test, d_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, d_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, d_pred, average="weighted", zero_division=0),
        },
        "y_pred": d_pred,
        "y_proba": d_proba,
    }

    for name, model in models.items():
        print(f"Entrenando {name} ...")
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)

        results[name] = {
            "model": pipeline,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name

    results["best_model_name"] = best_name
    return results


def log_to_mlflow(name: str, model, metrics: dict, params: dict):
    """Registra un experimento en MLflow (metricas, parametros y artefactos)."""
    clean_metrics = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)
    }
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(clean_metrics)
        mlflow.sklearn.log_model(model, "model")
        return run.info.run_id


def save_best_model(model, name: str):
    """Guarda el mejor modelo en models/best_model.pkl (comprimido)."""
    path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"model": model, "name": name}, path, compress=3)
    print(f"Mejor modelo ({name}) guardado en {path}")
    return path
