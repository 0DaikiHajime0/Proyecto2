"""
Modulo de entrenamiento de modelos
-----------------------------------
Entrena y evalua modelos de clasificacion para Churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn


def get_models() -> dict:
    """Retorna diccionario con modelos a evaluar."""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evalua un modelo y retorna metricas."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """Entrena todos los modelos y retorna resultados."""
    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {'model': model, 'metrics': metrics}

    return results


def log_to_mlflow(model, metrics: dict, params: dict, run_name: str):
    """Registra un experimento en MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model')
