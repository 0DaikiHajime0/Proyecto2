"""
Modulo de evaluacion
---------------------
Utilidades para calcular y reportar metricas de clasificacion multiclase.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

CLASS_LABELS = ["Low", "Medium", "High"]


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula un diccionario de metricas multiclase."""
    y_true_str = np.asarray(y_true, dtype=str)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        try:
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_true_str, classes=CLASS_LABELS)
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")
    return metrics


def print_classification_report(y_true, y_pred):
    """Imprime el reporte de clasificacion por clase."""
    print(classification_report(y_true, y_pred, labels=CLASS_LABELS, zero_division=0))


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Devuelve la matriz de confusion alineada a CLASS_LABELS."""
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
