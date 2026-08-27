"""
Modulo de visualizacion
------------------------
Genera las figuras requeridas por el README a partir de los datos y los
resultados de los modelos. Usa matplotlib (sin dependencias externas).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["Low", "Medium", "High"]
COLORS = {"Low": "#d73027", "Medium": "#fc8d59", "High": "#1a9850"}


def plot_target_distribution(y: pd.Series, path: Path = None):
    """Distribucion de la variable objetivo (categoria de popularidad)."""
    path = path or (FIG_DIR / "popularity_distribution.png")
    counts = y.value_counts().reindex(CLASS_LABELS).fillna(0)
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    total = counts.sum()
    for i, v in enumerate(counts.values):
        plt.text(i, v + total * 0.01, f"{v}\n({v/total:.1%})", ha="center", va="bottom")
    plt.title("Distribucion de la categoria de popularidad")
    plt.ylabel("Numero de tracks")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_numeric_vs_target(df: pd.DataFrame, numeric_cols: list, path: Path = None):
    """Boxplots de features numericas por categoria de popularidad."""
    path = path or (FIG_DIR / "numeric_feature_analysis.png")
    n = len(numeric_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, numeric_cols):
        data = [df[df["popularity_category"] == c][col].dropna() for c in CLASS_LABELS]
        ax.boxplot(data, tick_labels=CLASS_LABELS)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Features numericas vs categoria de popularidad", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_categorical_vs_target(df: pd.DataFrame, cat_cols: list, path: Path = None):
    """Proporcion de cada clase de popularidad por valor categorico."""
    path = path or (FIG_DIR / "categorical_analysis.png")
    n = len(cat_cols)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, cat_cols):
        ct = pd.crosstab(df[col], df["popularity_category"])
        ct = ct.reindex(columns=CLASS_LABELS, fill_value=0)
        ct = ct.div(ct.sum(axis=1), axis=0)
        ct.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in CLASS_LABELS])
        ax.set_title(col)
        ax.set_ylabel("Proporcion")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Popularidad")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_correlation(df: pd.DataFrame, numeric_cols: list, path: Path = None):
    """Matriz de correlacion de variables numericas."""
    path = path or (FIG_DIR / "correlation_matrix.png")
    corr = df[numeric_cols + ["popularity"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de correlacion (numericas)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_model_comparison(results: dict, path: Path = None):
    """Comparacion de modelos por F1-macro y Accuracy."""
    path = path or (FIG_DIR / "model_comparison.png")
    names = [k for k in results if k != "best_model_name"]
    f1 = [results[n]["metrics"]["f1_macro"] for n in names]
    acc = [results[n]["metrics"]["accuracy"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, acc, width, label="Accuracy")
    plt.bar(x + width / 2, f1, width, label="F1-macro")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Comparacion de modelos")
    plt.legend()
    for i, (a, f) in enumerate(zip(acc, f1)):
        plt.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center")
        plt.text(i + width / 2, f + 0.01, f"{f:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_roc_curves(y_test, results: dict, path: Path = None):
    """Curvas ROC one-vs-rest por clase para cada modelo."""
    path = path or (FIG_DIR / "roc_curves.png")
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    classes = CLASS_LABELS
    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8, 6))
    for name in results:
        if name == "best_model_name":
            continue
        y_proba = results[name]["y_proba"]
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            from sklearn.metrics import auc

            plt.plot(fpr, tpr, label=f"{name} - {c} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path: Path = None):
    """Matriz de confusion del mejor modelo."""
    path = path or (FIG_DIR / "confusion_matrix.png")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de confusion (mejor modelo)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names: list, path: Path = None, top_n: int = 20):
    """Importancia de features (basado en el modelo de la pipeline)."""
    path = path or (FIG_DIR / "feature_importance.png")
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).mean(axis=0)
    else:
        return
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = importances[order]
    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    plt.barh(range(len(names)), vals, color="#4575b4")
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importancia")
    plt.title("Top features mas importantes")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
