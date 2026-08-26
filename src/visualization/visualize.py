"""
Modulo de visualizacion
-----------------------
Funciones para graficas y reportes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def set_style():
    """Configura el estilo de las graficas."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_target_distribution(df: pd.DataFrame, target: str = 'Churn'):
    """Grafica distribucion de la variable objetivo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=df, x=target, ax=axes[0])
    axes[0].set_title(f'Distribucion de {target}')

    df[target].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title(f'{target} %')

    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, columns: list = None):
    """Grafica distribuciones de variables numericas."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f'Distribucion de {col}')

    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame):
    """Grafica matriz de correlacion."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlacion')
    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusion'):
    """Grafica matriz de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curves(y_true, models_dict: dict):
    """Grafica curvas ROC de multiples modelos."""
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Comparacion de Modelos')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_model_comparison(metrics_df: pd.DataFrame):
    """Grafica comparacion de metricas entre modelos."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    plt.title('Comparacion de Modelos')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    return fig
