"""
Script de entrenamiento
------------------------
Entrena modelos de clasificacion multiclase, evalua, registra experimentos
en MLflow, guarda el mejor modelo y genera las figuras del reporte.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv

load_dotenv(PROJECT_DIR / ".env")

import matplotlib

matplotlib.use("Agg")

from src.data.preprocess import ALL_FEATURES, get_feature_names, build_preprocessor
from src.models.train_model import (
    train_and_evaluate,
    log_to_mlflow,
    save_best_model,
    register_model_version,
)
from src.models.evaluate import get_confusion_matrix
from src.visualization import visualize as viz

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "reports" / "figures"


def main():
    train = __import__("pandas").read_csv(PROCESSED_DIR / "train.csv")
    test = __import__("pandas").read_csv(PROCESSED_DIR / "test.csv")

    X_train = train[ALL_FEATURES]
    y_train = train["popularity_category"]
    X_test = test[ALL_FEATURES]
    y_test = test["popularity_category"]

    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    best_name = results["best_model_name"]
    best = results[best_name]
    print(f"\nMejor modelo: {best_name} | F1-macro: {best['metrics']['f1_macro']:.4f}")

    # Registrar cada modelo en MLflow
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        if name not in results:
            continue
        r = results[name]
        print(f"Registrando {name} en MLflow...")
        run_id = log_to_mlflow(
            name,
            r["model"],
            params={"model_type": name, "features": len(ALL_FEATURES), "random_state": 42},
            metrics=r["metrics"],
        )
        results[name]["mlflow_run_id"] = run_id

    # Registrar el mejor modelo como productivo en el Model Registry
    best_run_id = results[best_name]["mlflow_run_id"]
    register_model_version(best_run_id, "spotify_popularity_model")

    # Guardar mejor modelo
    save_best_model(best["model"], best_name)

    # Figuras
    print("Generando figuras...")
    viz.plot_target_distribution(y_test, FIG_DIR / "popularity_distribution.png")
    viz.plot_numeric_vs_target(
        test, ["danceability", "energy", "loudness", "tempo", "duration_ms", "artist_track_count"],
        FIG_DIR / "numeric_feature_analysis.png",
    )
    viz.plot_categorical_vs_target(
        test, ["genre", "loudness_category", "mode_name", "release_day_of_week", "is_weekend_release"],
        FIG_DIR / "categorical_analysis.png",
    )
    viz.plot_model_comparison(results, FIG_DIR / "model_comparison.png")
    viz.plot_roc_curves(y_test, results, FIG_DIR / "roc_curves.png")
    viz.plot_confusion_matrix(
        y_test, best["y_pred"], FIG_DIR / "confusion_matrix.png"
    )

    # Importancia de features (reusa preprocesador ya ajustado)
    prep = build_preprocessor()
    prep.fit(X_train)
    feat_names = get_feature_names(prep)
    viz.plot_feature_importance(best["model"], feat_names, FIG_DIR / "feature_importance.png")

    print("Listo. Figuras en", FIG_DIR)


if __name__ == "__main__":
    main()
