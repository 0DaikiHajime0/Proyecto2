"""
Script de entrenamiento de modelos
-----------------------------------
Entrena modelos de ML y registra experimentos en MLflow.
"""
import os
import click
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "processed_data.csv"
MODELS_DIR = PROJECT_DIR / "models"


@click.command()
@click.option("--data", default=str(PROCESSED_DATA_PATH), help="Ruta a datos procesados")
@click.option("--output", default=str(MODELS_DIR), help="Ruta para guardar modelos")
def main(data: str, output: str):
    """Entrena modelos y registra en MLflow."""
    click.echo("Cargando datos procesados...")
    df = pd.read_csv(data)
    click.echo(f"Dimensiones: {df.shape}")

    # TODO: Implementar entrenamiento
    # - Split train/test
    # - Entrenar múltiples modelos
    # - Evaluar con métricas
    # - Registrar en MLflow
    # - Guardar mejor modelo

    click.echo("Entrenamiento completado.")


if __name__ == "__main__":
    main()
