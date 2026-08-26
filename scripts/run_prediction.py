"""
Script de predicción
--------------------
Carga un modelo entrenado y genera predicciones.
"""
import os
import click
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "processed_data.csv"
MODELS_DIR = PROJECT_DIR / "models"


@click.command()
@click.option("--data", default=str(PROCESSED_DATA_PATH), help="Ruta a datos de entrada")
@click.option("--model", default="model.pkl", help="Nombre del modelo")
@click.option("--output", default=str(PROCESSED_DATA_PATH.parent / "predictions.csv"), help="Ruta de salida")
def main(data: str, model: str, output: str):
    """Genera predicciones con el modelo entrenado."""
    click.echo("Cargando datos...")
    df = pd.read_csv(data)
    click.echo(f"Dimensiones: {df.shape}")

    # TODO: Implementar predicción
    # - Cargar modelo serializado
    # - Generar predicciones
    # - Guardar resultados

    click.echo("Predicciones generadas.")


if __name__ == "__main__":
    main()
