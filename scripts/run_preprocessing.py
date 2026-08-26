"""
Script de preprocesamiento de datos
-----------------------------------
Carga el dataset raw, aplica transformaciones y guarda los datos procesados.
"""
import os
import click
import pandas as pd
from pathlib import Path

# Configurar rutas
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed"


@click.command()
@click.option("--input", default=str(RAW_DATA_PATH), help="Ruta al archivo raw")
@click.option("--output", default=str(PROCESSED_DATA_PATH), help="Ruta de salida")
def main(input: str, output: str):
    """Ejecuta el pipeline de preprocesamiento."""
    click.echo("Cargando datos...")
    df = pd.read_csv(input)
    click.echo(f"Dimensiones originales: {df.shape}")

    # TODO: Agregar pipeline de preprocesamiento
    # - Manejar valores faltantes
    # - Codificar variables categóricas
    # - Escalar variables numéricas
    # - Feature engineering

    # Guardar datos procesados
    os.makedirs(output, exist_ok=True)
    df.to_csv(os.path.join(output, "processed_data.csv"), index=False)
    click.echo(f"Datos guardados en: {output}")


if __name__ == "__main__":
    main()
