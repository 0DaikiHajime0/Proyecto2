"""
Modulo de preprocesamiento de datos
-----------------------------------
Pipeline de transformacion para el dataset de Spotify Artist Streaming
Analytics (2015-2025). Prepara los datos para un problema de clasificacion
multiclase de la categoria de popularidad de los tracks.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Definicion del problema y del esquema de features
# ---------------------------------------------------------------------------
TARGET = "popularity_category"

# Columnas que provocan fuga de informacion (leakage) y deben descartarse.
# - `popularity` es la fuente de la que se deriva `popularity_category`.
# - `stream_count` / `log_stream_count` son consecuencia de la popularidad.
# - `upbeat_score` es una combinacion directa de `danceability` y `energy`.
LEAKY_COLUMNS = [
    "popularity",
    "stream_count",
    "log_stream_count",
    "upbeat_score",
    TARGET,
]

# Identificadores y texto libre de alta cardinalidad (no utiles como features).
ID_COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "release_date",
    "label",
]

NUMERIC_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "instrumentalness",
    "tempo",
    "release_year",
    "release_month",
    "release_quarter",
    "explicit",
    "artist_track_count",
    "dance_energy",
    "is_recent_release",
    "artist_exposure",
]

CATEGORICAL_FEATURES = [
    "genre",
    "loudness_category",
    "key_name",
    "mode_name",
    "release_day_of_week",
    "is_weekend_release",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Funciones del pipeline
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset de Spotify desde un archivo CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset y valida la variable objetivo.

    El dataset sintetico no presenta valores nulos, pero se documenta y
    defiende el manejo por si se reemplaza por datos reales.
    """
    df = df.copy()

    # Validar que la columna objetivo exista
    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no existe en el dataset.")

    # Asegurar tipos y eliminar duplicados por track_id
    df = df.drop_duplicates(subset=["track_id"] if "track_id" in df.columns else None)

    # Manejo defensivo de nulos en la columna objetivo
    df = df[df[TARGET].notna()]

    # Ordenar categorias para consistencia
    df[TARGET] = pd.Categorical(
        df[TARGET], categories=["Low", "Medium", "High"], ordered=True
    )
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las columnas utiles eliminando IDs, texto y leakage."""
    drop_cols = [c for c in (LEAKY_COLUMNS + ID_COLUMNS) if c in df.columns]
    return df.drop(columns=drop_cols)


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador (scaling numerico + OHE categorico)."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Devuelve los nombres de las features despues del preprocesamiento."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_features


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Divide en train/test estratificado por la categoria de popularidad."""
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def make_pipeline(model) -> Pipeline:
    """Crea un Pipeline sklearn con preprocesador + modelo."""
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", model)]
    )
