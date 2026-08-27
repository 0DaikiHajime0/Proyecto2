"""
Ingenieria de features para el dataset de Spotify
-------------------------------------------------
Crea variables derivadas a partir de los metadatos y caracteristicas de
audio de los tracks. Todas las features son calculables antes de conocer
la popularidad, por lo que no introducen fuga de informacion.
"""
from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anade features derivadas al dataframe de Spotify.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas originales del dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas features anadidas.
    """
    df = df.copy()

    # Interaccion entre bailabilidad y energia: proxy de "pegajosidad" del track.
    if "danceability" in df.columns and "energy" in df.columns:
        df["dance_energy"] = df["danceability"] * df["energy"]

    # Indicador de lanzamiento reciente (era de streaming mas competitiva).
    if "release_year" in df.columns:
        df["is_recent_release"] = (df["release_year"] >= 2020).astype(int)

    # Densidad de streams esperada por track del artista (proxy de exposicion).
    if "artist_track_count" in df.columns:
        df["artist_exposure"] = 1.0 / (df["artist_track_count"].clip(lower=1))

    return df
