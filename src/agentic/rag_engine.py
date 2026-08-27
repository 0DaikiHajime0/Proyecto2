"""
Motor RAG para analisis de popularidad de Spotify
-------------------------------------------------
Utiliza Retrieval-Augmented Generation para construir una base de
conocimiento a partir del dataset (perfiles por genero, tendencias por
ano, artistas destacados, insights de audio vs popularidad) y recuperar
contexto relevante mediante TF-IDF (sin dependencias externas).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parents[2]


class SpotifyRAGEngine:
    """Motor RAG sobre estadisticas agregadas del dataset de Spotify."""

    def __init__(self, data_path: str = None):
        self.data = None
        self.documents: list = []
        self.doc_metadata: list = []
        self.vectorizer = None
        self.doc_vectors = None
        if data_path:
            self.load_data(data_path)

    # ------------------------------------------------------------------ #
    def load_data(self, path: str):
        """Carga el dataset y construye la base de conocimiento."""
        self.data = pd.read_csv(path)
        if "popularity_category" not in self.data.columns and "popularity" in self.data.columns:
            self.data["popularity_category"] = pd.cut(
                self.data["popularity"],
                bins=[-1, 33, 66, 100],
                labels=["Low", "Medium", "High"],
            )
        self.build_knowledge_base()
        print(f"Base de conocimiento creada: {len(self.documents)} documentos")

    # ------------------------------------------------------------------ #
    def build_knowledge_base(self):
        """Genera documentos de texto a partir de estadisticas agregadas."""
        df = self.data
        docs, meta = [], []

        # 1) Perfil por genero
        for genre, g in df.groupby("genre"):
            prof = (
                f"Genero {genre}: {len(g)} tracks. "
                f"Popularidad promedio {g['popularity'].mean():.1f} "
                f"(Low { (g['popularity_category']=='Low').mean():.0%}, "
                f"Medium {(g['popularity_category']=='Medium').mean():.0%}, "
                f"High {(g['popularity_category']=='High').mean():.0%}). "
                f"Audio promedio: danceability {g['danceability'].mean():.2f}, "
                f"energy {g['energy'].mean():.2f}, "
                f"loudness {g['loudness'].mean():.1f} dB, "
                f"tempo {g['tempo'].mean():.0f} BPM. "
                f"Explicit {g['explicit'].mean():.0%}."
            )
            docs.append(prof)
            meta.append({"type": "genre", "genre": genre})

        # 2) Tendencias por ano
        for year, g in df.groupby("release_year"):
            prof = (
                f"Ano {year}: {len(g)} tracks lanzados. "
                f"Popularidad promedio {g['popularity'].mean():.1f}. "
                f"Streams promedio {g['stream_count'].mean():.0f}. "
                f"Porcentaje explicit {g['explicit'].mean():.0%}."
            )
            docs.append(prof)
            meta.append({"type": "year", "year": int(year)})

        # 3) Artistas destacados (top por popularidad promedio, min 3 tracks)
        artist_stats = (
            df.groupby("artist_name")
            .agg(n=("popularity", "size"), pop=("popularity", "mean"))
            .query("n >= 5")
            .sort_values("pop", ascending=False)
            .head(15)
        )
        for artist, row in artist_stats.iterrows():
            prof = (
                f"Artista {artist}: {int(row['n'])} tracks en el dataset, "
                f"popularidad promedio {row['pop']:.1f}. "
                f"Considerado referente de alto rendimiento."
            )
            docs.append(prof)
            meta.append({"type": "artist", "artist": artist})

        # 4) Insights globales de audio vs popularidad
        for col in ["danceability", "energy", "loudness", "tempo", "instrumentalness", "duration_ms"]:
            hi = df[df["popularity_category"] == "High"][col].mean()
            lo = df[df["popularity_category"] == "Low"][col].mean()
            docs.append(
                f"En {col}: los tracks de alta popularidad promedian {hi:.2f} "
                f"mientras que los de baja popularidad promedian {lo:.2f}. "
                f"Diferencia de {(hi - lo):.2f}."
            )
            meta.append({"type": "audio_insight", "feature": col})

        self.documents = docs
        self.doc_metadata = meta
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 5) -> list:
        """Recupera los k documentos mas relevantes para la consulta."""
        if self.vectorizer is None or self.doc_vectors is None:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.doc_vectors).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {"score": float(sims[i]), "text": self.documents[i], "meta": self.doc_metadata[i]}
            for i in top_idx
            if sims[i] > 0
        ]

    # ------------------------------------------------------------------ #
    def get_popularity_stats(self) -> dict:
        """Estadisticas generales de popularidad para contexto del agente."""
        if self.data is None:
            return {}
        df = self.data
        return {
            "total_tracks": int(len(df)),
            "avg_popularity": float(df["popularity"].mean()),
            "pct_high": float((df["popularity_category"] == "High").mean()),
            "pct_medium": float((df["popularity_category"] == "Medium").mean()),
            "pct_low": float((df["popularity_category"] == "Low").mean()),
            "n_genres": int(df["genre"].nunique()),
            "n_artists": int(df["artist_name"].nunique()),
            "year_min": int(df["release_year"].min()),
            "year_max": int(df["release_year"].max()),
        }

    def get_genre_profile(self, genre: str) -> Optional[dict]:
        """Devuelve el perfil agregado de un genero."""
        if self.data is None or genre not in self.data["genre"].unique():
            return None
        g = self.data[self.data["genre"] == genre]
        return {
            "genre": genre,
            "n": len(g),
            "avg_popularity": float(g["popularity"].mean()),
            "pct_high": float((g["popularity_category"] == "High").mean()),
            "avg_danceability": float(g["danceability"].mean()),
            "avg_energy": float(g["energy"].mean()),
        }
