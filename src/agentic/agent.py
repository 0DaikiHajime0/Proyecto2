"""
Agente de IA para analisis de popularidad de Spotify
-----------------------------------------------------
Combina las predicciones del modelo de ML con el motor RAG para explicar
predicciones de popularidad, generar recomendaciones de produccion y
responder preguntas en lenguaje natural sobre el dataset.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .rag_engine import SpotifyRAGEngine

CLASS_DESCRIPTIONS = {
    "Low": "baja popularidad (pocas reproducciones esperadas)",
    "Medium": "popularidad media (rendimiento intermedio)",
    "High": "alta popularidad (potencial de exito)",
}


class SpotifyPopularityAgent:
    """Agente que explica y recomienda acciones sobre la popularidad."""

    def __init__(self, model=None, rag_engine: Optional[SpotifyRAGEngine] = None):
        self.model = model
        self.rag_engine = rag_engine

    # ------------------------------------------------------------------ #
    def _feature_summary(self, track: dict) -> str:
        """Resumen legible de las features clave del track."""
        parts = []
        for key in ["genre", "danceability", "energy", "tempo", "loudness", "explicit"]:
            if key in track and track[key] is not None:
                if key in ("danceability", "energy"):
                    parts.append(f"{key}={track[key]:.2f}")
                elif key in ("tempo", "loudness"):
                    parts.append(f"{key}={track[key]:.1f}")
                else:
                    parts.append(f"{key}={track[key]}")
        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    def explain_prediction(self, track: dict, prediction: str, probability: float) -> str:
        """Explica la prediccion para un track especifico usando RAG."""
        prob = float(np.atleast_1d(probability)[0])
        confidence = "alta" if prob > 0.6 else "media" if prob > 0.4 else "baja"
        explanation = (
            f"**Analisis de popularidad del track**\n\n"
            f"- Prediccion: {prediction} ({CLASS_DESCRIPTIONS.get(prediction, prediction)})\n"
            f"- Confianza del modelo: {confidence} ({prob:.1%})\n"
            f"- Caracteristicas: {self._feature_summary(track)}\n\n"
        )

        # Contexto recuperado por RAG (perfil del genero)
        genre = track.get("genre")
        retrieved = []
        if self.rag_engine is not None:
            if genre:
                prof = self.rag_engine.get_genre_profile(genre)
                if prof:
                    retrieved.append(
                        f"En el genero {genre}, la popularidad promedio es "
                        f"{prof['avg_popularity']:.1f} y el {prof['pct_high']:.0%} de tracks "
                        f"alcanzan alta popularidad (danceability prom={prof['avg_danceability']:.2f}, "
                        f"energy prom={prof['avg_energy']:.2f})."
                    )
            retrieved += [d["text"] for d in self.rag_engine.retrieve(
                f"{prediction} popularity {genre or ''}", k=2
            )]

        if retrieved:
            explanation += "**Contexto relevante (RAG):**\n"
            explanation += "\n".join(f"- {r}" for r in retrieved[:3]) + "\n\n"

        explanation += self._drivers(track, prediction)
        return explanation

    # ------------------------------------------------------------------ #
    def _drivers(self, track: dict, prediction: str) -> str:
        """Identifica factores que impulsan o limitan la popularidad."""
        factors = []
        if track.get("danceability", 0) > 0.7:
            factors.append("- Alta bailabilidad (favorece la difusion)")
        if track.get("energy", 0) > 0.7:
            factors.append("- Alta energia (mayor enganche)")
        if track.get("instrumentalness", 0) > 0.5:
            factors.append("- Instrumentalidad alta (suele limitar alcance masivo)")
        if track.get("explicit"):
            factors.append("- Contenido explicito (puede restringir radios/playlists)")
        if track.get("artist_track_count", 99) <= 1:
            factors.append("- Artista con pocos tracks (menor exposicion acumulada)")
        if not factors:
            factors.append("- Sin factores de riesgo destacados en las features disponibles")
        title = "Factores detectados:" if prediction != "High" else "Factores de exito:"
        return f"**{title}**\n" + "\n".join(factors)

    # ------------------------------------------------------------------ #
    def generate_recommendation(self, track: dict, prediction: str, probability: float) -> str:
        """Genera recomendaciones de produccion/distribucion."""
        if prediction == "High":
            return (
                "**Plan de impulso**\n\n"
                "El track ya proyecta alta popularidad. Recomendaciones:\n"
                "1. Priorizar lanzamiento en playlists editoriales y redes sociales.\n"
                "2. Campana de streaming dirigida al genero y segmento del artista.\n"
                "3. Monitorizar retencion a 7 dias para consolidar el alcance."
            )

        recs = []
        if track.get("danceability", 0) < 0.5:
            recs.append("1. Incrementar bailabilidad (>=0.6) para mejorar difusion.")
        if track.get("energy", 0) < 0.5:
            recs.append("2. Subir el nivel de energia para mayor enganche.")
        if track.get("artist_track_count", 99) <= 1:
            recs.append("3. Aumentar frecuencia de lanzamiento para acumular exposicion del artista.")
        if not track.get("explicit"):
            pass
        if track.get("instrumentalness", 0) > 0.5:
            recs.append("4. Reducir instrumentalidad si se busca alcance masivo.")
        if not recs:
            recs.append("1. Evaluar posicionamiento en nichos del genero para construir traccion.")

        header = (
            f"**Plan de retencion/impulso** (Prediccion: {prediction}, "
            f"confianza {float(np.atleast_1d(probability)[0]):.1%})\n\n"
            "Acciones recomendadas:\n"
        )
        return header + "\n".join(recs)

    # ------------------------------------------------------------------ #
    def query(self, question: str) -> str:
        """Responde preguntas sobre el dataset usando RAG."""
        if self.rag_engine is None:
            return "Motor RAG no inicializado. Cargue los datos primero."
        docs = self.rag_engine.retrieve(question, k=4)
        stats = self.rag_engine.get_popularity_stats()
        context = "\n".join(f"- {d['text']}" for d in docs)
        response = (
            "**Respuesta basada en datos historicos (RAG)**\n\n"
            f"Estadisticas generales:\n"
            f"- Total de tracks: {stats.get('total_tracks', 0):,}\n"
            f"- Popularidad promedio: {stats.get('avg_popularity', 0):.1f}\n"
            f"- % Alta / Media / Baja: "
            f"{stats.get('pct_high', 0):.0%} / {stats.get('pct_medium', 0):.0%} / "
            f"{stats.get('pct_low', 0):.0%}\n"
            f"- Generos: {stats.get('n_genres', 0)}, Artistas: {stats.get('n_artists', 0):,}\n\n"
            f"Documentos recuperados:\n{context}"
        )
        return response
