"""Tests ligeros del pipeline de Spotify popularity."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocess import load_data, clean_data, split_data, ALL_FEATURES  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.agentic.rag_engine import SpotifyRAGEngine  # noqa: E402

RAW = PROJECT_DIR / "data" / "raw" / "spotify_data_processed.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_data(str(RAW))


def test_clean_data_keeps_target(raw_df):
    df = clean_data(raw_df)
    assert "popularity_category" in df.columns
    assert df["popularity_category"].isna().sum() == 0


def test_build_features_adds_columns(raw_df):
    df = build_features(raw_df)
    for col in ["dance_energy", "is_recent_release", "artist_exposure"]:
        assert col in df.columns


def test_split_shapes(raw_df):
    df = build_features(raw_df)
    X_train, X_test, y_train, y_test = split_data(df)
    assert list(X_train.columns) == ALL_FEATURES
    assert len(X_train) + len(X_test) == len(df)
    assert y_train.nunique() == 3


def test_rag_engine_builds_kb():
    eng = SpotifyRAGEngine(str(RAW))
    docs = eng.retrieve("popularidad genero Pop", k=2)
    assert isinstance(docs, list)
    stats = eng.get_popularity_stats()
    assert stats["total_tracks"] > 0
