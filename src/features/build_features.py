"""
Modulo de feature engineering
------------------------------
Funciones para crear y transformar features.
"""
import pandas as pd
import numpy as np


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Crea grupos de tenure."""
    df = df.copy()
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
        )
    return df


def create_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Crea ratio de cargo mensual vs total."""
    df = df.copy()
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta el numero de servicios contratados."""
    df = df.copy()
    service_cols = [
        'PhoneService', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df['total_services'] = df[existing].apply(
            lambda x: sum(1 for v in x if v in ['Yes', 'DSL', 'Fiber optic']),
            axis=1
        )
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las transformaciones de features."""
    df = create_tenure_groups(df)
    df = create_charge_ratio(df)
    df = create_service_count(df)
    return df
