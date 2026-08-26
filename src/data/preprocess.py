"""
Modulo de preprocesamiento de datos
-----------------------------------
Pipeline de transformacion para el dataset de Churn.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y preprocesa los datos."""
    df = df.copy()

    # Eliminar customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Corregir TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    return df


def encode_target(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """Codifica la variable objetivo."""
    df = df.copy()
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crea nuevas features derivadas."""
    df = df.copy()

    # Promedio de cargo mensual por tenure
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Indicador de servicios multiples
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df['num_services'] = df[existing].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding a variables categoricas."""
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   num_cols: list) -> tuple:
    """Escala variables numericas con StandardScaler."""
    scaler = StandardScaler()
    cols_to_scale = [c for c in num_cols if c in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler
