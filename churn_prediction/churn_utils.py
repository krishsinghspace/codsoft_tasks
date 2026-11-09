"""Shared helpers: feature engineering + preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"
# Identifiers: carry no behavioural signal (and would let a model memorise rows)
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "EstimatedSalary", "BalanceSalaryRatio", "TenureByAge", "HasBalance"]
BINARY = ["HasCrCard", "IsActiveMember"]
CATEGORICAL = ["Geography", "Gender"]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def load_data(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (applied identically in training and prediction)."""
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["TenureByAge"] = df["Tenure"] / df["Age"]
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    return df[FEATURES]


def make_preprocessor(scale=True):
    """Scaling matters for Logistic Regression, not for tree models."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def risk_tier(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"
