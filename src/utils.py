import numpy as np
import pandas as pd


def clean_telco(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if is_training and "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = [-1, 12, 24, 48, np.inf]
    labels = ["0-12", "12-24", "24-48", "48+"]
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    available = [c for c in service_cols if c in df.columns]

    def to01(x):
        return 1 if x == "Yes" else 0

    for c in available:
        df[c + "_01"] = df[c].map(to01)

    if available:
        df["services_count"] = df[[c + "_01" for c in available]].sum(axis=1)
        df = df.drop(columns=[c + "_01" for c in available])

    return df
