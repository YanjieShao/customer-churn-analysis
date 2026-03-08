import os
import argparse
import joblib
import pandas as pd

from utils import clean_telco, add_features


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model


def load_new_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning and feature engineering logic as in training.
    Important:
    - The new input data should NOT contain the target column 'Churn'
    - customerID can exist or not
    """
    df = df.copy()

    # If the input accidentally contains Churn, drop it
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # clean_telco() in train.py expects a Churn column for mapping,
    # so here we implement only the relevant cleaning steps for prediction.
    # We should NOT call clean_telco() directly if it strictly depends on Churn existing.

    # Convert TotalCharges to numeric if it exists
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Apply feature engineering
    df = add_features(df)

    return df


def predict(model, X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Return both probability and class prediction.
    """
    churn_prob = model.predict_proba(X_new)[:, 1]
    churn_pred = (churn_prob >= 0.5).astype(int)

    result = X_new.copy()
    result["churn_probability"] = churn_prob
    result["predicted_churn"] = churn_pred
    result["predicted_churn_label"] = result["predicted_churn"].map({0: "No", 1: "Yes"})

    return result


def save_predictions(df_pred: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pred.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn using a trained model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/logistic_regression.joblib",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/sample_new_customers.csv",
        help="Path to input CSV containing new customer records"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/predictions.csv",
        help="Path to save prediction output CSV"
    )

    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)

    print("Loading input data...")
    df_new = load_new_data(args.input_path)

    print("Preparing features...")
    X_new = prepare_features(df_new)

    print("Generating predictions...")
    df_pred = predict(model, X_new)

    print("\nPrediction preview:")
    print(df_pred.head())

    save_predictions(df_pred, args.output_path)


if __name__ == "__main__":
    main()
