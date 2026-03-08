import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import clean_telco, add_features
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

RANDOM_STATE = 42
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def fit_and_eval(model, X_train, X_test, y_train, y_test, preprocessor, model_name: str):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"\n===== {model_name} =====")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    metrics = {
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }

    return pipe, metrics


def plot_roc_curve(model_pipe, X_test, y_test, title: str, save_path: str = None):
    y_prob = model_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_pr_curve(model_pipe, X_test, y_test, title: str, save_path: str = None):
    y_prob = model_pipe.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def save_model(model_pipe, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_pipe, path)
    print(f"Model saved to {path}")


def get_feature_names(preprocessor: ColumnTransformer):
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_features).tolist()

    return list(num_features) + cat_feature_names


def show_lr_top_features(lr_pipeline: Pipeline, top_k=15):
    pre = lr_pipeline.named_steps["preprocess"]
    model = lr_pipeline.named_steps["model"]

    feature_names = get_feature_names(pre)
    coefs = model.coef_.ravel()

    top_positive_idx = np.argsort(coefs)[-top_k:][::-1]
    top_negative_idx = np.argsort(coefs)[:top_k]

    print("\nTop churn-increasing features:")
    for idx in top_positive_idx:
        print(f"{feature_names[idx]}: {coefs[idx]:.4f}")

    print("\nTop churn-decreasing features:")
    for idx in top_negative_idx:
        print(f"{feature_names[idx]}: {coefs[idx]:.4f}")


def show_rf_feature_importance(rf_pipeline: Pipeline, top_k=15):
    pre = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    feature_names = get_feature_names(pre)
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\nTop Random Forest feature importances:")
    print(importance_df.head(top_k))

    return importance_df


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load and prepare data
    df = load_data(DATA_PATH)
    df = clean_telco(df)
    df = add_features(df)

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    lr_pipe, lr_metrics = fit_and_eval(
        lr_model, X_train, X_test, y_train, y_test, preprocessor, "Logistic Regression"
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf_pipe, rf_metrics = fit_and_eval(
        rf_model, X_train, X_test, y_train, y_test, preprocessor, "Random Forest"
    )

    # Optional XGBoost
    xgb_pipe = None
    xgb_metrics = None
    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )

        xgb_pipe, xgb_metrics = fit_and_eval(
            xgb_model, X_train, X_test, y_train, y_test, preprocessor, "XGBoost"
        )
    except Exception as e:
        print("\nXGBoost skipped:", e)

    # Plot curves
    plot_roc_curve(lr_pipe, X_test, y_test, "Logistic Regression ROC", "results/lr_roc.png")
    plot_pr_curve(lr_pipe, X_test, y_test, "Logistic Regression PR", "results/lr_pr.png")

    plot_roc_curve(rf_pipe, X_test, y_test, "Random Forest ROC", "results/rf_roc.png")
    plot_pr_curve(rf_pipe, X_test, y_test, "Random Forest PR", "results/rf_pr.png")

    if xgb_pipe is not None:
        plot_roc_curve(xgb_pipe, X_test, y_test, "XGBoost ROC", "results/xgb_roc.png")
        plot_pr_curve(xgb_pipe, X_test, y_test, "XGBoost PR", "results/xgb_pr.png")

    # Model interpretation
    show_lr_top_features(lr_pipe, top_k=15)
    rf_importance_df = show_rf_feature_importance(rf_pipe, top_k=15)
    rf_importance_df.to_csv("results/rf_feature_importance.csv", index=False)

    # Save models
    save_model(lr_pipe, "models/logistic_regression.joblib")
    save_model(rf_pipe, "models/random_forest.joblib")
    if xgb_pipe is not None:
        save_model(xgb_pipe, "models/xgboost.joblib")

    # Save metrics
    all_metrics = [lr_metrics, rf_metrics]
    if xgb_metrics is not None:
        all_metrics.append(xgb_metrics)

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nAll done. Results saved under results/ and models/.")


if __name__ == "__main__":
    main()
