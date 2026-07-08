"""
Training script for the Movie Revenue Prediction MLOps Pipeline.

Run from the project root:

    python -m src.train

This script:
    1. Loads raw TMDB data
    2. Cleans data
    3. Uses the high_revenue target
    4. Splits train/test data
    5. Fits preprocessing on training data only
    6. Trains an XGBoost model
    7. Evaluates Accuracy, F1, and ROC-AUC
    8. Logs parameters and metrics with MLflow
    9. Saves model and metadata artifacts
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data import load_and_clean_data
from src.features import fit_transform_features, transform_features


DATA_PATH = Path("data/raw/TMDB_movie_dataset_v11.csv")
MODELS_DIR = Path("models")

MODEL_PATH = MODELS_DIR / "revenue_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

TARGET_COLUMN = "high_revenue"
EXPERIMENT_NAME = "movie-revenue-prediction"


def build_model() -> XGBClassifier:
    """
    Create the XGBoost model.

    These are solid baseline parameters for a structured-data binary
    classification problem. You can tune them later with GridSearchCV.
    """
    return XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate the trained model.

    Args:
        model: Trained XGBoost model.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary containing accuracy, F1 score, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    return metrics


def save_model_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
) -> None:
    """
    Save model metadata as JSON.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def train() -> None:
    """
    Main training pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    print("Cleaned dataset shape:", df.shape)

    y = df[TARGET_COLUMN]

    print("Target distribution:")
    print(y.value_counts(normalize=True))

    print("Splitting train/test data...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    print("Fitting preprocessing on training data only...")
    X_train = fit_transform_features(
        train_df,
        artifacts_dir=MODELS_DIR,
        save_artifacts=True,
    )

    print("Transforming test data using saved preprocessing artifacts...")
    X_test = transform_features(
        test_df,
        artifacts_dir=MODELS_DIR,
    )

    print("Training XGBoost model...")
    model = build_model()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_high_revenue_baseline"):
        model.fit(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        model_params = model.get_params()

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param(
            "target_definition",
            "1 if revenue is above the median revenue in the cleaned training dataset, else 0",
        )
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
            
        for param_name in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "objective",
            "eval_metric",
        ]:
            mlflow.log_param(param_name, model_params[param_name])

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print("Saving trained model...")
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(MODELS_DIR / "feature_columns.json"))
        mlflow.log_artifact(str(MODELS_DIR / "preprocessing.joblib"))

        metadata = {
            "project_name": "Movie Revenue Prediction MLOps Pipeline",
            "model_name": "XGBoost High Revenue Classifier",
            "model_type": "XGBClassifier",
            "target_column": TARGET_COLUMN,
            "target_definition": (
                "high_revenue = 1 if revenue is above the median revenue "
                "in the cleaned dataset, else 0"
            ),
            "problem_type": "binary_classification",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(DATA_PATH),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "total_clean_rows": int(len(df)),
            "feature_count": int(X_train.shape[1]),
            "feature_columns_path": str(MODELS_DIR / "feature_columns.json"),
            "preprocessing_path": str(MODELS_DIR / "preprocessing.joblib"),
            "model_path": str(MODEL_PATH),
            "revenue_median_threshold": float(df.attrs.get("revenue_median")),
            "metrics": metrics,
            "model_params": {
                "n_estimators": model_params["n_estimators"],
                "max_depth": model_params["max_depth"],
                "learning_rate": model_params["learning_rate"],
                "subsample": model_params["subsample"],
                "colsample_bytree": model_params["colsample_bytree"],
                "objective": model_params["objective"],
                "eval_metric": model_params["eval_metric"],
                "random_state": model_params["random_state"],
                "gamma": model_params["gamma"],
            },
        }

        save_model_metadata(METADATA_PATH, metadata)
        mlflow.log_artifact(str(METADATA_PATH))

    print("\nTraining complete.")
    print("Saved files:")
    print(f"- {MODEL_PATH}")
    print(f"- {MODELS_DIR / 'feature_columns.json'}")
    print(f"- {MODELS_DIR / 'preprocessing.joblib'}")
    print(f"- {METADATA_PATH}")

    print("\nModel performance:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    train()