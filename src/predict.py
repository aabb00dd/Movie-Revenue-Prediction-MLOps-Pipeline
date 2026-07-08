"""
Prediction utilities for the Movie Revenue Prediction MLOps Pipeline.

This module loads the trained model and preprocessing artifacts, transforms
new movie metadata, and returns a high/low revenue prediction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, MODELS_DIR, TARGET_DEFINITION
from src.features import transform_features


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained model from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run training first: python -m src.train"
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_model_metadata() -> dict[str, Any]:
    """
    Load saved model metadata.
    """
    if not METADATA_PATH.exists():
        return {
            "model_type": "XGBClassifier",
            "target_definition": TARGET_DEFINITION,
        }

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_movie_revenue(movie_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict whether a movie is likely to achieve high revenue.

    Args:
        movie_data: Dictionary containing movie metadata.

    Returns:
        Prediction response dictionary.
    """
    model = load_model()
    metadata = load_model_metadata()

    input_df = pd.DataFrame([movie_data])

    X = transform_features(
        input_df,
        artifacts_dir=MODELS_DIR,
    )

    high_revenue_probability = float(model.predict_proba(X)[0, 1])
    prediction_class = int(high_revenue_probability >= 0.5)

    prediction_label = "high_revenue" if prediction_class == 1 else "low_revenue"

    return {
        "prediction": prediction_label,
        "prediction_class": prediction_class,
        "high_revenue_probability": high_revenue_probability,
        "target_definition": metadata.get("target_definition", TARGET_DEFINITION),
        "model": "XGBoost",
        "feature_count": int(X.shape[1]),
    }