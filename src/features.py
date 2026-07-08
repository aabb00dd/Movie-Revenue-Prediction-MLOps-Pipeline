"""
Feature engineering utilities for the Movie Revenue Prediction MLOps Pipeline.

This module converts notebook-based feature engineering into reusable functions.

Main purpose:
    During training:
        fit_transform_features()

    During API prediction:
        transform_features()

This guarantees that the API uses the exact same feature columns, encoders,
top categories, and scalers that were fitted during training.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DEFAULT_ARTIFACTS_DIR = Path("models")

FEATURE_COLUMNS_FILE = "feature_columns.json"
TOP_COMPANIES_FILE = "top_100_companies.json"
TOP_KEYWORDS_FILE = "top_100_keywords.json"
PREPROCESSING_FILE = "preprocessing.joblib"


NUMERIC_STANDARD_COLUMNS = [
    "runtime",
    "release_year",
]

NUMERIC_MINMAX_COLUMNS = [
    "budget",
]

BINARY_COLUMNS = [
    "adult",
]


TARGET_AND_LEAKAGE_COLUMNS = [
    "high_revenue",
    "revenue",
]


def slugify(value: str) -> str:
    """
    Convert category names into safe feature names.

    Example:
        "Warner Bros. Pictures" -> "warner_bros_pictures"
    """
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_text(value: Any) -> str:
    """
    Normalize text category values.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def parse_multi_value_cell(value: Any) -> list[str]:
    """
    Parse TMDB-style multi-value columns.

    Handles:
        - "drama, crime"
        - "drama|crime"
        - "['Drama', 'Crime']"
        - [{"name": "Drama"}, {"name": "Crime"}]
        - already existing Python lists from the API
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        parsed_items = value

    elif isinstance(value, tuple) or isinstance(value, set):
        parsed_items = list(value)

    else:
        value_str = str(value).strip()

        if not value_str:
            return []

        if value_str.startswith("[") or value_str.startswith("{"):
            try:
                parsed_items = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                parsed_items = re.split(r"[,|;]", value_str)
        else:
            parsed_items = re.split(r"[,|;]", value_str)

    cleaned_items: list[str] = []

    for item in parsed_items:
        if isinstance(item, dict):
            item_value = item.get("name") or item.get("english_name") or ""
        else:
            item_value = item

        item_value = normalize_text(item_value)

        if item_value:
            cleaned_items.append(item_value)

    return sorted(set(cleaned_items))


def get_top_values(df: pd.DataFrame, column: str, top_n: int) -> list[str]:
    """
    Get the most frequent values from a multi-value column.

    Args:
        df: Input dataframe.
        column: Column containing multi-value data.
        top_n: Number of most frequent values to keep.

    Returns:
        List of top values.
    """
    if column not in df.columns:
        return []

    value_counts: dict[str, int] = {}

    for cell in df[column]:
        values = parse_multi_value_cell(cell)

        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1

    sorted_values = sorted(
        value_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [value for value, _ in sorted_values[:top_n]]


def get_all_values(df: pd.DataFrame, column: str) -> list[str]:
    """
    Get all unique values from a multi-value column.
    Useful for genres because the number of genres is manageable.
    """
    if column not in df.columns:
        return []

    unique_values: set[str] = set()

    for cell in df[column]:
        values = parse_multi_value_cell(cell)
        unique_values.update(values)

    return sorted(unique_values)


def add_multi_hot_features(
    features: pd.DataFrame,
    source_df: pd.DataFrame,
    source_column: str,
    categories: list[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Add multi-hot encoded features for a multi-value column.

    Example:
        genres = ["drama", "crime"]

        Creates:
            genre_drama = 1
            genre_crime = 1
    """
    for category in categories:
        feature_name = f"{prefix}_{slugify(category)}"

        if source_column not in source_df.columns:
            features[feature_name] = 0
            continue

        features[feature_name] = source_df[source_column].apply(
            lambda cell: int(category in parse_multi_value_cell(cell))
        )

    return features


def add_single_value_one_hot_features(
    features: pd.DataFrame,
    source_df: pd.DataFrame,
    source_column: str,
    categories: list[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Add one-hot encoded features for a single-value categorical column.

    Example:
        original_language = "en"

        Creates:
            original_language_en = 1
    """
    for category in categories:
        feature_name = f"{prefix}_{slugify(category)}"

        if source_column not in source_df.columns:
            features[feature_name] = 0
            continue

        features[feature_name] = source_df[source_column].apply(
            lambda value: int(normalize_text(value) == category)
        )

    return features


def prepare_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date-based features from release_date.
    """
    df = df.copy()

    if "release_date" in df.columns:
        release_date = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = release_date.dt.year
        df["release_month"] = release_date.dt.month
    else:
        df["release_year"] = np.nan
        df["release_month"] = np.nan

    return df


def create_base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the base dataframe used for feature engineering.
    """
    df = df.copy()

    # Remove target/leakage columns if they exist.
    df = df.drop(
        columns=[col for col in TARGET_AND_LEAKAGE_COLUMNS if col in df.columns],
        errors="ignore",
    )

    df = prepare_date_features(df)

    for column in NUMERIC_STANDARD_COLUMNS + NUMERIC_MINMAX_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            df[column] = 0

        df[column] = df[column].astype(str).str.lower().map(
            {
                "true": 1,
                "1": 1,
                "yes": 1,
                "false": 0,
                "0": 0,
                "no": 0,
            }
        )

        df[column] = df[column].fillna(0).astype(int)

    return df


def fit_transform_features(
    df: pd.DataFrame,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Fit feature engineering artifacts on training data and transform the data.

    Use this ONLY during training.

    This function:
        1. Creates numerical features.
        2. Fits scalers.
        3. Finds all genres.
        4. Finds top 100 production companies.
        5. Finds top 100 keywords.
        6. Creates one-hot/multi-hot encoded features.
        7. Saves feature columns and preprocessing artifacts.

    Args:
        df: Cleaned training dataframe.
        artifacts_dir: Directory where feature artifacts should be saved.
        save_artifacts: Whether to save artifacts to disk.

    Returns:
        Feature dataframe ready for model training.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_df = create_base_feature_frame(df)

    # Store medians for API-time missing value handling.
    numeric_medians = {}

    for column in NUMERIC_STANDARD_COLUMNS + NUMERIC_MINMAX_COLUMNS:
        median_value = base_df[column].median()
        numeric_medians[column] = float(median_value)
        base_df[column] = base_df[column].fillna(median_value)

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_scaled = standard_scaler.fit_transform(base_df[NUMERIC_STANDARD_COLUMNS])
    minmax_scaled = minmax_scaler.fit_transform(base_df[NUMERIC_MINMAX_COLUMNS])

    features = pd.DataFrame(index=base_df.index)

    for index, column in enumerate(NUMERIC_STANDARD_COLUMNS):
        features[f"{column}_scaled"] = standard_scaled[:, index]

    for index, column in enumerate(NUMERIC_MINMAX_COLUMNS):
        features[f"{column}_scaled"] = minmax_scaled[:, index]

    # Keep release_month as a simple numeric feature.
    features["release_month"] = base_df["release_month"].fillna(
        base_df["release_month"].median()
    )

    # Binary columns.
    for column in BINARY_COLUMNS:
        features[column] = base_df[column]

    # Fit category lists from training data only.
    genres = get_all_values(base_df, "genres")
    top_100_companies = get_top_values(base_df, "production_companies", top_n=100)
    top_100_keywords = get_top_values(base_df, "keywords", top_n=100)
    original_languages = sorted(
        {
            normalize_text(value)
            for value in base_df["original_language"].dropna()
        }
    ) if "original_language" in base_df.columns else []

    spoken_languages = get_top_values(base_df, "spoken_languages", top_n=50)
    production_countries = get_top_values(base_df, "production_countries", top_n=50)

    # Add encoded categorical features.
    features = add_multi_hot_features(
        features,
        base_df,
        source_column="genres",
        categories=genres,
        prefix="genre",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="production_companies",
        categories=top_100_companies,
        prefix="company",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="keywords",
        categories=top_100_keywords,
        prefix="keyword",
    )

    features = add_single_value_one_hot_features(
        features,
        base_df,
        source_column="original_language",
        categories=original_languages,
        prefix="original_language",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="spoken_languages",
        categories=spoken_languages,
        prefix="spoken_language",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="production_countries",
        categories=production_countries,
        prefix="country",
    )

    # Ensure all feature names are strings.
    features.columns = features.columns.astype(str)

    feature_columns = list(features.columns)

    preprocessing_artifacts = {
        "standard_scaler": standard_scaler,
        "minmax_scaler": minmax_scaler,
        "numeric_standard_columns": NUMERIC_STANDARD_COLUMNS,
        "numeric_minmax_columns": NUMERIC_MINMAX_COLUMNS,
        "binary_columns": BINARY_COLUMNS,
        "numeric_medians": numeric_medians,
        "release_month_median": float(base_df["release_month"].median()),
        "genres": genres,
        "top_100_companies": top_100_companies,
        "top_100_keywords": top_100_keywords,
        "original_languages": original_languages,
        "spoken_languages": spoken_languages,
        "production_countries": production_countries,
        "feature_columns": feature_columns,
    }

    if save_artifacts:
        save_feature_artifacts(
            artifacts=preprocessing_artifacts,
            artifacts_dir=artifacts_dir,
        )

    return features


def transform_features(
    df: pd.DataFrame,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> pd.DataFrame:
    """
    Transform new data using previously saved feature engineering artifacts.

    Use this during:
        - model evaluation
        - FastAPI prediction
        - inference on new data

    Important:
        This function does NOT fit anything.
        It only uses the saved scalers, categories, and feature columns.

    Args:
        df: New raw/cleaned dataframe.
        artifacts_dir: Directory containing saved feature artifacts.

    Returns:
        Feature dataframe with the exact same columns used during training.
    """
    artifacts = load_feature_artifacts(artifacts_dir)

    base_df = create_base_feature_frame(df)

    for column in artifacts["numeric_standard_columns"] + artifacts["numeric_minmax_columns"]:
        if column not in base_df.columns:
            base_df[column] = artifacts["numeric_medians"][column]

        base_df[column] = pd.to_numeric(base_df[column], errors="coerce")
        base_df[column] = base_df[column].fillna(artifacts["numeric_medians"][column])

    standard_scaled = artifacts["standard_scaler"].transform(
        base_df[artifacts["numeric_standard_columns"]]
    )

    minmax_scaled = artifacts["minmax_scaler"].transform(
        base_df[artifacts["numeric_minmax_columns"]]
    )

    features = pd.DataFrame(index=base_df.index)

    for index, column in enumerate(artifacts["numeric_standard_columns"]):
        features[f"{column}_scaled"] = standard_scaled[:, index]

    for index, column in enumerate(artifacts["numeric_minmax_columns"]):
        features[f"{column}_scaled"] = minmax_scaled[:, index]

    features["release_month"] = base_df["release_month"].fillna(
        artifacts["release_month_median"]
    )

    for column in artifacts["binary_columns"]:
        if column not in base_df.columns:
            features[column] = 0
        else:
            features[column] = base_df[column]

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="genres",
        categories=artifacts["genres"],
        prefix="genre",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="production_companies",
        categories=artifacts["top_100_companies"],
        prefix="company",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="keywords",
        categories=artifacts["top_100_keywords"],
        prefix="keyword",
    )

    features = add_single_value_one_hot_features(
        features,
        base_df,
        source_column="original_language",
        categories=artifacts["original_languages"],
        prefix="original_language",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="spoken_languages",
        categories=artifacts["spoken_languages"],
        prefix="spoken_language",
    )

    features = add_multi_hot_features(
        features,
        base_df,
        source_column="production_countries",
        categories=artifacts["production_countries"],
        prefix="country",
    )

    # Force exact training columns and order.
    feature_columns = artifacts["feature_columns"]

    for column in feature_columns:
        if column not in features.columns:
            features[column] = 0

    features = features[feature_columns]

    return features


def save_feature_artifacts(
    artifacts: dict[str, Any],
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """
    Save feature engineering artifacts.

    Saves:
        models/feature_columns.json
        models/top_100_companies.json
        models/top_100_keywords.json
        models/preprocessing.joblib
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as file:
        json.dump(artifacts["feature_columns"], file, indent=2)

    with open(artifacts_dir / TOP_COMPANIES_FILE, "w", encoding="utf-8") as file:
        json.dump(artifacts["top_100_companies"], file, indent=2)

    with open(artifacts_dir / TOP_KEYWORDS_FILE, "w", encoding="utf-8") as file:
        json.dump(artifacts["top_100_keywords"], file, indent=2)

    joblib.dump(artifacts, artifacts_dir / PREPROCESSING_FILE)


def load_feature_artifacts(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    """
    Load saved feature engineering artifacts.
    """
    artifacts_dir = Path(artifacts_dir)
    preprocessing_path = artifacts_dir / PREPROCESSING_FILE

    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Feature preprocessing artifacts not found at: {preprocessing_path}. "
            "Run training first with fit_transform_features()."
        )

    return joblib.load(preprocessing_path)


if __name__ == "__main__":
    from src.data import load_and_clean_data

    data_path = Path("data/raw/TMDB_movie_dataset_v11.csv")

    df = load_and_clean_data(data_path)
    X = fit_transform_features(df, artifacts_dir=DEFAULT_ARTIFACTS_DIR)

    print("Feature matrix shape:", X.shape)
    print("Saved feature artifacts to:", DEFAULT_ARTIFACTS_DIR)
    print("First 10 feature columns:")
    print(X.columns[:10].tolist())