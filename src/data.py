"""
Data loading and cleaning utilities for the Movie Revenue Prediction MLOps Pipeline.

This module converts the notebook-based TMDB cleaning process into reusable
production-style Python functions.

Main output:
    A cleaned dataframe with a binary target column:
        high_revenue = 1 if revenue is above the median revenue, else 0
"""

from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = {
    "status",
    "vote_count",
    "release_date",
    "revenue",
    "budget",
    "runtime",
}


COLUMNS_TO_DROP = [
    # URL/image columns: not useful for structured prediction
    "backdrop_path",
    "homepage",
    "poster_path",

    # Unique identifiers: no predictive value
    "id",
    "imdb_id",

    # Text-heavy columns: kept out of version 1 API/training pipeline
    "tagline",
    "overview",
]


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load the raw TMDB dataset from a CSV file.

    Args:
        data_path: Path to the raw TMDB CSV file.

    Returns:
        Raw pandas DataFrame.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    return pd.read_csv(data_path)


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Check that the required columns exist in the dataset.

    Args:
        df: Input DataFrame.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {sorted(missing_columns)}"
        )


def clean_tmdb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw TMDB dataset.

    Cleaning steps:
        1. Keep only released movies.
        2. Remove rows with zero vote_count.
        3. Drop unusable URL, ID, and text-heavy columns.
        4. Convert release_date to datetime.
        5. Remove rows with missing values.
        6. Remove invalid revenue, budget, and runtime values.
        7. Create high_revenue target based on median revenue.

    Args:
        df: Raw TMDB DataFrame.

    Returns:
        Cleaned DataFrame with high_revenue target.
    """
    validate_required_columns(df)

    df = df.copy()

    # Keep only released movies because unreleased movies do not have reliable performance data.
    df = df[df["status"] == "Released"]

    # Remove rows with unreliable ratings/votes.
    df = df[df["vote_count"] > 0]

    # Drop columns that are not useful for version 1 of the ML pipeline.
    existing_drop_columns = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing_drop_columns)

    # Convert release_date to datetime. Invalid dates become NaT and are removed later.
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # Convert important numeric columns safely.
    numeric_columns = ["revenue", "budget", "runtime", "vote_count"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows with missing values after conversion.
    df = df.dropna()

    # Remove invalid or unusable values.
    # Revenue must be positive because the target is based on revenue.
    df = df[df["revenue"] > 0]

    # Budget and runtime should also be positive for realistic movie metadata.
    df = df[df["budget"] > 0]
    df = df[df["runtime"] > 0]

    # Create target variable.
    revenue_median = df["revenue"].median()
    df["high_revenue"] = (df["revenue"] > revenue_median).astype(int)

    # Save the threshold as metadata inside dataframe attrs.
    # This is useful later for documentation/model metadata.
    df.attrs["revenue_median"] = revenue_median

    return df.reset_index(drop=True)


def load_and_clean_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load and clean the TMDB dataset in one step.

    Args:
        data_path: Path to raw CSV file.

    Returns:
        Cleaned DataFrame with high_revenue target.
    """
    raw_df = load_raw_data(data_path)
    clean_df = clean_tmdb_data(raw_df)

    return clean_df


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = "high_revenue",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split cleaned data into raw input features and target.

    Important:
        revenue is removed from X because it was used to create the target.
        Keeping revenue as an input feature would cause data leakage.

    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.

    Returns:
        X_raw, y
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    leakage_columns = [
        target_column,
        "revenue",
    ]

    X = df.drop(columns=[col for col in leakage_columns if col in df.columns])
    y = df[target_column]

    return X, y


if __name__ == "__main__":
    data_path = Path("data/raw/TMDB_movie_dataset_v11.csv")

    df = load_and_clean_data(data_path)
    X, y = get_feature_target_split(df)

    print("Cleaned dataset shape:", df.shape)
    print("Feature dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=True))
    print("Revenue median threshold:", df.attrs["revenue_median"])