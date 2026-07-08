import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS_PATH, MODELS_DIR, PREPROCESSING_PATH
from src.features import load_feature_artifacts, transform_features


def test_feature_preprocessing_returns_same_columns_as_training():
    if not PREPROCESSING_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        pytest.skip("Feature artifacts missing. Run: python -m src.train")

    artifacts = load_feature_artifacts(MODELS_DIR)

    sample = pd.DataFrame(
        [
            {
                "budget": 50000000,
                "runtime": 120,
                "release_date": "2026-07-10",
                "adult": False,
                "original_language": "en",
                "genres": ["action", "adventure", "science fiction"],
                "production_companies": ["warner bros. pictures"],
                "spoken_languages": ["english"],
                "keywords": ["superhero", "based on comic"],
                "production_countries": ["united states of america"],
            }
        ]
    )

    X = transform_features(sample, artifacts_dir=MODELS_DIR)

    assert X.shape[1] == len(artifacts["feature_columns"])
    assert list(X.columns) == artifacts["feature_columns"]