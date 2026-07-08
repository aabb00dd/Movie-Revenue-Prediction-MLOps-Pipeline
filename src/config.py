"""
Central configuration for paths used by training, prediction, API, and tests.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "TMDB_movie_dataset_v11.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "revenue_model.joblib"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
TOP_COMPANIES_PATH = MODELS_DIR / "top_100_companies.json"
TOP_KEYWORDS_PATH = MODELS_DIR / "top_100_keywords.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

TARGET_DEFINITION = "Revenue above median revenue in the training dataset"