import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import FEATURE_COLUMNS_PATH, MODEL_PATH, PREPROCESSING_PATH


client = TestClient(app)


def artifacts_exist() -> bool:
    return (
        MODEL_PATH.exists()
        and PREPROCESSING_PATH.exists()
        and FEATURE_COLUMNS_PATH.exists()
    )


def test_health_endpoint_works():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "artifacts" in data


def test_predict_endpoint_returns_prediction():
    if not artifacts_exist():
        pytest.skip("Model artifacts missing. Run: python -m src.train")

    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["high_revenue", "low_revenue"]
    assert data["prediction_class"] in [0, 1]
    assert 0.0 <= data["high_revenue_probability"] <= 1.0
    assert data["model"] == "XGBoost"
    assert data["feature_count"] > 0