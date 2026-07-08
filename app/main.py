"""
FastAPI application for the Movie Revenue Prediction MLOps Pipeline.

Run locally:

    uvicorn app.main:app --reload

Open:

    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    FEATURE_COLUMNS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSING_PATH,
)
from src.predict import predict_movie_revenue


app = FastAPI(
    title="Movie Revenue Prediction API",
    description=(
        "A FastAPI service that predicts whether a movie is likely to achieve "
        "high revenue based on structured TMDB-style metadata."
    ),
    version="1.0.0",
)


class MovieInput(BaseModel):
    budget: float = Field(..., ge=0, example=50000000)
    runtime: float = Field(..., gt=0, example=120)
    release_date: str = Field(..., example="2026-07-10")
    adult: bool = Field(False, example=False)
    original_language: str = Field("en", example="en")

    genres: list[str] = Field(
        default_factory=list,
        example=["action", "adventure", "science fiction"],
    )
    production_companies: list[str] = Field(
        default_factory=list,
        example=["warner bros. pictures"],
    )
    spoken_languages: list[str] = Field(
        default_factory=list,
        example=["english"],
    )
    keywords: list[str] = Field(
        default_factory=list,
        example=["superhero", "based on comic"],
    )
    production_countries: list[str] = Field(
        default_factory=list,
        example=["united states of america"],
    )


class PredictionOutput(BaseModel):
    prediction: str
    prediction_class: int
    high_revenue_probability: float
    target_definition: str
    model: str
    feature_count: int


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Health check endpoint.
    """
    artifacts = {
        "model": MODEL_PATH.exists(),
        "preprocessing": PREPROCESSING_PATH.exists(),
        "feature_columns": FEATURE_COLUMNS_PATH.exists(),
        "metadata": METADATA_PATH.exists(),
    }

    is_ready = all(artifacts.values())

    return {
        "status": "healthy" if is_ready else "missing_artifacts",
        "artifacts": artifacts,
        "message": (
            "API is ready for prediction."
            if is_ready
            else "Some model artifacts are missing. Run: python -m src.train"
        ),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(movie: MovieInput) -> dict[str, Any]:
    """
    Predict whether a movie is likely to achieve high revenue.
    """
    try:
        return predict_movie_revenue(movie.model_dump())

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error