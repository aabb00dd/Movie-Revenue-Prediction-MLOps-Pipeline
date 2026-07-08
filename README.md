# Movie Revenue Prediction MLOps Pipeline

Production-style machine learning pipeline for predicting whether a movie is likely to achieve **high revenue** using structured TMDB-style metadata.

This project upgrades a notebook-based machine learning workflow into a reusable MLOps project with data cleaning, feature engineering, model training, MLflow experiment tracking, FastAPI inference, automated tests, and Docker support.

---

## Overview

The model predicts whether a movie's revenue is above the median revenue in the cleaned dataset.

```text
high_revenue = 1 if revenue > median_revenue
high_revenue = 0 otherwise
```

This is a binary classification task, not exact revenue regression.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- MLflow
- FastAPI
- Pytest
- Docker

---

## Pipeline

```text
Raw TMDB data
→ Data cleaning
→ Feature engineering
→ Train/test split
→ XGBoost training
→ MLflow experiment tracking
→ Saved model artifacts
→ FastAPI prediction endpoint
→ Dockerized API
```

---

## Model Performance

The XGBoost classifier was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.823 |
| F1 Score | 0.825 |
| ROC-AUC | 0.904 |

---

## MLflow Experiment Tracking

The training script logs model parameters, metrics, and artifacts with MLflow.

Logged items include:

- Model type
- Target variable
- Accuracy
- F1 score
- ROC-AUC
- XGBoost hyperparameters
- Train/test size
- Number of features
- Saved model and preprocessing artifacts

![MLflow Metrics](docs/images/mlflow_metrics.png)

---

## FastAPI Prediction API

Run the API locally:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

![FastAPI Docs](docs/images/fastapi_docs.png)

---

## Example Prediction

### Request

```json
{
  "budget": 50000000,
  "runtime": 120,
  "release_date": "2026-07-10",
  "adult": false,
  "original_language": "en",
  "genres": ["action", "adventure", "science fiction"],
  "production_companies": ["warner bros. pictures"],
  "spoken_languages": ["english"],
  "keywords": ["superhero", "based on comic"],
  "production_countries": ["united states of america"]
}
```

### Response

```json
{
  "prediction": "high_revenue",
  "prediction_class": 1,
  "high_revenue_probability": 0.8946,
  "target_definition": "high_revenue = 1 if revenue is above the median revenue in the cleaned dataset, else 0",
  "model": "XGBoost",
  "feature_count": 378
}
```

![Prediction Response](docs/images/prediction_response.png)

---

## Run Training

```bash
python -m src.train
```

This creates the trained model and preprocessing artifacts:

```text
models/revenue_model.joblib
models/feature_columns.json
models/top_100_companies.json
models/top_100_keywords.json
models/preprocessing.joblib
models/model_metadata.json
```

---

## Run MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

---

## Run Tests

```bash
pytest
```

The tests check:

- API health endpoint
- API prediction endpoint
- Feature preprocessing column consistency

---

## Docker

Build the Docker image:

```bash
docker build -t movie-revenue-api .
```

Run the container:

```bash
docker run -p 8000:8000 movie-revenue-api
```

Open:

```text
http://localhost:8000/docs
```

![Docker Image](docs/images/docker_image.png)

---

## Project Structure

```text
app/
  main.py

src/
  config.py
  data.py
  features.py
  predict.py
  train.py

tests/
  test_api.py
  test_features.py

models/
  revenue_model.joblib
  preprocessing.joblib
  feature_columns.json
  model_metadata.json
```

---

## Limitations

- The model predicts high vs low revenue, not exact revenue.
- The target is based on the median revenue of the cleaned dataset.
- Predictions depend on the quality and completeness of movie metadata.
- The dataset may contain noisy, missing, or inconsistent records.

---

## Future Improvements

- Deploy the API publicly
- Add CI/CD with GitHub Actions
- Add model registry support
- Add monitoring for data drift and prediction drift
- Compare additional models and tuned hyperparameters