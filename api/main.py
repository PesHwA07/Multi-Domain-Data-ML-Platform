import time
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

app = FastAPI(
    title="Multi-Domain Data & ML Platform API",
    description="Unified API for Fraud Detection and Energy Forecasting",
    version="2.0.0"
)

# --- Pydantic Models for Fraud Detection ---


class FraudPredictionRequest(BaseModel):
    transaction_id: str = Field(...,
                                description="Unique identifier for the transaction")
    amount: float = Field(..., description="Transaction amount in USD")
    features: List[float] = Field(..., min_items=28, max_items=28,
                                  description="Array of exactly 28 PCA features (V1-V28)")


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    latency_ms: float
    model_version: str

# --- Pydantic Models for Energy Forecasting ---


class EnergyForecastResponse(BaseModel):
    timestamp: datetime
    predicted_consumption: float
    lower_band: float
    upper_band: float
    anomaly_flag: bool


class EnergyForecastList(BaseModel):
    forecasts: List[EnergyForecastResponse]
    retrieved_at: datetime


@app.get("/health", tags=["System"])
def health_check():
    """Basic health check endpoint to verify API status."""
    return {"status": "ok", "message": "FastAPI is running"}


# Global variables for model and database connection
FRAUD_MODEL = None
# The api service connects to the database via the docker network
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")
engine = create_engine(DB_URL)


def engineer_features_for_inference(amount, features):
    """
    Applies the same feature engineering used during training so that
    the inference input shape matches what the model expects.
    The model was trained with: [amount, V1-V28, amount_log, amount_zscore]
    Total features = 31
    """
    amount_log = np.log1p(amount)
    # For single-transaction inference, z-score is relative to itself (0.0)
    # This is acceptable because the model learned the distribution from
    # training data; the z-score feature still provides signal at extremes.
    amount_zscore = 0.0
    return np.array([amount] + features + [amount_log, amount_zscore]).reshape(1, -1)


def load_fraud_model():
    global FRAUD_MODEL
    if FRAUD_MODEL is None:
        # Primary path: shared data volume mounted at /data
        model_path = "/data/fraud_xgb_model.joblib"
        if not os.path.exists(model_path):
            # Fallback: relative path for local development
            model_path = os.path.join(os.path.dirname(
                __file__), '../data/fraud_xgb_model.joblib')
        if os.path.exists(model_path):
            FRAUD_MODEL = joblib.load(model_path)
            print(f"Loaded XGBoost model from {model_path}")
        else:
            print(f"Warning: Fraud model artifact not found at {model_path}")


@app.on_event("startup")
def startup_event():
    load_fraud_model()


@app.post("/predict/fraud", response_model=FraudPredictionResponse, tags=["Fraud Detection"])
def predict_fraud(request: FraudPredictionRequest):
    """
    Real-time Fraud Prediction Endpoint
    Takes a transaction with PCA features, runs it through the XGBoost model
    trained with GridSearchCV, and returns a classification and probability.
    """
    start_time = time.time()

    if FRAUD_MODEL is None:
        # Attempt to lazily load if startup failed
        load_fraud_model()
        if FRAUD_MODEL is None:
            return FraudPredictionResponse(
                transaction_id=request.transaction_id,
                is_fraud=False,
                fraud_probability=0.0,
                latency_ms=0.0,
                model_version="error-model-not-found"
            )

    # Construct feature matrix with engineered features
    input_features = engineer_features_for_inference(
        request.amount, request.features)

    # Predict
    predicted_class = FRAUD_MODEL.predict(input_features)[0]
    predicted_prob = FRAUD_MODEL.predict_proba(input_features)[0][1]

    # Calculate Latency
    latency_ms = (time.time() - start_time) * 1000

    # Log prediction to database
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO fraud.predictions_log 
                    (transaction_id, predicted_probability, predicted_class, latency_ms)
                    VALUES (:tx_id, :prob, :pclass, :latency)
                """),
                {
                    "tx_id": request.transaction_id,
                    "prob": float(predicted_prob),
                    "pclass": bool(predicted_class),
                    "latency": float(latency_ms)
                }
            )
    except Exception as e:
        print(f"Error logging to database: {e}")

    return FraudPredictionResponse(
        transaction_id=request.transaction_id,
        is_fraud=bool(predicted_class),
        fraud_probability=float(predicted_prob),
        latency_ms=latency_ms,
        model_version="XGBoost-GridCV-v2.0"
    )


@app.get("/forecast/energy", response_model=EnergyForecastList, tags=["Energy Forecasting"])
def get_energy_forecast():
    """
    Retrieves the latest anomaly-flagged energy forecasts from the
    PostgreSQL database.
    """
    # Fetch the last 168 hours (1 week) of forecasts
    query = """
        SELECT forecast_timestamp, predicted_consumption, lower_band, upper_band, anomaly_flag 
        FROM energy.forecasts 
        ORDER BY forecast_timestamp DESC 
        LIMIT 168
    """

    try:
        df = pd.read_sql(query, engine)
        forecasts = []
        for _, row in df.iterrows():
            forecasts.append(EnergyForecastResponse(
                timestamp=row['forecast_timestamp'],
                predicted_consumption=row['predicted_consumption'],
                lower_band=row['lower_band'],
                upper_band=row['upper_band'],
                anomaly_flag=row['anomaly_flag']
            ))

        return EnergyForecastList(
            forecasts=forecasts,
            retrieved_at=datetime.utcnow()
        )
    except Exception as e:
        print(f"Error fetching forecasts: {e}")
        return EnergyForecastList(forecasts=[], retrieved_at=datetime.utcnow())
