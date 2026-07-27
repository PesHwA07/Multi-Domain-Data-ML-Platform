from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

app = FastAPI(
    title="Multi-Domain Data & ML Platform API",
    description="Unified API for Fraud Detection and Energy Forecasting",
    version="1.0.0"
)

# --- Pydantic Models for Fraud Detection ---

class FraudPredictionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    amount: float = Field(..., description="Transaction amount in USD")
    features: List[float] = Field(..., min_items=28, max_items=28, description="Array of exactly 28 PCA features (V1-V28)")

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

# Note: The actual endpoint implementations for /predict/fraud and /forecast/energy 
# will be implemented on Day 20 as per the architectural plan.
