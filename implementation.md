# Multi-Domain Data & ML Platform: Master Implementation Document

This document serves as the single source of truth for the project, combining the Product Requirements Document, Technical Specifications, Application Flow, Design Guidelines, Database Schema, and the Day-by-Day Implementation Plan.

---

# 1. Product Requirements Document (PRD)

## 1.1 Project Overview
The Multi-Domain Data & ML Platform is a unified system designed to handle three distinct data engineering patterns: batch ETL, time-series forecasting, and real-time serving. 

## 1.2 Goals & Objectives
- Combine three data domains (Spotify ETL, PJM energy forecasting, credit card fraud detection) into one cohesive platform.
- Build a robust infrastructure to support batch processing, periodic ML retraining, and real-time inference.
- Demonstrate advanced SQL capabilities, time-series modeling, and handling of imbalanced datasets.

## 1.3 Features
1. **Batch ETL (Spotify Data)**: Daily extraction, cleaning, and loading of track data into a structured schema.
2. **Time-Series Forecasting (PJM Energy)**: Weekly ingestion and forecasting (using Prophet/ARIMA) of energy consumption.
3. **Real-time Fraud Classification**: Retraining on imbalanced credit card data (SMOTE) and a real-time prediction endpoint.
4. **Unified Dashboard**: A Streamlit interface to monitor data freshness, visualize analytical queries, track model metrics, and monitor API logs.
5. **REST API**: A FastAPI service providing endpoints for fraud prediction and energy forecasting lookups.

## 1.4 Non-Functional Requirements
- **Performance**: API endpoints must respond in real-time. 
- **Scalability**: Architecture must be containerized and deployable to cloud services (AWS/GCP).
- **Monitoring**: Centralized tracking for DAG runs, data quality, and model performance.

---

# 2. Technical Specifications (TechSpecs)

## 2.1 Technology Stack
- **Orchestration**: Apache Airflow (Docker)
- **Database**: PostgreSQL (Docker) - Single database, multiple schemas.
- **Forecasting Model**: Prophet (Primary) and ARIMA (Baseline)
- **Classification Model**: Random Forest / Logistic Regression with `imbalanced-learn` (SMOTE)
- **Serving Layer**: FastAPI
- **Dashboard**: Streamlit
- **Experiment Tracking**: Weights & Biases (Optional)
- **Infrastructure**: Docker Compose

## 2.2 System Architecture
- **Airflow**: Manages 3 separate DAGs (Spotify Batch, Energy Train/Forecast, Fraud Retrain).
- **PostgreSQL**: Central storage layer acting as the Data Warehouse and Model Registry data store.
- **FastAPI**: Pulls forecasts and runs fraud predictions based on the latest models/data in PostgreSQL.
- **Streamlit**: Queries PostgreSQL directly to visualize metrics and logs.

## 2.3 Data Flow
- **Ingestion**: Airflow DAGs pull from CSVs/APIs into Raw schemas.
- **Transformation**: Python/Pandas logic orchestrates cleaning and writes to Clean/Serving schemas.
- **Inference**: FastAPI receives HTTP requests, processes through models loaded in memory (or pulls from DB), and logs to DB.

---

# 3. Application Flow

## 3.1 ETL Workflow (Spotify)
1. **Trigger**: Scheduled daily Airflow DAG.
2. **Extract**: Load raw Spotify tracks CSV.
3. **Transform**: Clean nulls, normalize metrics, deduplicate, derive new features.
4. **Load**: Insert transformed data into `spotify.tracks_clean`.

## 3.2 Forecasting Workflow (Energy)
1. **Trigger**: Scheduled weekly Airflow DAG.
2. **Extract**: Fetch latest PJM hourly data.
3. **Transform**: Handle missing hours, resample, split train/test sequentially.
4. **Train**: Fit Prophet & ARIMA models.
5. **Evaluate & Load**: Calculate RMSE/MAE, flag anomalies, insert forecasts into `energy.forecasts`.

## 3.3 Fraud Detection Workflow (Credit Card)
1. **Trigger**: Periodic/weekly Airflow DAG.
2. **Extract**: Load new credit card transactions.
3. **Train**: Apply SMOTE for imbalance, train Random Forest/Logistic Regression.
4. **Evaluate & Load**: Calculate Precision/Recall/F1/PR-AUC, save model artifact/metrics.

## 3.4 Serving & Dashboard Flow
1. **Client Request**: User hits `/predict/fraud` (POST) or `/forecast/energy` (GET).
2. **API Processing**: FastAPI computes prediction or fetches forecast.
3. **Logging**: Request, response, and latency logged to Postgres.
4. **Dashboard View**: Streamlit queries Postgres to display real-time analytics, model metrics, and ETL freshness.

---

# 4. Design Guidelines

## 4.1 Architecture Principles
- **Separation of Concerns**: Each domain (Spotify, Energy, Fraud) maintains isolated schemas within the shared database.
- **Idempotency**: All Airflow DAGs must be idempotent, allowing safe reruns on failure.
- **Reproducibility**: Containerized via `docker-compose` to ensure local environments mirror production.

