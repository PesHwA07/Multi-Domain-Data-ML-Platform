# Architecture Notes & Developer Guide

## Real-Time Serving
FastAPI endpoints must maintain sub-100ms latency for fraud detection to ensure transaction processing is not bottlenecked. Current average is ~86ms.

## Class Imbalance (Fraud)
To handle the 0.17% fraud rate, SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively within the cross-validation folds to prevent data leakage.

## Energy Forecasting
Prophet v2.0 incorporates temporal regressors such as `hour_of_day` and `day_of_week` alongside US holiday effects to capture grid load patterns.

## Hyperparameter Tuning
GridSearchCV for XGBoost explores `max_depth` (4, 6, 8), `n_estimators` (100, 200), and `learning_rate` (0.01, 0.1) using a 3-fold stratified cross-validation.

## Batch ETL
The Spotify data extraction pipeline runs on a daily batch schedule via Airflow, transforming JSON responses into normalized PostgreSQL tables.

## Anomaly Storage
The `energy.forecasts` table includes boolean flags for anomalies where actual consumption exceeds the Prophet upper confidence interval.

## Velocity Features
Fraud detection incorporates velocity metrics including `amount_log` and `amount_zscore` to normalize the transaction amounts against historical distributions.

## Model Artifacts
Trained models are serialized using `joblib` and stored in the local registry. The API loads the latest version dynamically at startup.

## Containerization
Docker Compose orchestrates the Airflow scheduler, webserver, PostgreSQL database, and the FastAPI inference service within a unified bridge network.

## Evaluation Metrics
Precision-Recall Area Under Curve (PR-AUC) is the primary metric for fraud detection due to the extreme class imbalance, out-prioritizing ROC-AUC.

## Data Ingestion
The `fetch_data.py` script utilizes the Kaggle API to securely pull raw transaction and energy datasets without committing them to version control.

## Monitoring UI
A Streamlit application queries the PostgreSQL database to visualize historical telemetry and real-time fraud predictions.

## DAG Architecture
The Airflow DAGs enforce strong dependencies: data must be fully extracted and loaded before any model retraining tasks are triggered.

## Threshold Tuning
Classification thresholds may be adjusted post-training to prioritize recall over precision, minimizing the financial impact of false negatives.

## Database Connections
SQLAlchemy connection pooling is configured to handle concurrent requests from both the Airflow workers and the FastAPI service.

## Baselines
An ARIMA(1,1,1) model serves as the baseline for energy forecasting, establishing the minimum performance threshold for the Prophet model.

## Feature Space
The original V1-V28 PCA features in the fraud dataset are preserved without additional scaling, as PCA components are already standardized.

