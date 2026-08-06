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

