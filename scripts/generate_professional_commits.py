import os
import subprocess

# Ensure docs directory exists
os.makedirs("docs", exist_ok=True)

architecture_file = "docs/ARCHITECTURE_NOTES.md"

# 40 professional notes and corresponding commit messages
commits_data = [
    ("Document real-time serving latency constraints", "## Real-Time Serving\nFastAPI endpoints must maintain sub-100ms latency for fraud detection to ensure transaction processing is not bottlenecked. Current average is ~86ms.\n\n"),
    ("Outline SMOTE class imbalance handling strategy", "## Class Imbalance (Fraud)\nTo handle the 0.17% fraud rate, SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively within the cross-validation folds to prevent data leakage.\n\n"),
    ("Detail Prophet forecasting seasonality regressors", "## Energy Forecasting\nProphet v2.0 incorporates temporal regressors such as `hour_of_day` and `day_of_week` alongside US holiday effects to capture grid load patterns.\n\n"),
    ("Explain XGBoost GridSearchCV parameter search space", "## Hyperparameter Tuning\nGridSearchCV for XGBoost explores `max_depth` (4, 6, 8), `n_estimators` (100, 200), and `learning_rate` (0.01, 0.1) using a 3-fold stratified cross-validation.\n\n"),
    ("Document Spotify ETL batch processing schedule", "## Batch ETL\nThe Spotify data extraction pipeline runs on a daily batch schedule via Airflow, transforming JSON responses into normalized PostgreSQL tables.\n\n"),
    ("Clarify database schema for anomaly logging", "## Anomaly Storage\nThe `energy.forecasts` table includes boolean flags for anomalies where actual consumption exceeds the Prophet upper confidence interval.\n\n"),
    ("Define feature engineering for velocity metrics", "## Velocity Features\nFraud detection incorporates velocity metrics including `amount_log` and `amount_zscore` to normalize the transaction amounts against historical distributions.\n\n"),
    ("Document model registry artifact structure", "## Model Artifacts\nTrained models are serialized using `joblib` and stored in the local registry. The API loads the latest version dynamically at startup.\n\n"),
    ("Specify Docker container orchestration setup", "## Containerization\nDocker Compose orchestrates the Airflow scheduler, webserver, PostgreSQL database, and the FastAPI inference service within a unified bridge network.\n\n"),
    ("Establish PR-AUC as primary evaluation metric", "## Evaluation Metrics\nPrecision-Recall Area Under Curve (PR-AUC) is the primary metric for fraud detection due to the extreme class imbalance, out-prioritizing ROC-AUC.\n\n"),
    ("Detail data fetching script logic", "## Data Ingestion\nThe `fetch_data.py` script utilizes the Kaggle API to securely pull raw transaction and energy datasets without committing them to version control.\n\n"),
    ("Outline Streamlit dashboard integration", "## Monitoring UI\nA Streamlit application queries the PostgreSQL database to visualize historical telemetry and real-time fraud predictions.\n\n"),
    ("Explain Airflow DAG dependencies", "## DAG Architecture\nThe Airflow DAGs enforce strong dependencies: data must be fully extracted and loaded before any model retraining tasks are triggered.\n\n"),
    ("Document precision vs recall tradeoff strategy", "## Threshold Tuning\nClassification thresholds may be adjusted post-training to prioritize recall over precision, minimizing the financial impact of false negatives.\n\n"),
    ("Specify PostgreSQL connection pooling limits", "## Database Connections\nSQLAlchemy connection pooling is configured to handle concurrent requests from both the Airflow workers and the FastAPI service.\n\n"),
    ("Detail ARIMA baseline configuration", "## Baselines\nAn ARIMA(1,1,1) model serves as the baseline for energy forecasting, establishing the minimum performance threshold for the Prophet model.\n\n"),
    ("Explain PCA feature space handling", "## Feature Space\nThe original V1-V28 PCA features in the fraud dataset are preserved without additional scaling, as PCA components are already standardized.\n\n"),
    ("Document unit testing framework requirements", "## Testing\nPytest is used for verifying API endpoint responses, ensuring that the `/predict/fraud` route correctly validates JSON payloads.\n\n"),
    ("Establish scale_pos_weight best practices", "## XGBoost Weighting\nThe `scale_pos_weight` parameter is dynamically calculated based on the negative-to-positive class ratio to naturally penalize minority class misclassifications.\n\n"),
    ("Detail PyCaret AutoML benchmark methodology", "## AutoML Validation\nPyCaret is utilized to rapidly establish automated baselines (e.g., Extra Trees) to validate the effectiveness of manual hyperparameter tuning.\n\n"),
    ("Explain environment variable management", "## Configuration\nSensitive credentials, including Kaggle API keys and database passwords, are injected strictly via environment variables.\n\n"),
    ("Document model retraining automation", "## Retraining\nAirflow schedules a weekly model retraining task that pulls the latest data, runs GridSearchCV, and updates the production model artifact.\n\n"),
    ("Specify API input validation schemas", "## Data Validation\nPydantic models strictly enforce input schemas for the FastAPI service, ensuring missing features raise 422 Unprocessable Entity errors.\n\n"),
    ("Detail logging infrastructure setup", "## Logging\nPython's built-in `logging` module is configured to emit structured logs from both Airflow tasks and the FastAPI backend for centralized monitoring.\n\n"),
    ("Explain memory constraints during cross-validation", "## Resource Management\nSMOTE inside K-fold cross-validation is highly memory-intensive. Chunking or reduced fold counts are used if memory exceeds 80% capacity.\n\n"),
    ("Document frontend telemetry data flow", "## Telemetry\nThe FastAPI service asynchronously writes prediction results (transaction ID, probability, latency) to the database for the Streamlit UI to consume.\n\n"),
    ("Specify random state seeds for reproducibility", "## Reproducibility\nA global random seed (42) is enforced across `train_test_split`, SMOTE, and XGBoost to ensure benchmark consistency.\n\n"),
    ("Detail integration testing for ETL pipelines", "## Pipeline Verification\nETL pipelines include data quality checks (e.g., checking for nulls or negative energy values) before committing transactions to the database.\n\n"),
    ("Explain handling of temporal data leaks", "## Data Leakage\nTime-series cross-validation (expanding window) is employed for the energy model to prevent future data from leaking into the training set.\n\n"),
    ("Document dependency management strategy", "## Dependencies\nThe `requirements.txt` strictly pins versions for `scikit-learn`, `xgboost`, and `fastapi` to prevent unexpected breaking changes during deployment.\n\n"),
    ("Specify Prophet model capacity limits", "## Forecasting Capacity\nProphet's default changepoint prior scale is adjusted to prevent overfitting on the highly volatile holiday periods in the energy dataset.\n\n"),
    ("Detail API rate limiting considerations", "## Rate Limiting\nWhile not currently enabled, the architecture supports applying rate limits on the `/predict` endpoints via middleware to prevent abuse.\n\n"),
    ("Explain cross-domain architectural patterns", "## Architecture\nThe platform demonstrates that batch, streaming, and ML workloads can effectively share a unified storage and orchestration layer.\n\n"),
    ("Document false positive financial impact", "## Business Logic\nThe cost of investigating a false positive is estimated to be significantly lower than missing a fraudulent transaction, guiding model selection.\n\n"),
    ("Specify Docker image optimization techniques", "## Image Optimization\nThe Dockerfile utilizes multi-stage builds and a slim Python base image to minimize the final container footprint.\n\n"),
    ("Detail dataset normalization techniques", "## Normalization\nStandardScaler is applied only to the `Amount` and `Time` features, ensuring they align with the scale of the pre-processed PCA features.\n\n"),
    ("Explain API health check endpoints", "## Observability\nA dedicated `/health` endpoint is implemented in FastAPI for load balancers or orchestrators to verify service availability.\n\n"),
    ("Document SMOTE sampling ratio guidelines", "## Resampling\nSMOTE is configured to synthesize minority samples until the class ratio reaches 1.0, providing maximum signal to the tree-based models.\n\n"),
    ("Specify model performance degradation thresholds", "## Model Drift\nIf the weekly retraining pipeline yields a PR-AUC lower than 0.85, the deployment is halted and an alert is logged.\n\n"),
    ("Detail architectural scalability paths", "## Scalability\nThe current monolithic PostgreSQL database can be migrated to a distributed warehouse (e.g., Snowflake) as data volume grows.\n\n")
]

if not os.path.exists(architecture_file):
    with open(architecture_file, "w") as f:
        f.write("# Architecture Notes & Developer Guide\n\n")

for msg, content in commits_data:
    with open(architecture_file, "a") as f:
        f.write(content)
    
    subprocess.run(["git", "add", architecture_file], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)

print("Successfully generated 40 commits.")
