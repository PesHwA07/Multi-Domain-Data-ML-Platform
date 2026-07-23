# Multi-Domain Data & ML Platform

*One shared architecture, three different data patterns — batch ETL, time-series forecasting, real-time serving.*

## Architecture

```text
                     ┌─────────────────────────┐
                     │   Airflow (orchestrator)  │
                     └───────────┬─────────────┘
                                 │
      ┌──────────────────┬──────┴───────────┬──────────────────┐
      │                  │                  │                  │
┌─────▼─────┐     ┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
│ Spotify   │     │  PJM Energy │   │ Credit Card  │   │  Shared      │
│ ETL DAG   │     │  Forecast   │   │ Fraud Train  │   │  PostgreSQL  │
│ (batch,   │     │  DAG (train │   │ DAG (retrain │◄──┤  (all 3      │
│ daily)    │     │  Prophet,   │   │  classifier) │   │  schemas)    │
└───────────┘     │  weekly)    │   └──────────────┘   └──────┬───────┘
                   └──────────────┘                             │
                                                          ┌──────▼───────┐
                                                          │  FastAPI     │
                                                          │  /forecast/  │
                                                          │  /predict/   │
                                                          └──────┬───────┘
                                                                 │
                                                          ┌──────▼───────┐
                                                          │  Streamlit   │
                                                          │  monitoring  │
                                                          │  dashboard   │
                                                          └──────────────┘
```

One orchestrator, one database (three schemas), one serving layer, one dashboard — three genuinely different data engineering patterns running through it.

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | Apache Airflow (Docker) |
| Storage | PostgreSQL (Docker) |
| Forecasting model | Prophet (or ARIMA as baseline) |
| Fraud model | Random Forest / Logistic Regression + `imbalanced-learn` (SMOTE) |
| Serving | FastAPI |
| Monitoring dashboard | Streamlit |
| Containerization | Docker Compose |

## Dataset Setup & API Keys

To keep the repository lightweight and comply with GitHub's file size limits, **raw datasets are not uploaded to this repository.**

Instead, the platform uses `kagglehub` in `scripts/fetch_data.py` to pull the datasets directly. Because these datasets are hosted on Kaggle, **users must configure their Kaggle API keys** to download them successfully.

### How to configure your Kaggle API Key:

1. **Get your API Key**:
   - Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
   - Scroll down to the **API** section and click **Create New API Token**.
   - This will download a `kaggle.json` file containing your credentials.

2. **Configure your local environment**:
   You can authenticate in one of two ways:

   **Option A: Environment Variables (Recommended for Docker)**
   Export the keys in your terminal before running the fetch script:
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_secret_key"
   ```

   **Option B: File placement**
   Place the downloaded `kaggle.json` file in the default Kaggle configuration directory on your machine:
   - Mac/Linux: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`

3. **Fetch the data**:
   Once your keys are configured, simply run the fetch script from the root of the project:
   ```bash
   python scripts/fetch_data.py
   ```
   This will securely download the public datasets and place them into the `data/raw/` directory, which is ignored by Git but mounted to the Airflow container.

## Running the Platform

1. Ensure Docker Desktop is running.
2. Spin up the cluster:
   ```bash
   docker-compose up -d
   ```
3. Access the Airflow UI at `http://localhost:8080` to trigger your DAGs.

## What this demonstrates
This repository demonstrates a complete data engineering lifecycle:
> "Built a multi-domain data platform (Airflow + PostgreSQL + FastAPI) handling three distinct patterns: batch ETL (Spotify track data, scheduled daily), time-series forecasting (energy demand via Prophet, RMSE/MAE-evaluated, weekly retraining), and real-time fraud classification (imbalanced-class handling via SMOTE, precision/recall/PR-AUC evaluated) — served through one shared API and monitored via a unified dashboard."
