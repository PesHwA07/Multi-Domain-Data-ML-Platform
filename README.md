# Multi-Domain Data ML Platform

A comprehensive, containerized data engineering and machine learning serving platform that processes multiple distinct datasets (Spotify, Energy, and Fraud). The platform is orchestrated via Apache Airflow, stores transformed data in PostgreSQL, and serves data via a FastAPI backend.

## Architecture

- **PostgreSQL**: Central data warehouse hosting schemas for `spotify`, `energy`, and `fraud`.
- **Apache Airflow**: Orchestrates the ETL pipelines to extract raw datasets, perform Pandas-based transformations, and load them into PostgreSQL.
- **FastAPI**: Serves the transformed data and analytical queries to upstream dashboards.
- **Docker**: The entire stack is containerized for reproducible local development.

---

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

---

## Running the Platform

1. Ensure Docker Desktop is running.
2. Spin up the cluster:
   ```bash
   docker-compose up -d
   ```
3. Access the Airflow UI at `http://localhost:8080` to trigger your DAGs.
