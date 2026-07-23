# Project: Multi-Domain Data & ML Platform
*One shared architecture, three different data patterns — batch ETL, time-series
forecasting, real-time serving. 4-week focused build.*

## Why combine instead of building three separate projects

The three source ideas (Spotify ETL, PJM energy forecasting, credit card fraud detection)
don't share data, so there's no honest way to merge them into one dataset story. What they
*do* share is infrastructure — and building one platform that handles all three data
patterns is a stronger, more senior-sounding resume line than three disconnected Kaggle
projects, because it's the thing that's actually hard: making one system flexible enough
to serve batch, time-series, and real-time workloads instead of three throwaway scripts.

This also directly closes the two real gaps flagged in your role-skill comparison: SQL
depth (currently "listed, not proven") and time-series (currently a total blank).

---

## Architecture

```
                     ┌─────────────────────────┐
                     │   Airflow (orchestrator)  │
                     └───────────┬─────────────┘
                                 │
      ┌──────────────────┬──────┴───────────┬──────────────────┐
      │                  │                  │                  │
┌─────▼─────┐     ┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
│ Spotify   │     │  PJM Energy  │   │ Credit Card  │   │  Shared      │
│ ETL DAG   │     │  Forecast    │   │ Fraud Train  │   │  PostgreSQL  │
│ (batch,   │     │  DAG (train  │   │ DAG (retrain │◄──┤  (all 3      │
│ daily)    │     │  Prophet,    │   │  classifier) │   │  schemas)    │
└───────────┘     │  weekly)     │   └──────────────┘   └──────┬───────┘
                   └──────────────┘                             │
                                                          ┌──────▼───────┐
                                                          │  FastAPI      │
                                                          │  /forecast/energy
                                                          │  /predict/fraud
                                                          └──────┬───────┘
                                                                 │
                                                          ┌──────▼───────┐
                                                          │  Streamlit    │
                                                          │  monitoring   │
                                                          │  dashboard    │
                                                          └───────────────┘
```

One orchestrator, one database (three schemas), one serving layer, one dashboard —
three genuinely different data engineering patterns running through it.

---

## Tech stack

| Component | Tool | Cost |
|---|---|---|
| Orchestration | Apache Airflow (Docker) | Free, self-hosted |
| Storage | PostgreSQL (Docker) | Free, self-hosted |
| Forecasting model | Prophet (or ARIMA as a baseline comparison) | Free |
| Fraud model | Random Forest / Logistic Regression + `imbalanced-learn` (SMOTE) | Free — reuses your existing classical ML skill from CS:GO |
| Serving | FastAPI | Free — reuses your Week 8 RAG deployment skill |
| Monitoring dashboard | Streamlit | Free — reuses your Week 6 RAG dashboard skill |
| Experiment tracking | Weights & Biases (optional, reuse from RAG project) | Free tier |
| Containerization | Docker Compose (Postgres + Airflow + FastAPI in one stack) | Free |

Notice most of the "serving/dashboard/tracking" row is **skill reuse**, not new learning —
the new material is concentrated in Airflow orchestration and the two new data domains.

---

## Week 1 — Infra + Spotify batch ETL

- [ ] `docker-compose.yml` with Postgres + Airflow (use the official Airflow docker-compose
      template as a base, trim it down)
- [ ] Design schemas: `spotify.tracks_raw`, `spotify.tracks_clean`, `energy.hourly_readings`,
      `energy.forecasts`, `fraud.transactions`, `fraud.predictions`
- [ ] Download Spotify Tracks dataset, write ETL logic:
      - **Extract**: read CSV
      - **Transform**: clean nulls, normalize `danceability`/`energy`/`popularity`,
        dedupe, derive features (e.g., `decade` from release year)
      - **Load**: write to `spotify.tracks_clean` in Postgres
- [ ] Wrap as an Airflow DAG, scheduled daily, with basic retry/failure alerting
- [ ] Write 3–5 analytical SQL queries against the loaded data (e.g., "top genres by
      average energy," "popularity trend by decade") — this is the part that actually
      proves real SQL depth, not just "I loaded data into a table"

**Milestone:** Airflow DAG runs on schedule, Spotify data lands cleaned in Postgres,
analytical queries return sensible results.

---

## Week 2 — Time-series forecasting pipeline

- [ ] Ingest PJM hourly energy consumption dataset into `energy.hourly_readings`
- [ ] Preprocessing: handle missing hours, resample, train/test split respecting time
      order (no random shuffling — a common time-series mistake worth explicitly avoiding
      and mentioning you avoided)
- [ ] Train **Prophet** as your primary model, and a simple **ARIMA** baseline for
      comparison — being able to say "I compared against a classical baseline" is a good
      signal
- [ ] Evaluate with RMSE and MAE, store forecasts + actuals in `energy.forecasts`
- [ ] Wrap training as a second Airflow DAG, scheduled weekly (retrain as new data arrives)
- [ ] Add a simple anomaly flag: if actual deviates from forecast band by more than N%,
      log it — small addition, ties back to the "real-time alerting" angle from the
      original idea without needing a separate system

**Milestone:** forecasting DAG runs weekly, RMSE/MAE logged, forecast-vs-actual data
queryable in Postgres.

---

## Week 3 — Fraud detection + real-time serving

- [ ] Load credit card fraud dataset into `fraud.transactions`
- [ ] Train classifier (Random Forest + Logistic Regression, same comparison approach as
      CS:GO) — this dataset is heavily imbalanced (fraud is <1% of transactions), so
      apply and document **SMOTE or class-weighting**, and evaluate with **precision/recall/
      F1 and PR-AUC**, not accuracy (accuracy is meaningless on this class distribution —
      explicitly noting this in your README shows you understand why)
- [ ] Wrap training as a third Airflow DAG (periodic retrain)
- [ ] Build the shared **FastAPI** service with two endpoints:
      - `POST /predict/fraud` — takes a transaction, returns fraud probability + flag
      - `GET /forecast/energy` — returns latest forecast for a given hour range
- [ ] Log every prediction request (input, output, latency) to Postgres for the
      dashboard to read

**Milestone:** one FastAPI service serves both a real-time classification endpoint and a
forecast-lookup endpoint, backed by models trained via Airflow.

---

## Week 4 — Unified dashboard + polish

- [ ] Streamlit dashboard with three tabs:
      - **Spotify**: data freshness, row counts, a couple of the analytical query results
        as charts
      - **Energy**: forecast vs. actual chart, RMSE/MAE trend over retraining runs
      - **Fraud**: precision/recall/F1 over retraining runs, recent prediction log
- [ ] Add DAG run history (success/failure, duration) to the dashboard — ties the
      orchestration layer to something visible, not just running invisibly in Airflow's UI
- [ ] README: architecture diagram, why one platform instead of three scripts, setup
      instructions, example queries/API calls
- [ ] **Optional stretch, closes the cloud gap from your role analysis:** deploy the
      Postgres + FastAPI portion to one free-tier cloud instance (AWS RDS free tier +
