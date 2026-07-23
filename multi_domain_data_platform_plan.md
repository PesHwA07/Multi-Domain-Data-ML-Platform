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
