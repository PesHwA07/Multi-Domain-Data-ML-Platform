-- Create Schemas
CREATE SCHEMA IF NOT EXISTS spotify;
CREATE SCHEMA IF NOT EXISTS energy;
CREATE SCHEMA IF NOT EXISTS fraud;

-- ==========================================
-- Schema: spotify
-- ==========================================
CREATE TABLE IF NOT EXISTS spotify.tracks_clean (
    track_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    artist VARCHAR,
    danceability FLOAT,
    energy FLOAT,
    popularity FLOAT,
    decade INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Note: spotify.tracks_raw will be created dynamically by pandas.to_sql during the Airflow DAG to accommodate raw CSV schema.

-- ==========================================
-- Schema: energy
-- ==========================================
CREATE TABLE IF NOT EXISTS energy.hourly_readings (
    timestamp TIMESTAMP PRIMARY KEY,
    consumption FLOAT
);

CREATE TABLE IF NOT EXISTS energy.forecasts (
    forecast_timestamp TIMESTAMP PRIMARY KEY,
    predicted_consumption FLOAT,
    lower_band FLOAT,
    upper_band FLOAT,
    model_version VARCHAR,
    anomaly_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- Schema: fraud
-- ==========================================
CREATE TABLE IF NOT EXISTS fraud.transactions (
    transaction_id VARCHAR PRIMARY KEY,
    amount FLOAT,
    timestamp TIMESTAMP,
    features FLOAT[], -- Array to hold V1-V28 PCA features
