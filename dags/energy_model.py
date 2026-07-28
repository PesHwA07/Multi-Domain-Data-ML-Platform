"""
Energy Modeling and Forecasting DAG (Prophet v2.0 + Holidays + Regressors)
==========================================================================
This module handles the extraction, preprocessing, baseline modeling (ARIMA),
advanced modeling (Prophet with US holidays and temporal regressors), and
anomaly detection for PJM energy data. It is orchestrated as a weekly
Airflow DAG.
"""

from airflow import DAG
from sqlalchemy import text
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sqlalchemy import create_engine

# Database connection URL for the Airflow container environment
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"


def preprocess_energy_data():
    """
    Time-Series Preprocessing
    Fetches raw readings, handles missing values, resamples to ensure
    continuous hourly frequency, and performs a sequential train/test split.
    """
    engine = create_engine(DB_URL)

    print("Reading from energy.hourly_readings...")
    df = pd.read_sql(
        "SELECT * FROM energy.hourly_readings ORDER BY timestamp", engine)
    print(f"Loaded {len(df)} initial rows.")

    # Set timestamp as index for time-series operations
    df.set_index('timestamp', inplace=True)

    # 1. Resample to strict hourly frequency to explicitly expose missing hours
    print("Resampling to strict hourly frequency...")
    df = df.resample('h').mean()

    # 2. Handle missing values via time-based interpolation
    missing_before = df['consumption'].isna().sum()
    print(f"Missing hours detected before interpolation: {missing_before}")

    df['consumption'] = df['consumption'].interpolate(method='time')

    missing_after = df['consumption'].isna().sum()
    print(f"Missing hours after interpolation: {missing_after}")

    # 3. Sequential Train/Test Split (80/20)
    # CRITICAL: Strict chronological split, NO random shuffling
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(
        f"Sequential Split -> Train size: {len(train_df)} rows, "
        f"Test size: {len(test_df)} rows")

    return train_df, test_df


def add_temporal_regressors(df):
    """
    Feature Engineering: Adds temporal regressors for Prophet.
    - hour_of_day: Hour (0-23), captures intra-day demand patterns
    - day_of_week: Day (0=Mon, 6=Sun), captures weekday/weekend patterns

    Parameters:
        df: DataFrame with a 'ds' column (datetime)
    Returns:
        df with two additional regressor columns
    """
    df['hour_of_day'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    return df


def train_arima_baseline():
    """
    Baseline Modeling
    Trains an ARIMA baseline model on the training set and evaluates it.
    """
    train_df, test_df = preprocess_energy_data()

    print("Training ARIMA baseline model (order=1,1,1)...")
    model = ARIMA(train_df['consumption'].values, order=(1, 1, 1))
    fitted_model = model.fit()

    print("Generating forecasts for the test set...")
    predictions = fitted_model.forecast(steps=len(test_df))

    # Calculate Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(
        test_df['consumption'].values, predictions))
    mae = mean_absolute_error(test_df['consumption'].values, predictions)

    print(f"--- ARIMA Baseline Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"------------------------------")

    return fitted_model, predictions


def train_prophet_model():
    """
    Prophet v2.0 Modeling with US Holidays and Temporal Regressors
    Trains a Facebook Prophet model with country-level holiday effects and
    hour/day regressors, evaluates against the test set, and returns the
    forecast with confidence intervals.
    """
    train_df, test_df = preprocess_energy_data()

    # Prophet requires columns named 'ds' (datestamp) and 'y' (target)
    prophet_train = train_df.reset_index().rename(
        columns={'timestamp': 'ds', 'consumption': 'y'})
    prophet_test = test_df.reset_index().rename(
        columns={'timestamp': 'ds', 'consumption': 'y'})

    # Add temporal regressors to both train and test sets
    print("Engineering temporal regressors (hour_of_day, day_of_week)...")
    prophet_train = add_temporal_regressors(prophet_train)
    prophet_test = add_temporal_regressors(prophet_test)

    print("Training Prophet v2.0 model with US holidays and regressors...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )

    # Add US holiday effects (Thanksgiving, Christmas, July 4th, etc.)
    model.add_country_holidays(country_name='US')

    # Add temporal regressors for stronger intra-day/weekly signals
    model.add_regressor('hour_of_day')
    model.add_regressor('day_of_week')

    model.fit(prophet_train)

    print("Generating forecasts for the test set...")
    forecast = model.predict(prophet_test[['ds', 'hour_of_day', 'day_of_week']])

    # Calculate Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(
        prophet_test['y'].values, forecast['yhat'].values))
    mae = mean_absolute_error(
        prophet_test['y'].values, forecast['yhat'].values)

    print(f"--- Prophet v2.0 Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"----------------------------")

    return model, forecast, prophet_test


def evaluate_and_store(forecast, actual_df):
    """
    Evaluation & Storage
    Flags anomalies where actual consumption falls outside Prophet's
    confidence bands, and stores the final forecast data into the
    energy.forecasts PostgreSQL table.
    """
    # Merge the forecast with the actual test data on the timestamp ('ds')
    merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                      actual_df[['ds', 'y']],
                      on='ds',
                      how='inner')

    # An anomaly occurs if actual 'y' is > upper_band or < lower_band
    merged['anomaly_flag'] = (merged['y'] > merged['yhat_upper']) | (
        merged['y'] < merged['yhat_lower'])

    print(
        f"Detected {merged['anomaly_flag'].sum()} anomalies "
        f"out of {len(merged)} test hours.")

    # Prepare the DataFrame for SQL insertion according to the schema
    db_df = pd.DataFrame({
        'forecast_timestamp': merged['ds'],
        'predicted_consumption': merged['yhat'],
        'lower_band': merged['yhat_lower'],
        'upper_band': merged['yhat_upper'],
        'model_version': 'Prophet-v2.0-holidays',
        'anomaly_flag': merged['anomaly_flag']
    })

    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE energy.forecasts;"))
        db_df.to_sql(
            name='forecasts',
            schema='energy',
            con=conn,
            if_exists='append',
            index=False
        )

    print("Successfully saved predictions and anomalies to energy.forecasts.")


def run_energy_forecasting_pipeline():
    """Executes the complete energy forecasting and evaluation pipeline."""
    # Train the baseline to log comparison metrics,
    # then rely on Prophet v2.0 for the actual forecast
    train_arima_baseline()
    model, forecast, test_df = train_prophet_model()
    evaluate_and_store(forecast, test_df)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Wrap the pipeline into an Airflow DAG running on a weekly schedule
with DAG(
    'energy_forecasting_weekly',
    default_args=default_args,
    description='Weekly DAG: Prophet v2.0 energy model with US holidays and temporal regressors',
    schedule_interval='@weekly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['energy', 'ml', 'prophet', 'holidays'],
) as dag:

    forecast_task = PythonOperator(
        task_id='train_and_forecast_energy',
        python_callable=run_energy_forecasting_pipeline,
    )
