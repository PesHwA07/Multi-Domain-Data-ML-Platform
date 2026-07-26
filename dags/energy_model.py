import pandas as pd
from sqlalchemy import create_engine

# Database connection URL for the Airflow container environment
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"

def preprocess_energy_data():
    """
    Day 9: Time-Series Preprocessing
    Fetches raw readings, handles missing values, resamples to ensure 
    continuous hourly frequency, and performs a sequential train/test split.
    """
    engine = create_engine(DB_URL)
    
    print("Reading from energy.hourly_readings...")
    df = pd.read_sql("SELECT * FROM energy.hourly_readings ORDER BY timestamp", engine)
    print(f"Loaded {len(df)} initial rows.")
    
    # Set timestamp as index for time-series operations
    df.set_index('timestamp', inplace=True)
    
    # 1. Resample to strict hourly frequency to explicitly expose missing hours
    # 'h' enforces hourly frequency, .mean() handles any potential duplicates within an hour
    print("Resampling to strict hourly frequency...")
    df = df.resample('h').mean()
    
    # 2. Handle missing values via time-based interpolation
    missing_before = df['consumption'].isna().sum()
    print(f"Missing hours detected before interpolation: {missing_before}")
    
    df['consumption'] = df['consumption'].interpolate(method='time')
    
    missing_after = df['consumption'].isna().sum()
    print(f"Missing hours after interpolation: {missing_after}")
    
    # 3. Sequential Train/Test Split (80/20)
    # CRITICAL: Strict chronological split, NO random shuffling to prevent data leakage
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Sequential Split -> Train size: {len(train_df)} rows, Test size: {len(test_df)} rows")
    
    # Returning the dataframes to be passed to the modeling functions (Prophet/ARIMA)
    return train_df, test_df

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_arima_baseline():
    """
    Day 10: Baseline Modeling
    Trains an ARIMA baseline model on the training set and evaluates it.
    """
    train_df, test_df = preprocess_energy_data()
    
    print("Training ARIMA baseline model (order=1,1,1)...")
    # We use a simple order for the baseline.
    model = ARIMA(train_df['consumption'].values, order=(1, 1, 1))
    fitted_model = model.fit()
    
    print("Generating forecasts for the test set...")
    predictions = fitted_model.forecast(steps=len(test_df))
    
    # Calculate Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(test_df['consumption'].values, predictions))
    mae = mean_absolute_error(test_df['consumption'].values, predictions)
    
    print(f"--- ARIMA Baseline Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"------------------------------")
    
    return fitted_model, predictions

from prophet import Prophet

def train_prophet_model():
    """
    Day 11: Prophet Modeling
    Trains a Facebook Prophet model, evaluates it against the test set,
    and calculates final RMSE/MAE metrics.
    """
    train_df, test_df = preprocess_energy_data()
    
    # Prophet strictly requires columns named 'ds' (datestamp) and 'y' (target)
    # Our data currently has 'timestamp' as the index and 'consumption' as the column
    prophet_train = train_df.reset_index().rename(columns={'timestamp': 'ds', 'consumption': 'y'})
    prophet_test = test_df.reset_index().rename(columns={'timestamp': 'ds', 'consumption': 'y'})
    
    print("Training Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(prophet_train)
    
    print("Generating forecasts for the test set...")
    forecast = model.predict(prophet_test[['ds']])
    
    # Calculate Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(prophet_test['y'].values, forecast['yhat'].values))
    mae = mean_absolute_error(prophet_test['y'].values, forecast['yhat'].values)
    
    print(f"--- Prophet Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"-----------------------")
    
    return model, forecast, prophet_test

from sqlalchemy import text

def evaluate_and_store(forecast, actual_df):
    """
    Day 12: Evaluation & Storage
    Flags anomalies where actual consumption falls outside Prophet's confidence bands,
    and stores the final forecast data into the energy.forecasts PostgreSQL table.
    """
    # Merge the forecast with the actual test data on the timestamp ('ds')
    merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                      actual_df[['ds', 'y']], 
                      on='ds', 
                      how='inner')
    
    # An anomaly occurs if actual 'y' is > upper_band or < lower_band
    merged['anomaly_flag'] = (merged['y'] > merged['yhat_upper']) | (merged['y'] < merged['yhat_lower'])
    
    print(f"Detected {merged['anomaly_flag'].sum()} anomalies out of {len(merged)} test hours.")
    
    # Prepare the DataFrame for SQL insertion according to the schema
    db_df = pd.DataFrame({
        'forecast_timestamp': merged['ds'],
        'predicted_consumption': merged['yhat'],
        'lower_band': merged['yhat_lower'],
        'upper_band': merged['yhat_upper'],
        'model_version': 'Prophet-v1.0',
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

if __name__ == "__main__":
    # Note: If running locally, DB_URL must point to localhost.
    # train_arima_baseline()
    # model, forecast, test_df = train_prophet_model()
    # evaluate_and_store(forecast, test_df)
    pass
