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

if __name__ == "__main__":
    # Test the preprocessing logic (requires DB_URL to point to localhost if run outside Docker)
    # train, test = preprocess_energy_data()
    pass
