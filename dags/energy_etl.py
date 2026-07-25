import pandas as pd
from sqlalchemy import create_engine, text

# We use the internal docker network address for Postgres
# since this will run inside the Airflow container.
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"

def ingest_energy_data():
    """
    Day 8: Data Ingestion
    Reads the raw PJM Energy dataset and loads it into the 
    `energy.hourly_readings` table.
    """
    file_path = "/opt/airflow/data/raw/energy/PJME_hourly.csv"
    print(f"Reading dataset from {file_path}")
    
    # Read the dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} raw rows from CSV.")
    
    # Rename columns to match the database schema
    df = df.rename(columns={
        'Datetime': 'timestamp',
        'PJME_MW': 'consumption'
    })
    
    # Ensure timestamp is parsed as datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically
    df = df.sort_values('timestamp')
    
    # The dataset might have duplicates, which would violate the PRIMARY KEY
    df = df.drop_duplicates(subset=['timestamp'])
    print(f"After deduplication, {len(df)} rows remain.")
    
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        # Clear existing data in case of rerun (idempotent)
        conn.execute(text("TRUNCATE TABLE energy.hourly_readings;"))
        
        # Load into hourly_readings
        df.to_sql(
            name='hourly_readings',
            schema='energy',
            con=conn,
            if_exists='append',
            index=False
        )
    print("Successfully loaded PJM Energy data into energy.hourly_readings")

if __name__ == "__main__":
    # Note: If running this directly on your host machine, you will need to change 
    # file_path to './data/raw/energy/PJME_hourly.csv' and DB_URL to localhost.
    # This is designed to be executed by Airflow in the Docker container.
    
    # ingest_energy_data()
    pass
