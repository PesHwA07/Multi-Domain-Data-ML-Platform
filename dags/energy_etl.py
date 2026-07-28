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


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'energy_etl_pipeline',
    default_args=default_args,
    description='A daily DAG to ingest raw PJM Energy data into PostgreSQL',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['energy', 'etl'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_energy_data',
        python_callable=ingest_energy_data,
    )
