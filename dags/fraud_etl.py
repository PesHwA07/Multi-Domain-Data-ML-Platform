import pandas as pd
import uuid
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Database connection URL for the Airflow container environment
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"


def ingest_fraud_data():
    """
    Day 15: Fraud Data Ingestion
    Loads the credit card fraud CSV in chunks, transforms V1-V28 into a PCA feature array,
    generates UUIDs and proper timestamps, and inserts them into PostgreSQL.
    """
    # Try the Docker path first, fallback to local path
    data_path = "/opt/airflow/data/raw/fraud/creditcard.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(
            __file__), '../data/raw/fraud/creditcard.csv')

    if not os.path.exists(data_path):
        print(
            f"Dataset not found at {data_path}. Please ensure kagglehub fetched it successfully.")
        return

    print(f"Loading data from {data_path}...")
    engine = create_engine(DB_URL)

    # TRUNCATE to ensure idempotency (rerunning doesn't duplicate data)
    print("Truncating existing fraud.transactions table...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fraud.transactions;"))

    # The dataset 'Time' column represents seconds elapsed from the first transaction.
    # We anchor this to an arbitrary base timestamp for database consistency.
    base_time = datetime(2023, 1, 1)
    chunk_size = 50000
    total_inserted = 0

    # Read in chunks to prevent memory blowouts on the Airflow worker
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        # 1. Generate unique transaction IDs
        chunk['transaction_id'] = [str(uuid.uuid4())
                                   for _ in range(len(chunk))]

        # 2. Convert raw seconds into a proper timestamp
        chunk['timestamp'] = chunk['Time'].apply(
            lambda x: base_time + timedelta(seconds=x))

        # 3. Consolidate PCA features (V1-V28) into a single array column for PostgreSQL
        v_cols = [f'V{i}' for i in range(1, 29)]
        chunk['features'] = chunk[v_cols].values.tolist()

        # 4. Map columns directly to the schema requirements
        chunk['amount'] = chunk['Amount']
        chunk['is_fraud'] = chunk['Class'].astype(bool)

        # Filter down to the final target schema
        final_cols = ['transaction_id', 'amount',
                      'timestamp', 'features', 'is_fraud']
        db_df = chunk[final_cols]

        # 5. Insert directly into the Postgres schema
        db_df.to_sql('transactions', schema='fraud', con=engine,
                     if_exists='append', index=False)
        total_inserted += len(db_df)
        print(f"Inserted {total_inserted} rows...")

    print("Fraud data ingestion successfully completed!")


from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'fraud_etl_pipeline',
    default_args=default_args,
    description='A daily DAG to ingest raw credit card fraud data into PostgreSQL',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['fraud', 'etl'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_fraud_data',
        python_callable=ingest_fraud_data,
    )
