from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import our ETL functions
from spotify_etl import extract_spotify_data, transform_and_load_spotify_data

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'spotify_etl_pipeline',
    default_args=default_args,
    description='Daily ETL pipeline for Spotify Tracks dataset',
    schedule_interval='@daily',
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=['spotify', 'etl', 'batch'],
) as dag:

    extract_task = PythonOperator(
        task_id='extract_raw_data',
        python_callable=extract_spotify_data,
    )

    transform_load_task = PythonOperator(
        task_id='transform_and_load_data',
        python_callable=transform_and_load_spotify_data,
    )

    # Define task dependencies (Extract -> Transform/Load)
    extract_task >> transform_load_task
