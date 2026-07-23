from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import our ETL functions
from spotify_etl import extract_spotify_data, transform_and_load_spotify_data

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
