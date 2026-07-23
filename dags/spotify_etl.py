import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import hashlib

# We use the internal docker network address for Postgres
# since this will run inside the Airflow container.
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"

def extract_spotify_data():
    """
    Day 3: Extract
    Reads the raw Spotify dataset from the mounted data volume
    and loads it into the `spotify.tracks_raw` staging table.
    """
    file_path = "/opt/airflow/data/raw/spotify/dataset.csv"
    print(f"Reading dataset from {file_path}")
    
    # Read the dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from CSV.")
    
    # Load into tracks_raw staging table
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        # We use if_exists='replace' for the raw table to ensure idempotency (reruns are safe)
        df.to_sql(
            name='tracks_raw',
            schema='spotify',
            con=conn,
            if_exists='replace',
            index=False
        )
    print("Successfully loaded raw data into spotify.tracks_raw")


def transform_and_load_spotify_data():
    """
    Day 4: Transform & Load
    Reads from `spotify.tracks_raw`, cleans data, derives features,
    and loads into `spotify.tracks_clean`.
