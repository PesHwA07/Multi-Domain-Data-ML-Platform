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
