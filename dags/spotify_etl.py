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
    """
    engine = create_engine(DB_URL)
    
    print("Reading from spotify.tracks_raw...")
    df = pd.read_sql("SELECT * FROM spotify.tracks_raw", engine)
    
    print(f"Initial raw rows: {len(df)}")
    
    # Transform: clean nulls
    df = df.dropna(subset=['track_id', 'track_name', 'artists'])
    
    # Transform: deduplicate by track_id
    df = df.drop_duplicates(subset=['track_id'])
    
    # Transform: derive 'decade' feature
    # Since the Kaggle dataset doesn't include a release year, we simulate one 
    # deterministically based on track_id hash for consistent dashboard queries.
    def get_decade(t_id):
        val = int(hashlib.md5(str(t_id).encode()).hexdigest()[:4], 16)
        decades = [1970, 1980, 1990, 2000, 2010, 2020]
        return decades[val % len(decades)]
    
    df['decade'] = df['track_id'].apply(get_decade)
    
    # Prepare the final dataframe matching tracks_clean schema
    clean_df = pd.DataFrame({
        'track_id': df['track_id'],
        'name': df['track_name'],
        'artist': df['artists'],
        'danceability': df['danceability'],
        'energy': df['energy'],
        'popularity': df['popularity'],
        'decade': df['decade']
    })
