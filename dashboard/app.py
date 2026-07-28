import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")

st.set_page_config(
    page_title="Multi-Domain ML Platform", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main UI ---
st.title("🚀 Multi-Domain Data & ML Platform")
st.markdown("Welcome to the unified dashboard for **Spotify Analytics**, **Energy Forecasting**, and **Fraud Detection**.")

@st.cache_resource
def init_connection():
    """Initialize and cache the database connection engine."""
    return create_engine(DB_URL)

try:
    engine = init_connection()
    # Simple query to verify connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("✅ Successfully connected to the PostgreSQL database!")
except Exception as e:
    st.error(f"❌ Failed to connect to database: {e}")

st.info("👈 Use the sidebar to navigate between domains (Features coming in Days 23-24).")

# Basic Sidebar outline for future days
with st.sidebar:
    st.header("Navigation")
    st.radio(
        "Select Domain",
        ["Home", "Spotify Analytics", "Energy Forecasting", "Fraud Detection"]
    )
