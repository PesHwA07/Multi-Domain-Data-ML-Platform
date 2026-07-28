import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os

# --- Configuration ---
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")

st.set_page_config(
    page_title="Multi-Domain ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_connection():
    """Initialize and cache the database connection engine."""
    return create_engine(DB_URL)


engine = init_connection()

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Navigation")
    selection = st.radio(
        "Select Domain",
        ["Home", "Spotify Analytics", "Airflow Status",
            "Energy Forecasting", "Fraud Detection"]
    )

# --- Pages ---

if selection == "Home":
    st.title("🚀 Multi-Domain Data & ML Platform")
    st.markdown(
        "Welcome to the unified dashboard for **Spotify Analytics**, **Energy Forecasting**, and **Fraud Detection**.")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.success("✅ Successfully connected to the PostgreSQL database!")
    except Exception as e:
        st.error(f"❌ Failed to connect to database: {e}")

    st.info("👈 Use the sidebar to navigate between domains.")

elif selection == "Spotify Analytics":
    st.title("🎵 Spotify Analytics")
    st.markdown("Visualizing curated data from the batch ETL pipeline.")

    try:
        query = """
            SELECT artist, COUNT(*) as track_count, AVG(popularity) as avg_popularity, 
                   AVG(danceability) as avg_danceability
            FROM spotify.tracks_clean
            GROUP BY artist
            ORDER BY track_count DESC
            LIMIT 20
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            st.subheader("Top 20 Artists by Track Count")
            st.dataframe(df, use_container_width=True)

            st.subheader("Popularity vs Danceability")
            st.scatter_chart(df, x="avg_popularity", y="avg_danceability")
        else:
            st.warning(
                "No data found in spotify.tracks_clean. Please run the Spotify Airflow DAG first.")
    except Exception as e:
        st.error(f"Error loading Spotify data: {e}")

elif selection == "Airflow Status":
    st.title("⚙️ Airflow DAG Run History")
    st.markdown(
        "Monitoring automated data pipeline executions directly from the database.")

    try:
        query = """
            SELECT dag_id, execution_date, state, run_type
            FROM dag_run
            ORDER BY execution_date DESC
            LIMIT 50
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            st.subheader("Recent DAG Runs")
            st.dataframe(df, use_container_width=True)

            # Simple bar chart of statuses
            status_counts = df['state'].value_counts()
            st.subheader("Run Status Distribution")
            st.bar_chart(status_counts)
        else:
            st.info("No DAG runs found yet.")
    except Exception as e:
        st.error(
            f"Error loading Airflow data (this is expected if Airflow hasn't fully initialized the DB yet): {e}")

elif selection == "Energy Forecasting":
    st.title("⚡ Energy Forecasting")
    st.markdown(
        "Visualizing Prophet's energy consumption forecasts and anomaly bounds.")

    try:
        query = """
            SELECT forecast_timestamp, predicted_consumption, lower_band, upper_band, anomaly_flag 
            FROM energy.forecasts 
            ORDER BY forecast_timestamp DESC 
            LIMIT 168
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            # Sort chronologically for the chart
            df = df.sort_values(by='forecast_timestamp')

            st.subheader("1-Week Consumption Forecast")
            # Set index for streamlit line_chart
            chart_df = df.set_index("forecast_timestamp")[
                ["predicted_consumption", "lower_band", "upper_band"]]
            st.line_chart(chart_df)

            # Show Anomalies
            anomalies = df[df["anomaly_flag"] == True]
            st.subheader(f"Detected Anomalies: {len(anomalies)}")
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
        else:
            st.info(
                "No forecasts found in the database. Run the Energy Airflow DAG first.")
    except Exception as e:
        st.error(f"Error loading Energy data: {e}")

elif selection == "Fraud Detection":
    st.title("💳 Fraud Detection")
    st.markdown(
        "Live telemetry and metrics from the real-time Fraud Detection FastAPI endpoint.")

    try:
        query = """
            SELECT timestamp, predicted_probability, predicted_class, latency_ms
            FROM fraud.predictions_log
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total API Requests Logged", len(df))
            with col2:
                fraud_count = df['predicted_class'].sum()
                st.metric("Flagged Fraud Transactions", int(fraud_count))
            with col3:
                avg_latency = df['latency_ms'].mean()
                st.metric("Average API Latency", f"{avg_latency:.2f} ms")

            st.subheader("Recent API Latency (ms)")
            chart_data = df.sort_values(by="timestamp").set_index(
                "timestamp")[["latency_ms"]]
            st.line_chart(chart_data)

            st.subheader("Recent API Telemetry")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info(
                "No predictions logged yet. Send a POST request to FastAPI /predict/fraud first.")
    except Exception as e:
        st.error(f"Error loading Fraud telemetry data: {e}")
