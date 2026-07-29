import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- Custom CSS for a premium look ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_connection():
    """Initialize and cache the database connection engine."""
    return create_engine(DB_URL)


engine = init_connection()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("### 🚀 ML Platform")
    st.markdown("---")
    selection = st.radio(
        "Navigate",
        ["Home", "Spotify Analytics", "Airflow Status",
            "Energy Forecasting", "Fraud Detection"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Built with Airflow · FastAPI · Streamlit")

# --- Color Palette ---
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6584',
    'success': '#00C9A7',
    'warning': '#FFB800',
    'danger': '#FF4757',
    'info': '#54A0FF',
    'dark_bg': '#0E1117',
    'card_bg': '#1a1a2e',
    'text': '#E2E8F0',
    'grid': '#2D3748',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=COLORS['text'], family='Inter, sans-serif'),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
    yaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
)

# =====================================================
# HOME PAGE
# =====================================================
if selection == "Home":
    st.title("🚀 Multi-Domain Data & ML Platform")
    st.markdown(
        "Welcome to the unified dashboard for **Spotify Analytics**, "
        "**Energy Forecasting**, and **Fraud Detection**.")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.success("✅ Successfully connected to the PostgreSQL database!")
    except Exception as e:
        st.error(f"❌ Failed to connect to database: {e}")

    st.info("👈 Use the sidebar to navigate between domains.")

# =====================================================
# SPOTIFY ANALYTICS
# =====================================================
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
                "No data found in spotify.tracks_clean. "
                "Please run the Spotify Airflow DAG first.")
    except Exception as e:
        st.error(f"Error loading Spotify data: {e}")

# =====================================================
# AIRFLOW STATUS
# =====================================================
elif selection == "Airflow Status":
    st.title("⚙️ Airflow DAG Run History")
    st.markdown(
        "Monitoring automated data pipeline executions "
        "directly from the database.")

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

            status_counts = df['state'].value_counts()
            st.subheader("Run Status Distribution")
            st.bar_chart(status_counts)
        else:
            st.info("No DAG runs found yet.")
    except Exception as e:
        st.error(
            f"Error loading Airflow data (this is expected if Airflow "
            f"hasn't fully initialized the DB yet): {e}")

# =====================================================
# ENERGY FORECASTING
# =====================================================
elif selection == "Energy Forecasting":
    st.title("⚡ Energy Forecasting")
    st.markdown(
        "Prophet v2.0 energy consumption forecasts with "
        "US holiday effects and anomaly detection.")

    try:
        query = """
            SELECT forecast_timestamp, predicted_consumption, 
                   lower_band, upper_band, anomaly_flag 
            FROM energy.forecasts 
            ORDER BY forecast_timestamp 
            LIMIT 500
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            df = df.sort_values(by='forecast_timestamp')

            # --- KPI Row ---
            col1, col2, col3, col4 = st.columns(4)
            anomaly_count = df['anomaly_flag'].sum()
            avg_consumption = df['predicted_consumption'].mean()
            peak = df['predicted_consumption'].max()
            band_width = (df['upper_band'] - df['lower_band']).mean()

            with col1:
                st.metric("Forecast Points", f"{len(df):,}")
            with col2:
                st.metric("Avg Consumption", f"{avg_consumption:,.0f} MW")
            with col3:
                st.metric("Peak Demand", f"{peak:,.0f} MW")
            with col4:
                st.metric("Anomalies Detected", int(anomaly_count))

            st.markdown("---")

            # --- Forecast Chart with Confidence Bands ---
            st.subheader("📈 Consumption Forecast with Confidence Bands")

            fig = go.Figure()

            # Confidence band (shaded area)
            fig.add_trace(go.Scatter(
                x=df['forecast_timestamp'], y=df['upper_band'],
                mode='lines', line=dict(width=0),
                showlegend=False, name='Upper Band'
            ))
            fig.add_trace(go.Scatter(
                x=df['forecast_timestamp'], y=df['lower_band'],
                mode='lines', line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(108,99,255,0.15)',
                name='Confidence Band'
            ))

            # Predicted line
            fig.add_trace(go.Scatter(
                x=df['forecast_timestamp'],
                y=df['predicted_consumption'],
                mode='lines',
                line=dict(color=COLORS['primary'], width=2),
                name='Predicted'
            ))

            # Anomaly markers
            anomalies = df[df['anomaly_flag'] == True]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies['forecast_timestamp'],
                    y=anomalies['predicted_consumption'],
                    mode='markers',
                    marker=dict(color=COLORS['danger'], size=8,
                                symbol='x', line=dict(width=1)),
                    name='Anomaly'
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=450,
                legend=dict(orientation='h', y=-0.15),
                xaxis_title='Timestamp',
                yaxis_title='Consumption (MW)',
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Anomaly Table ---
            if not anomalies.empty:
                st.subheader(
                    f"🔴 Anomalous Hours ({len(anomalies)} detected)")
                st.dataframe(
                    anomalies[['forecast_timestamp',
                               'predicted_consumption',
                               'lower_band', 'upper_band']],
                    use_container_width=True)

        else:
            st.info(
                "No forecasts found in the database. "
                "Run the Energy Airflow DAG first.")
    except Exception as e:
        st.error(f"Error loading Energy data: {e}")

# =====================================================
# FRAUD DETECTION — FULLY REDESIGNED
# =====================================================
elif selection == "Fraud Detection":
    st.title("💳 Fraud Detection Analytics")
    st.markdown(
        "Comprehensive analysis of the XGBoost fraud classification model "
        "with real-time API telemetry.")

    try:
        # --- Fetch Prediction Telemetry ---
        telemetry_query = """
            SELECT transaction_id, timestamp, predicted_probability, 
                   predicted_class, latency_ms
            FROM fraud.predictions_log
            ORDER BY timestamp DESC
            LIMIT 5000
        """
        df = pd.read_sql(telemetry_query, engine)

        # --- Fetch Raw Transaction Distribution ---
        distribution_query = """
            SELECT is_fraud, COUNT(*) as count
            FROM fraud.transactions
            GROUP BY is_fraud
        """
        dist_df = pd.read_sql(distribution_query, engine)

        # --- Fetch Amount Statistics ---
        amount_query = """
            SELECT amount, is_fraud
            FROM fraud.transactions
            ORDER BY RANDOM()
            LIMIT 10000
        """
        amount_df = pd.read_sql(amount_query, engine)

        # --- Fetch Feature Correlations (top features) ---
        feature_query = """
            SELECT amount, features[1] as V1, features[2] as V2,
                   features[3] as V3, features[4] as V4,
                   features[14] as V14, features[17] as V17,
                   is_fraud
            FROM fraud.transactions
            ORDER BY RANDOM()
            LIMIT 5000
        """
        feature_df = pd.read_sql(feature_query, engine)

        has_telemetry = not df.empty
        has_transactions = not dist_df.empty

        if not has_telemetry and not has_transactions:
            st.info(
                "No data available. Run the Fraud ETL pipeline "
                "and send POST requests to /predict/fraud first.")
        else:
            # =========================================
            # ROW 1: KPI METRICS
            # =========================================
            st.subheader("📊 Key Performance Indicators")

            if has_telemetry:
                fraud_count = int(df['predicted_class'].sum())
                legit_count = len(df) - fraud_count
                fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                avg_latency = df['latency_ms'].mean()
                avg_prob = df['predicted_probability'].mean()
                p95_latency = df['latency_ms'].quantile(0.95)

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Predictions", f"{len(df):,}")
                with col2:
                    st.metric("Flagged Fraud", f"{fraud_count:,}")
                with col3:
                    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                with col4:
                    st.metric("Avg Latency", f"{avg_latency:.1f} ms")
                with col5:
                    st.metric("P95 Latency", f"{p95_latency:.1f} ms")

            st.markdown("---")

            # =========================================
            # TABBED VISUALIZATION SECTIONS
            # =========================================
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Dataset Analysis",
                "📈 Probability Distribution",
                "⚡ API Telemetry",
                "🧬 Feature Analysis"
            ])

            # =========================================
            # TAB 1: DATASET ANALYSIS
            # =========================================
            with tab1:
                if has_transactions:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("#### Class Distribution")
                        dist_df['label'] = dist_df['is_fraud'].map(
                            {True: 'Fraud', False: 'Legitimate'})
                        fig_pie = px.pie(
                            dist_df, values='count', names='label',
                            color='label',
                            color_discrete_map={
                                'Legitimate': COLORS['primary'],
                                'Fraud': COLORS['danger']
                            },
                            hole=0.55
                        )
                        fig_pie.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            legend=dict(orientation='h', y=-0.1),
                            annotations=[dict(
                                text=(f"{dist_df[dist_df['is_fraud'] == True]"
                                      f"['count'].sum():,}<br>Fraud"),
                                x=0.5, y=0.5, font_size=16,
                                font_color=COLORS['danger'],
                                showarrow=False
                            )]
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_right:
                        st.markdown("#### Transaction Amount Distribution")
                        if not amount_df.empty:
                            amount_df['class'] = amount_df['is_fraud'].map(
                                {True: 'Fraud', False: 'Legitimate'})

                            fig_hist = px.histogram(
                                amount_df, x='amount', color='class',
                                nbins=80, barmode='overlay',
                                color_discrete_map={
                                    'Legitimate': COLORS['primary'],
                                    'Fraud': COLORS['danger']
                                },
                                opacity=0.7,
                                log_y=True
                            )
                            fig_hist.update_layout(
                                **PLOTLY_LAYOUT, height=380,
                                xaxis_title='Transaction Amount ($)',
                                yaxis_title='Count (log scale)',
                                legend=dict(orientation='h', y=-0.15),
                            )
                            st.plotly_chart(
                                fig_hist, use_container_width=True)

                    # Amount Box Plot
                    st.markdown("#### Amount Comparison: Fraud vs Legitimate")
                    if not amount_df.empty:
                        fig_box = px.box(
                            amount_df, x='class', y='amount',
                            color='class',
                            color_discrete_map={
                                'Legitimate': COLORS['primary'],
                                'Fraud': COLORS['danger']
                            },
                            points='outliers'
                        )
                        fig_box.update_layout(
                            **PLOTLY_LAYOUT, height=350,
                            xaxis_title='', yaxis_title='Amount ($)',
                            showlegend=False,
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Run the Fraud ETL pipeline to load "
                            "transaction data.")

            # =========================================
            # TAB 2: PROBABILITY DISTRIBUTION
            # =========================================
            with tab2:
                if has_telemetry:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("#### Fraud Probability Distribution")
                        fig_prob = px.histogram(
                            df, x='predicted_probability',
                            nbins=50,
                            color_discrete_sequence=[COLORS['primary']],
                            opacity=0.8
                        )
                        fig_prob.add_vline(
                            x=0.5, line_dash="dash",
                            line_color=COLORS['danger'],
                            annotation_text="Decision Threshold (0.5)",
                            annotation_position="top right"
                        )
                        fig_prob.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            xaxis_title='Predicted Fraud Probability',
                            yaxis_title='Count',
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

                    with col_right:
                        st.markdown("#### Prediction Outcome Breakdown")
                        outcome_df = pd.DataFrame({
                            'Outcome': ['Legitimate', 'Fraud'],
                            'Count': [legit_count, fraud_count]
                        })
                        fig_bar = px.bar(
                            outcome_df, x='Outcome', y='Count',
                            color='Outcome',
                            color_discrete_map={
                                'Legitimate': COLORS['success'],
                                'Fraud': COLORS['danger']
                            },
                            text='Count'
                        )
                        fig_bar.update_traces(textposition='outside')
                        fig_bar.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            showlegend=False,
                            xaxis_title='',
                            yaxis_title='Number of Predictions',
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Probability over time scatter
                    st.markdown(
                        "#### Fraud Probability Over Time")
                    df_sorted = df.sort_values('timestamp')
                    df_sorted['class'] = df_sorted[
                        'predicted_class'].map(
                        {True: 'Fraud', False: 'Legitimate'})

                    fig_scatter = px.scatter(
                        df_sorted, x='timestamp',
                        y='predicted_probability',
                        color='class',
                        color_discrete_map={
                            'Legitimate': COLORS['primary'],
                            'Fraud': COLORS['danger']
                        },
                        opacity=0.7,
                        size='predicted_probability',
                        size_max=12,
                    )
                    fig_scatter.add_hline(
                        y=0.5, line_dash="dash",
                        line_color=COLORS['warning'],
                        annotation_text="Threshold"
                    )
                    fig_scatter.update_layout(
                        **PLOTLY_LAYOUT, height=400,
                        xaxis_title='Timestamp',
                        yaxis_title='Fraud Probability',
                        legend=dict(orientation='h', y=-0.15),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Send POST requests to /predict/fraud "
                            "to generate probability data.")

            # =========================================
            # TAB 3: API TELEMETRY
            # =========================================
            with tab3:
                if has_telemetry:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("#### API Latency Over Time")
                        df_sorted = df.sort_values('timestamp')
                        fig_latency = go.Figure()

                        fig_latency.add_trace(go.Scatter(
                            x=df_sorted['timestamp'],
                            y=df_sorted['latency_ms'],
                            mode='lines+markers',
                            marker=dict(size=4, color=COLORS['info']),
                            line=dict(color=COLORS['info'], width=1.5),
                            name='Latency'
                        ))

                        # Add P95 line
                        fig_latency.add_hline(
                            y=p95_latency, line_dash="dot",
                            line_color=COLORS['warning'],
                            annotation_text=f"P95: {p95_latency:.1f}ms"
                        )

                        fig_latency.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            xaxis_title='Timestamp',
                            yaxis_title='Latency (ms)',
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig_latency, use_container_width=True)

                    with col_right:
                        st.markdown("#### Latency Distribution")
                        fig_lat_hist = px.histogram(
                            df, x='latency_ms', nbins=40,
                            color_discrete_sequence=[COLORS['info']],
                            opacity=0.8
                        )
                        fig_lat_hist.add_vline(
                            x=avg_latency, line_dash="dash",
                            line_color=COLORS['success'],
                            annotation_text=(
                                f"Mean: {avg_latency:.1f}ms"),
                        )
                        fig_lat_hist.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            xaxis_title='Latency (ms)',
                            yaxis_title='Count',
                        )
                        st.plotly_chart(
                            fig_lat_hist, use_container_width=True)

                    # Recent telemetry table
                    st.markdown("#### 📋 Recent API Telemetry Log")
                    display_df = df.head(25).copy()
                    display_df['predicted_class'] = display_df[
                        'predicted_class'].map(
                        {True: '🔴 Fraud', False: '🟢 Legit'})
                    display_df['predicted_probability'] = display_df[
                        'predicted_probability'].apply(
                        lambda x: f"{x:.4f}")
                    display_df['latency_ms'] = display_df[
                        'latency_ms'].apply(lambda x: f"{x:.2f}")

                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("Send POST requests to /predict/fraud "
                            "to generate telemetry data.")

            # =========================================
            # TAB 4: FEATURE ANALYSIS
            # =========================================
            with tab4:
                if not feature_df.empty:
                    st.markdown(
                        "#### PCA Feature Distributions: "
                        "Fraud vs Legitimate")
                    st.caption(
                        "These plots reveal how key PCA features "
                        "separate fraud from legitimate transactions. "
                        "Greater separation = stronger signal.")

                    feature_df['class'] = feature_df['is_fraud'].map(
                        {True: 'Fraud', False: 'Legitimate'})

                    # Create a 2x3 grid of violin plots for key features
                    features_to_plot = [
                        ('v1', 'V1 (Top Discriminator)'),
                        ('v2', 'V2'),
                        ('v3', 'V3'),
                        ('v4', 'V4'),
                        ('v14', 'V14 (Strong Signal)'),
                        ('v17', 'V17 (Strong Signal)'),
                    ]

                    col1, col2 = st.columns(2)
                    for idx, (feat, label) in enumerate(
                            features_to_plot):
                        target_col = col1 if idx % 2 == 0 else col2
                        with target_col:
                            fig_violin = px.violin(
                                feature_df, y=feat, x='class',
                                color='class', box=True,
                                color_discrete_map={
                                    'Legitimate': COLORS['primary'],
                                    'Fraud': COLORS['danger']
                                },
                            )
                            fig_violin.update_layout(
                                **PLOTLY_LAYOUT, height=300,
                                title=dict(text=label, font_size=14),
                                showlegend=False,
                                xaxis_title='',
                                yaxis_title=feat.upper(),
                            )
                            st.plotly_chart(
                                fig_violin,
                                use_container_width=True)

                    # Feature correlation heatmap
                    st.markdown("#### Feature Correlation Matrix")
                    corr_cols = ['amount', 'v1', 'v2', 'v3',
                                 'v4', 'v14', 'v17']
                    corr_matrix = feature_df[corr_cols].corr()

                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        zmin=-1, zmax=1
                    )
                    fig_heat.update_layout(
                        **PLOTLY_LAYOUT, height=450,
                        title=dict(text='PCA Feature Correlations',
                                   font_size=14),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Run the Fraud ETL pipeline to load "
                            "transaction data for feature analysis.")

    except Exception as e:
        st.error(f"Error loading Fraud data: {e}")
