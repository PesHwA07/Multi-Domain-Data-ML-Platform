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
        "A unified infrastructure powering **3 domains**, "
        "**5 microservices**, and **400K+ records** with "
        "sub-100ms inference latency.")

    st.markdown("---")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Fetch live counts from all 3 domains
        spotify_count = 0
        energy_count = 0
        fraud_count = 0
        prediction_count = 0
        try:
            with engine.connect() as conn:
                r = conn.execute(text(
                    "SELECT COUNT(*) FROM spotify.tracks_clean"))
                spotify_count = r.scalar() or 0
                r = conn.execute(text(
                    "SELECT COUNT(*) FROM energy.hourly_readings"))
                energy_count = r.scalar() or 0
                r = conn.execute(text(
                    "SELECT COUNT(*) FROM fraud.transactions"))
                fraud_count = r.scalar() or 0
                r = conn.execute(text(
                    "SELECT COUNT(*) FROM fraud.predictions_log"))
                prediction_count = r.scalar() or 0
        except Exception:
            pass

        total_records = spotify_count + energy_count + fraud_count

        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Spotify Tracks", f"{spotify_count:,}")
        with col3:
            st.metric("Energy Readings", f"{energy_count:,}")
        with col4:
            st.metric("Fraud Transactions", f"{fraud_count:,}")
        with col5:
            st.metric("API Predictions", f"{prediction_count:,}")

        st.markdown("---")

        # Domain cards
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("### 🎵 Spotify Analytics")
            st.markdown(
                "Batch ETL pipeline processing artist metadata, "
                "audio features, and popularity scores.")
            fig = go.Figure(go.Indicator(
                mode="number", value=spotify_count,
                title=dict(text="Tracks Loaded",
                           font=dict(size=14)),
                number=dict(font=dict(size=36,
                                      color=COLORS['primary'])),
            ))
            fig.update_layout(
                height=150,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("### ⚡ Energy Forecasting")
            st.markdown(
                "Prophet v2.0 time-series forecasting with "
                "US holidays and anomaly detection.")
            fig = go.Figure(go.Indicator(
                mode="number", value=energy_count,
                title=dict(text="Hourly Readings",
                           font=dict(size=14)),
                number=dict(font=dict(size=36,
                                      color=COLORS['success'])),
            ))
            fig.update_layout(
                height=150,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_c:
            st.markdown("### 💳 Fraud Detection")
            st.markdown(
                "XGBoost real-time classifier with GridSearchCV "
                "and sub-100ms FastAPI serving.")
            fig = go.Figure(go.Indicator(
                mode="number", value=fraud_count,
                title=dict(text="Transactions Analyzed",
                           font=dict(size=14)),
                number=dict(font=dict(size=36,
                                      color=COLORS['danger'])),
            ))
            fig.update_layout(
                height=150,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.success("✅ All systems connected and operational.")

    except Exception as e:
        st.error(f"❌ Failed to connect to database: {e}")

# =====================================================
# SPOTIFY ANALYTICS — FULLY REDESIGNED
# =====================================================
elif selection == "Spotify Analytics":
    st.title("🎵 Spotify Analytics")
    st.markdown(
        "Deep-dive into audio features, artist popularity, "
        "and decade trends from the batch ETL pipeline.")

    try:
        # Fetch comprehensive data
        overview_query = """
            SELECT COUNT(*) as total_tracks,
                   COUNT(DISTINCT artist) as unique_artists,
                   AVG(popularity) as avg_popularity,
                   AVG(danceability) as avg_danceability,
                   AVG(energy) as avg_energy
            FROM spotify.tracks_clean
        """
        overview = pd.read_sql(overview_query, engine)

        top_artists_query = """
            SELECT artist, COUNT(*) as track_count,
                   AVG(popularity) as avg_popularity,
                   AVG(danceability) as avg_danceability,
                   AVG(energy) as avg_energy
            FROM spotify.tracks_clean
            GROUP BY artist
            ORDER BY track_count DESC
            LIMIT 15
        """
        top_artists = pd.read_sql(top_artists_query, engine)

        decade_query = """
            SELECT decade, COUNT(*) as track_count,
                   AVG(popularity) as avg_popularity,
                   AVG(danceability) as avg_danceability,
                   AVG(energy) as avg_energy
            FROM spotify.tracks_clean
            WHERE decade IS NOT NULL
            GROUP BY decade
            ORDER BY decade
        """
        decade_df = pd.read_sql(decade_query, engine)

        scatter_query = """
            SELECT name, artist, popularity, danceability, energy
            FROM spotify.tracks_clean
            ORDER BY popularity DESC
            LIMIT 500
        """
        scatter_df = pd.read_sql(scatter_query, engine)

        all_tracks_query = """
            SELECT danceability, energy, popularity
            FROM spotify.tracks_clean
            ORDER BY RANDOM()
            LIMIT 5000
        """
        all_tracks = pd.read_sql(all_tracks_query, engine)

        if overview.iloc[0]['total_tracks'] == 0:
            st.warning("No data found. Run the Spotify DAG first.")
        else:
            # --- KPI Row ---
            row = overview.iloc[0]
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Tracks",
                          f"{int(row['total_tracks']):,}")
            with col2:
                st.metric("Unique Artists",
                          f"{int(row['unique_artists']):,}")
            with col3:
                st.metric("Avg Popularity",
                          f"{row['avg_popularity']:.1f}")
            with col4:
                st.metric("Avg Danceability",
                          f"{row['avg_danceability']:.2f}")
            with col5:
                st.metric("Avg Energy",
                          f"{row['avg_energy']:.2f}")

            st.markdown("---")

            # --- Tabbed Views ---
            tab1, tab2, tab3 = st.tabs([
                "🏆 Artist Insights",
                "📅 Decade Trends",
                "🔬 Audio Feature Analysis"
            ])

            # ============================
            # TAB 1: ARTIST INSIGHTS
            # ============================
            with tab1:
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### Top 15 Artists by Track Count")
                    fig_bar = px.bar(
                        top_artists.sort_values('track_count'),
                        x='track_count', y='artist',
                        orientation='h',
                        color='avg_popularity',
                        color_continuous_scale='Viridis',
                        labels={'track_count': 'Tracks',
                                'avg_popularity': 'Avg Pop.'},
                    )
                    fig_bar.update_layout(
                        **PLOTLY_LAYOUT, height=500,
                        yaxis=dict(gridcolor=COLORS['grid']),
                        coloraxis_colorbar=dict(title="Pop."),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_right:
                    st.markdown("#### Artist Audio DNA Radar")
                    # Radar chart for top 5 artists
                    top5 = top_artists.head(5)
                    fig_radar = go.Figure()
                    categories = ['Popularity', 'Danceability',
                                  'Energy']

                    radar_colors = [
                        COLORS['primary'], COLORS['secondary'],
                        COLORS['success'], COLORS['warning'],
                        COLORS['info']
                    ]

                    for i, (_, art) in enumerate(top5.iterrows()):
                        vals = [
                            art['avg_popularity'] / 100,
                            art['avg_danceability'],
                            art['avg_energy']
                        ]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=categories + [categories[0]],
                            fill='toself',
                            name=art['artist'][:20],
                            line=dict(color=radar_colors[
                                i % len(radar_colors)]),
                            opacity=0.6
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(
                                visible=True, range=[0, 1],
                                gridcolor=COLORS['grid']),
                            angularaxis=dict(
                                gridcolor=COLORS['grid']),
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS['text']),
                        height=500,
                        legend=dict(orientation='h', y=-0.15),
                        margin=dict(l=60, r=60, t=40, b=40),
                    )
                    st.plotly_chart(
                        fig_radar, use_container_width=True)

                # Popularity vs Track Count bubble
                st.markdown(
                    "#### Artist Popularity vs Catalog Size")
                fig_bubble = px.scatter(
                    top_artists, x='track_count',
                    y='avg_popularity',
                    size='avg_energy',
                    color='avg_danceability',
                    hover_name='artist',
                    color_continuous_scale='Plasma',
                    size_max=40,
                    labels={
                        'track_count': 'Total Tracks',
                        'avg_popularity': 'Avg Popularity',
                        'avg_energy': 'Avg Energy',
                        'avg_danceability': 'Danceability'},
                )
                fig_bubble.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                )
                st.plotly_chart(fig_bubble,
                                use_container_width=True)

            # ============================
            # TAB 2: DECADE TRENDS
            # ============================
            with tab2:
                if not decade_df.empty:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown(
                            "#### Tracks Produced Per Decade")
                        decade_df['decade_label'] = (
                            decade_df['decade'].astype(str) + 's')
                        fig_decade_bar = px.bar(
                            decade_df, x='decade_label',
                            y='track_count',
                            color='avg_popularity',
                            color_continuous_scale='Turbo',
                            labels={
                                'decade_label': 'Decade',
                                'track_count': 'Tracks',
                                'avg_popularity': 'Pop.'},
                            text='track_count',
                        )
                        fig_decade_bar.update_traces(
                            textposition='outside')
                        fig_decade_bar.update_layout(
                            **PLOTLY_LAYOUT, height=400,
                        )
                        st.plotly_chart(
                            fig_decade_bar,
                            use_container_width=True)

                    with col_right:
                        st.markdown(
                            "#### Audio Feature Evolution "
                            "Across Decades")
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(
                            x=decade_df['decade'],
                            y=decade_df['avg_danceability'],
                            mode='lines+markers',
                            name='Danceability',
                            line=dict(
                                color=COLORS['primary'],
                                width=3),
                            marker=dict(size=8),
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=decade_df['decade'],
                            y=decade_df['avg_energy'],
                            mode='lines+markers',
                            name='Energy',
                            line=dict(
                                color=COLORS['secondary'],
                                width=3),
                            marker=dict(size=8),
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=decade_df['decade'],
                            y=decade_df['avg_popularity'] / 100,
                            mode='lines+markers',
                            name='Popularity (scaled)',
                            line=dict(
                                color=COLORS['success'],
                                width=3),
                            marker=dict(size=8),
                        ))
                        fig_line.update_layout(
                            **PLOTLY_LAYOUT, height=400,
                            xaxis_title='Decade',
                            yaxis_title='Score (0-1)',
                            legend=dict(
                                orientation='h', y=-0.15),
                        )
                        st.plotly_chart(
                            fig_line, use_container_width=True)

                    # Decade heatmap
                    st.markdown(
                        "#### Decade × Feature Heatmap")
                    heat_data = decade_df[[
                        'decade_label', 'avg_popularity',
                        'avg_danceability', 'avg_energy'
                    ]].set_index('decade_label')
                    heat_data.columns = [
                        'Popularity', 'Danceability', 'Energy']
                    # Normalize
                    heat_norm = heat_data.copy()
                    for col in heat_norm.columns:
                        cmin = heat_norm[col].min()
                        cmax = heat_norm[col].max()
                        denom = cmax - cmin
                        if denom > 0:
                            heat_norm[col] = (
                                (heat_norm[col] - cmin) / denom)
                        else:
                            heat_norm[col] = 0

                    fig_heat = px.imshow(
                        heat_norm.T,
                        text_auto='.2f',
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                    )
                    fig_heat.update_layout(
                        **PLOTLY_LAYOUT, height=300,
                        xaxis_title='Decade',
                        yaxis_title='Feature',
                    )
                    st.plotly_chart(
                        fig_heat, use_container_width=True)
                else:
                    st.info("No decade data available.")

            # ============================
            # TAB 3: AUDIO FEATURE ANALYSIS
            # ============================
            with tab3:
                if not all_tracks.empty:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown(
                            "#### Danceability vs Energy")
                        fig_scatter = px.scatter(
                            all_tracks,
                            x='danceability', y='energy',
                            color='popularity',
                            color_continuous_scale='Inferno',
                            opacity=0.5,
                            labels={
                                'danceability': 'Danceability',
                                'energy': 'Energy',
                                'popularity': 'Popularity'},
                        )
                        fig_scatter.update_traces(
                            marker=dict(size=4))
                        fig_scatter.update_layout(
                            **PLOTLY_LAYOUT, height=450,
                        )
                        st.plotly_chart(
                            fig_scatter,
                            use_container_width=True)

                    with col_right:
                        st.markdown(
                            "#### Feature Distributions")
                        fig_dist = make_subplots(
                            rows=3, cols=1, shared_xaxes=False,
                            subplot_titles=[
                                'Danceability', 'Energy',
                                'Popularity'],
                            vertical_spacing=0.12,
                        )
                        fig_dist.add_trace(go.Histogram(
                            x=all_tracks['danceability'],
                            nbinsx=40,
                            marker_color=COLORS['primary'],
                            opacity=0.8, name='Dance',
                        ), row=1, col=1)
                        fig_dist.add_trace(go.Histogram(
                            x=all_tracks['energy'],
                            nbinsx=40,
                            marker_color=COLORS['secondary'],
                            opacity=0.8, name='Energy',
                        ), row=2, col=1)
                        fig_dist.add_trace(go.Histogram(
                            x=all_tracks['popularity'],
                            nbinsx=40,
                            marker_color=COLORS['success'],
                            opacity=0.8, name='Pop',
                        ), row=3, col=1)
                        fig_dist.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=COLORS['text']),
                            height=450, showlegend=False,
                            margin=dict(
                                l=40, r=40, t=40, b=20),
                        )
                        for i in range(1, 4):
                            fig_dist.update_xaxes(
                                gridcolor=COLORS['grid'],
                                row=i, col=1)
                            fig_dist.update_yaxes(
                                gridcolor=COLORS['grid'],
                                row=i, col=1)
                        st.plotly_chart(
                            fig_dist,
                            use_container_width=True)

                    # Correlation matrix
                    st.markdown(
                        "#### Audio Feature Correlations")
                    corr = all_tracks[[
                        'danceability', 'energy',
                        'popularity']].corr()
                    fig_corr = px.imshow(
                        corr, text_auto='.3f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect='auto',
                    )
                    fig_corr.update_layout(
                        **PLOTLY_LAYOUT, height=350,
                    )
                    st.plotly_chart(
                        fig_corr, use_container_width=True)
                else:
                    st.info("No track data available.")

    except Exception as e:
        st.error(f"Error loading Spotify data: {e}")

# =====================================================
# AIRFLOW STATUS — FULLY REDESIGNED
# =====================================================
elif selection == "Airflow Status":
    st.title("⚙️ Airflow Pipeline Operations")
    st.markdown(
        "Real-time monitoring of automated DAG executions "
        "and pipeline health metrics.")

    try:
        query = """
            SELECT dag_id, execution_date, state, run_type,
                   start_date, end_date
            FROM dag_run
            ORDER BY execution_date DESC
            LIMIT 100
        """
        df = pd.read_sql(query, engine)

        if not df.empty:
            # --- KPI Row ---
            total_runs = len(df)
            unique_dags = df['dag_id'].nunique()
            success_count = len(df[df['state'] == 'success'])
            failed_count = len(df[df['state'] == 'failed'])
            success_rate = (
                success_count / total_runs * 100
                if total_runs > 0 else 0)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Runs", f"{total_runs}")
            with col2:
                st.metric("Unique DAGs", f"{unique_dags}")
            with col3:
                st.metric("Successful", f"{success_count}",
                          delta=f"{success_rate:.0f}%")
            with col4:
                st.metric("Failed", f"{failed_count}",
                          delta=f"-{failed_count}" if
                          failed_count > 0 else "0",
                          delta_color="inverse")
            with col5:
                running_count = len(
                    df[df['state'] == 'running'])
                st.metric("Running Now", f"{running_count}")

            st.markdown("---")

            tab1, tab2, tab3 = st.tabs([
                "📊 Overview",
                "📋 Run History",
                "🔍 Per-DAG Analysis"
            ])

            # ============================
            # TAB 1: OVERVIEW
            # ============================
            with tab1:
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### Pipeline Success Rate")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=success_rate,
                        number=dict(suffix="%"),
                        delta=dict(
                            reference=95,
                            increasing=dict(
                                color=COLORS['success']),
                            decreasing=dict(
                                color=COLORS['danger'])),
                        gauge=dict(
                            axis=dict(range=[0, 100]),
                            bar=dict(
                                color=COLORS['success']),
                            bgcolor=COLORS['grid'],
                            borderwidth=0,
                            steps=[
                                dict(range=[0, 50],
                                     color='#2D1B2E'),
                                dict(range=[50, 80],
                                     color='#1B2D2E'),
                                dict(range=[80, 100],
                                     color='#1B2E1B')],
                            threshold=dict(
                                line=dict(
                                    color=COLORS['warning'],
                                    width=3),
                                thickness=0.8,
                                value=95)),
                        title=dict(
                            text="Target: 95%",
                            font=dict(size=14)),
                    ))
                    fig_gauge.update_layout(
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS['text']),
                        margin=dict(l=30, r=30, t=60, b=20))
                    st.plotly_chart(
                        fig_gauge,
                        use_container_width=True)

                with col_right:
                    st.markdown(
                        "#### Run Status Distribution")
                    state_df = df['state'].value_counts(
                        ).reset_index()
                    state_df.columns = ['state', 'count']

                    state_colors = {
                        'success': COLORS['success'],
                        'failed': COLORS['danger'],
                        'running': COLORS['info'],
                        'queued': COLORS['warning'],
                    }
                    state_df['color'] = state_df['state'].map(
                        lambda x: state_colors.get(
                            x, COLORS['primary']))

                    fig_state = px.pie(
                        state_df, values='count',
                        names='state',
                        color='state',
                        color_discrete_map=state_colors,
                        hole=0.5,
                    )
                    fig_state.update_layout(
                        **PLOTLY_LAYOUT, height=350,
                        legend=dict(
                            orientation='h', y=-0.1),
                    )
                    st.plotly_chart(
                        fig_state,
                        use_container_width=True)

                # Timeline
                st.markdown("#### Execution Timeline")
                df_sorted = df.sort_values('execution_date')

                # Color map by state
                fig_timeline = px.scatter(
                    df_sorted, x='execution_date',
                    y='dag_id', color='state',
                    color_discrete_map=state_colors,
                    symbol='run_type',
                    hover_data=['state', 'run_type'],
                    size_max=12,
                )
                fig_timeline.update_traces(
                    marker=dict(size=10))
                fig_timeline.update_layout(
                    **PLOTLY_LAYOUT, height=350,
                    xaxis_title='Execution Date',
                    yaxis_title='',
                    legend=dict(orientation='h', y=-0.15),
                )
                st.plotly_chart(
                    fig_timeline,
                    use_container_width=True)

            # ============================
            # TAB 2: RUN HISTORY
            # ============================
            with tab2:
                st.markdown("#### 📋 Recent DAG Runs")

                display_df = df.copy()
                state_emoji = {
                    'success': '✅ Success',
                    'failed': '❌ Failed',
                    'running': '🔄 Running',
                    'queued': '⏳ Queued',
                }
                display_df['status'] = display_df[
                    'state'].map(
                    lambda x: state_emoji.get(x, x))

                show_cols = [
                    'dag_id', 'status',
                    'execution_date', 'run_type']
                available = [
                    c for c in show_cols
                    if c in display_df.columns]

                st.dataframe(
                    display_df[available].head(50),
                    use_container_width=True, height=500)

            # ============================
            # TAB 3: PER-DAG ANALYSIS
            # ============================
            with tab3:
                st.markdown("#### Per-DAG Performance")

                dag_stats = df.groupby('dag_id').agg(
                    total_runs=('state', 'count'),
                    successes=('state',
                               lambda x: (x == 'success').sum()),
                    failures=('state',
                              lambda x: (x == 'failed').sum()),
                ).reset_index()
                dag_stats['success_rate'] = (
                    dag_stats['successes'] /
                    dag_stats['total_runs'] * 100
                ).round(1)

                fig_dag_bar = px.bar(
                    dag_stats, x='dag_id',
                    y=['successes', 'failures'],
                    barmode='stack',
                    color_discrete_map={
                        'successes': COLORS['success'],
                        'failures': COLORS['danger']},
                    labels={'value': 'Runs',
                            'variable': 'Status'},
                    text_auto=True,
                )
                fig_dag_bar.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                    xaxis_title='DAG',
                    yaxis_title='Number of Runs',
                    legend=dict(orientation='h', y=-0.15),
                )
                st.plotly_chart(
                    fig_dag_bar,
                    use_container_width=True)

                # Success rate table
                st.markdown("#### Success Rate by DAG")
                st.dataframe(
                    dag_stats[['dag_id', 'total_runs',
                               'successes', 'failures',
                               'success_rate']],
                    use_container_width=True)

        else:
            st.info("No DAG runs found yet.")
    except Exception as e:
        st.error(
            f"Error loading Airflow data (expected if Airflow "
            f"hasn't fully initialized): {e}")

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
                        fraud_total = int(
                            dist_df[dist_df['is_fraud'] == True][
                                'count'].sum())
                        fig_pie.update_layout(
                            **PLOTLY_LAYOUT, height=380,
                            legend=dict(orientation='h', y=-0.1),
                            annotations=[dict(
                                text=f"{fraud_total:,}<br>Fraud",
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
