import streamlit as st
import pandas as pd
import sys
import os
import pickle
import json
from pathlib import Path

# Setup paths so src can be imported
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.recommender import get_top_k_recommendations, get_user_history, explain_recommendation
from src.data_pipeline import load_movie_titles

# Page config
st.set_page_config(
    page_title="🎬 Netflix Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Netflix-inspired dark theme
st.markdown("""
<style>
    .stApp { background-color: #141414; color: #ffffff; }
    .metric-card {
        background: #1f1f1f;
        border-left: 5px solid #E50914;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 { color: #E50914 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B20710 !important;
    }
    .stDataFrame { background: #1f1f1f; }
</style>
""", unsafe_allow_html=True)

# Load data and model (cache for performance)
@st.cache_resource
def load_everything():
    df_path = PROJECT_ROOT / 'data' / 'processed' / 'ratings_sample.parquet'
    model_path = PROJECT_ROOT / 'models' / 'svd_model.pkl'
    comparison_path = PROJECT_ROOT / 'reports' / 'results' / 'model_comparison.json'
    insights_path = PROJECT_ROOT / 'reports' / 'results' / 'eda_insights.json'
    
    df = pd.read_parquet(str(df_path))
    movies = load_movie_titles(str(PROJECT_ROOT / 'data' / 'raw'))
    
    with open(str(model_path), 'rb') as f:
        model = pickle.load(f)
        
    with open(str(comparison_path)) as f:
        results = json.load(f)
        
    with open(str(insights_path)) as f:
        eda = json.load(f)
        
    return df, movies, model, results, eda

# Check files exist before loading
required_files = [
    PROJECT_ROOT / 'data' / 'processed' / 'ratings_sample.parquet',
    PROJECT_ROOT / 'models' / 'svd_model.pkl',
    PROJECT_ROOT / 'reports' / 'results' / 'model_comparison.json',
    PROJECT_ROOT / 'reports' / 'results' / 'eda_insights.json'
]

missing = [f for f in required_files if not f.exists()]
if missing:
    st.title("🎬 Netflix Recommendation Engine")
    st.error("Missing required data, model, or report files. Please run the notebook pipeline (`run_notebooks.py`) first to train models and generate results.")
    st.info("Files missing:\n" + "\n".join([f"- {f.name}" for f in missing]))
    st.stop()

df, movies, model, results, eda = load_everything()
all_movie_ids = df['movie_id'].unique().tolist()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Navigation")
tab = st.sidebar.radio("", ["🎯 Get Recommendations", "🎬 Movie Explorer", 
                              "📊 Model Performance", "🔍 EDA Insights"])

# ─────────────────────────────────────────────────────────
# TAB 1: GET RECOMMENDATIONS
# ─────────────────────────────────────────────────────────
if tab == "🎯 Get Recommendations":
    st.title("🎬 Personalized Movie Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Provide sample user ID from the dataset
        sample_users = [1488844, 822109, 885013, 1677480]
        user_id = st.number_input("Enter User ID", min_value=1, value=sample_users[0])
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)
        show_explanation = st.checkbox("Show Explanations", value=True)
        get_recs_btn = st.button("🎯 Get My Recommendations", type="primary")
    
    if get_recs_btn:
        seen_movies = set(df[df['user_id'] == user_id]['movie_id'])
        
        if len(seen_movies) == 0:
            st.warning("User not found in dataset sample. Showing popular movies instead.")
            # Top rated popular movies
            popular = df.groupby('movie_id').size().reset_index(name='count')
            popular_movies = popular.nlargest(n_recs, 'count').merge(movies, on='movie_id')
            st.subheader(f"🍿 Top {n_recs} Most Popular Movies")
            for idx, row in popular_movies.iterrows():
                with st.container():
                    st.markdown(f"**#{idx+1} {row['title']}** ({int(row['year']) if pd.notna(row['year']) else 'N/A'}) — {row['count']:,} ratings")
        else:
            with col2:
                st.subheader(f"Your Watch History (Top Rated)")
                history = get_user_history(user_id, df, movies, n=8)
                st.dataframe(history[['title', 'year', 'rating']], use_container_width=True)
            
            st.subheader(f"🍿 Top {n_recs} Recommendations for You")
            with st.spinner("Generating personalized recommendations..."):
                recs = get_top_k_recommendations(model, user_id, all_movie_ids, seen_movies, movies, k=n_recs)
            
            for _, row in recs.iterrows():
                with st.container():
                    cols = st.columns([1, 6, 2])
                    cols[0].metric("Rank", f"#{int(row['rank'])}")
                    cols[1].markdown(f"### {row['title']} ({int(row['year']) if pd.notna(row['year']) else 'N/A'})")
                    cols[2].metric("Predicted Rating", f"⭐ {row['predicted_rating']:.2f}")
                    
                    if show_explanation:
                        explanation = explain_recommendation(user_id, row['movie_id'], model, df, movies)
                        cols[1].markdown(f"*💡 {explanation}*")
                    st.markdown("---")

# ─────────────────────────────────────────────────────────
# TAB 2: MOVIE EXPLORER
# ─────────────────────────────────────────────────────────
elif tab == "🎬 Movie Explorer":
    st.title("🎬 Movie Explorer")
    
    search = st.text_input("Search for a movie by title")
    if search:
        found = movies[movies['title'].str.contains(search, case=False, na=False)]
        st.dataframe(found.head(20), use_container_width=True)
    
    st.subheader("Top 20 Most Rated Movies")
    top = df.groupby('movie_id').agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index().nlargest(20, 'rating_count').merge(movies, on='movie_id')
    st.dataframe(top[['title', 'year', 'rating_count', 'avg_rating']].round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
elif tab == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    
    results_df = pd.DataFrame(results).T
    
    col1, col2, col3 = st.columns(3)
    best_model = results_df['RMSE'].idxmin()
    col1.metric("Best Model (RMSE)", best_model, f"{results_df.loc[best_model, 'RMSE']:.4f}")
    col2.metric("Best RMSE", f"{results_df['RMSE'].min():.4f}")
    col3.metric("Best MAP@10", f"{results_df['MAP@10'].max():.4f}")
    
    st.subheader("Full Comparison Table")
    st.dataframe(results_df.round(4), use_container_width=True)
    
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=results_df.index, y=results_df['RMSE'], marker_color='#E50914'),
        go.Bar(name='MAP@10', x=results_df.index, y=results_df['MAP@10'], marker_color='#B20710')
    ])
    fig.update_layout(
        title='Model Comparison: RMSE vs MAP@10',
        barmode='group',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 4: EDA INSIGHTS
# ─────────────────────────────────────────────────────────
elif tab == "🔍 EDA Insights":
    st.title("🔍 Dataset Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings (Sample)", f"{eda['total_ratings_sampled']:,}")
    col2.metric("Unique Users", f"{eda['unique_users']:,}")
    col3.metric("Unique Movies", f"{eda['unique_movies']:,}")
    col4.metric("Sparsity", f"{eda['sparsity_pct']:.2f}%")
    
    st.subheader("Rating Distributions")
    if os.path.exists('../reports/figures/rating_distribution.png'):
        st.image('../reports/figures/rating_distribution.png', use_container_width=True)
    else:
        st.image(str(PROJECT_ROOT / 'reports' / 'figures' / 'rating_distribution.png'), use_container_width=True)
        
    st.subheader("User Activity Cumulative Contribution")
    if os.path.exists('../reports/figures/user_activity.png'):
        st.image('../reports/figures/user_activity.png', use_container_width=True)
    else:
        st.image(str(PROJECT_ROOT / 'reports' / 'figures' / 'user_activity.png'), use_container_width=True)

    st.subheader("Top Movies")
    if os.path.exists('../reports/figures/top_movies.png'):
        st.image('../reports/figures/top_movies.png', use_container_width=True)
    else:
        st.image(str(PROJECT_ROOT / 'reports' / 'figures' / 'top_movies.png'), use_container_width=True)
        
    st.subheader("Temporal Rating Trends")
    if os.path.exists('../reports/figures/temporal_analysis.png'):
        st.image('../reports/figures/temporal_analysis.png', use_container_width=True)
    else:
        st.image(str(PROJECT_ROOT / 'reports' / 'figures' / 'temporal_analysis.png'), use_container_width=True)
