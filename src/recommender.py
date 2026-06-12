import pandas as pd
import numpy as np

def get_top_k_recommendations(model, user_id, all_movie_ids, seen_movie_ids, movies_df, k=10):
    """
    Generate Top-K movie recommendations for a user.
    
    Args:
        model: trained model (Surprise model or custom model with predict/predict_ncf_batch)
        user_id: target user ID
        all_movie_ids: list of all valid movie IDs
        seen_movie_ids: set of movies user has already rated
        movies_df: DataFrame with movie_id, title, year
        k: number of recommendations
    
    Returns:
        DataFrame with columns: [rank, movie_id, title, year, predicted_rating]
    """
    unseen = [m for m in all_movie_ids if m not in seen_movie_ids]
    
    # Check if NCF model
    is_ncf = hasattr(model, 'predict_ncf_batch')
    
    if is_ncf:
        preds = model.predict_ncf_batch(user_id, unseen)
    else:
        preds = [(m, model.predict(user_id, m).est) for m in unseen]
        
    top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
    
    recs_df = pd.DataFrame(top_k, columns=['movie_id', 'predicted_rating'])
    recs_df = recs_df.merge(movies_df[['movie_id', 'title', 'year']], on='movie_id', how='left')
    recs_df['rank'] = range(1, len(recs_df) + 1)
    
    return recs_df[['rank', 'movie_id', 'title', 'year', 'predicted_rating']]


def get_user_history(user_id, df, movies_df, n=10):
    """Get a user's top-rated movies for display."""
    user_ratings = df[df['user_id'] == user_id].merge(movies_df, on='movie_id')
    return user_ratings.nlargest(n, 'rating')[['movie_id', 'title', 'year', 'rating']].reset_index(drop=True)


def explain_recommendation(user_id, rec_movie_id, model, train_df, movies_df, n_explain=3):
    """
    Generate a human-readable explanation for a recommendation.
    
    Logic: Find top-rated movies by this user in train_df. 
    Select the top movies and present them as similar interests that led to the recommendation.
    """
    # Get user's top rated movies in train_df
    user_top_df = train_df[train_df['user_id'] == user_id]
    if user_top_df.empty:
        return f"We recommend this movie because it is one of the most popular items in the system."
        
    user_top = user_top_df.nlargest(10, 'rating')['movie_id'].tolist()
    
    # Get recommended movie title
    rec_title_row = movies_df[movies_df['movie_id'] == rec_movie_id]
    rec_title = rec_title_row['title'].values[0] if not rec_title_row.empty else f"Movie #{rec_movie_id}"
    
    explanation_movies = []
    for m in user_top[:n_explain]:
        title_row = movies_df[movies_df['movie_id'] == m]
        if not title_row.empty:
            explanation_movies.append(title_row['title'].values[0])
            
    if not explanation_movies:
        return f"We recommend '{rec_title}' based on similar user preferences."
        
    explanation = f"We recommend '{rec_title}' because users who highly rated "
    explanation += ", ".join([f"'{t}'" for t in explanation_movies])
    explanation += " also loved this film."
    
    return explanation
