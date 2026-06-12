from surprise import accuracy
import numpy as np
from collections import defaultdict
import torch

def compute_rmse(predictions):
    """Standard RMSE from Surprise predictions list."""
    return accuracy.rmse(predictions, verbose=False)

def compute_mae(predictions):
    """Standard MAE from Surprise predictions list."""
    return accuracy.mae(predictions, verbose=False)


def compute_map_at_k(model, test_df, train_df, all_movie_ids, k=10, threshold=3.5):
    """
    Mean Average Precision @ K
    
    For each user in test set:
    1. Get the movies they actually rated in the test set
    2. Generate Top-K recommendations from UNSEEN movies (not in train)
    3. Mark each rec as relevant if actual test rating >= threshold (3.5)
    4. Compute Average Precision for that user
    5. Average AP across all users = MAP@K
    
    Args:
        model: trained model (Surprise model, NCF model, or custom model wrapper)
        test_df: test set DataFrame [user_id, movie_id, rating]
        train_df: train set DataFrame (to know what user already saw)
        all_movie_ids: list or array of all movie IDs in the system
        k: cutoff (10 as per competition)
        threshold: relevance threshold (3.5 as per competition)
    
    Returns:
        float: MAP@K score
    """
    # Build lookup: user → set of movies rated in train
    user_train_movies = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()
    
    # Build lookup: user → {movie: actual_rating} in test
    user_test_ratings = defaultdict(dict)
    for _, row in test_df.iterrows():
        user_test_ratings[row['user_id']][row['movie_id']] = row['rating']
    
    # Only evaluate users who have at least 1 relevant item in test
    eligible_users = [
        uid for uid, ratings in user_test_ratings.items()
        if any(r >= threshold for r in ratings.values())
    ]
    
    # If the set of eligible users is too large, we can evaluate on a large random subset (e.g., 1000 users) 
    # to avoid extremely long computation times during notebook runs.
    max_eval_users = 1000
    if len(eligible_users) > max_eval_users:
        np.random.seed(42)
        eligible_users = np.random.choice(eligible_users, max_eval_users, replace=False)
        print(f"Subsampled {max_eval_users:,} eligible users for MAP@{k} calculation to optimize runtime.")
    else:
        print(f"Evaluating MAP@{k} on {len(eligible_users):,} eligible users...")
    
    average_precisions = []
    
    # Check if NCF model
    is_ncf = hasattr(model, 'predict_ncf_batch')
    
    for idx, user_id in enumerate(eligible_users):
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(eligible_users)} users...")
            
        actual_ratings = user_test_ratings[user_id]
        seen_movies = user_train_movies.get(user_id, set())
        
        # Candidate movies: in all_movie_ids but NOT seen in training
        candidate_movies = [m for m in all_movie_ids if m not in seen_movies]
        
        # Predict ratings for all candidates
        if is_ncf:
            # Predict using PyTorch NCF batch predictor
            predictions = model.predict_ncf_batch(user_id, candidate_movies)
        else:
            # Standard Surprise model predict
            predictions = [
                (movie_id, model.predict(user_id, movie_id).est)
                for movie_id in candidate_movies
            ]
        
        # Sort by predicted rating descending, take top K
        top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
        
        # Compute Average Precision
        n_relevant = sum(1 for r in actual_ratings.values() if r >= threshold)
        
        hits = 0
        sum_precisions = 0.0
        
        for rank, (movie_id, _) in enumerate(top_k, 1):
            if movie_id in actual_ratings and actual_ratings[movie_id] >= threshold:
                hits += 1
                sum_precisions += hits / rank  # Precision@rank
        
        # AP = sum of precisions at hits / min(k, n_relevant)
        ap = sum_precisions / min(k, n_relevant) if n_relevant > 0 else 0.0
        average_precisions.append(ap)
    
    map_k = np.mean(average_precisions) if average_precisions else 0.0
    return map_k


def compute_precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Precision@K and Recall@K from predictions.
    Supports Surprise prediction format (tuples of uid, iid, true_r, est, details) 
    or custom tuple format.
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        uid = pred[0]
        true_r = pred[2]
        est = pred[3]
        user_est_true[uid].append((est, true_r))
    
    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum(1 for (_, tr) in user_ratings if tr >= threshold)
        n_rec_k = sum(1 for (est, _) in user_ratings[:k] if est >= threshold)
        n_rel_and_rec_k = sum(1 for (est, tr) in user_ratings[:k]
                               if tr >= threshold and est >= threshold)
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0
    
    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))
