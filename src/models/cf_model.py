from surprise import KNNWithMeans
from src.models.svd_model import build_surprise_dataset
import pickle
import os
from pathlib import Path

# Resolve models folder path dynamically relative to file location
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'models'
DEFAULT_USER_CF_PATH = MODELS_DIR / 'user_cf_model.pkl'
DEFAULT_ITEM_CF_PATH = MODELS_DIR / 'item_cf_model.pkl'

def train_user_cf(train_df, k=40, similarity='cosine'):
    """
    User-Based Collaborative Filtering.
    Finds users with similar rating patterns and recommends what similar users liked.
    """
    data = build_surprise_dataset(train_df)
    trainset = data.build_full_trainset()
    
    model = KNNWithMeans(
        k=k,
        sim_options={
            'name': similarity,      # 'cosine', 'pearson', or 'msd'
            'user_based': True,      # User-based CF
            'min_support': 3         # Minimum common ratings required
        },
        verbose=True
    )
    print(f"Training User-Based CF (k={k}, similarity={similarity})...")
    model.fit(trainset)
    return model


def train_item_cf(train_df, k=40, similarity='cosine'):
    """
    Item-Based Collaborative Filtering.
    Recommends items similar to what the user has rated highly.
    """
    data = build_surprise_dataset(train_df)
    trainset = data.build_full_trainset()
    
    model = KNNWithMeans(
        k=k,
        sim_options={
            'name': similarity,
            'user_based': False,     # Item-based CF
            'min_support': 3
        },
        verbose=True
    )
    print(f"Training Item-Based CF (k={k}, similarity={similarity})...")
    model.fit(trainset)
    return model

def save_cf_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"CF model successfully saved to {path}")

def load_cf_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
