from surprise import SVD, Dataset, Reader
import pickle
import os
from pathlib import Path

# Resolve models folder path dynamically relative to file location
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'models'
DEFAULT_SVD_PATH = MODELS_DIR / 'svd_model.pkl'

def build_surprise_dataset(df):
    """Convert a Pandas DataFrame to a Surprise Dataset."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    return data

def train_svd(train_df, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Train SVD Matrix Factorization model.
    
    Why SVD: Best balance of accuracy and scalability for sparse explicit feedback.
    Learns latent user preferences and item characteristics simultaneously.
    """
    data = build_surprise_dataset(train_df)
    trainset = data.build_full_trainset()
    
    model = SVD(
        n_factors=n_factors,    # Number of latent dimensions
        n_epochs=n_epochs,      # Training iterations
        lr_all=lr_all,          # Learning rate for all parameters
        reg_all=reg_all,        # L2 regularization (prevent overfitting)
        random_state=42,
        verbose=True
    )
    print(f"Training SVD model (factors={n_factors}, epochs={n_epochs}, lr={lr_all}, reg={reg_all})...")
    model.fit(trainset)
    return model

def save_model(model, path=None):
    if path is None:
        path = str(DEFAULT_SVD_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"SVD model successfully saved to {path}")

def load_model(path=None):
    if path is None:
        path = str(DEFAULT_SVD_PATH)
    with open(path, 'rb') as f:
        return pickle.load(f)
