import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from pathlib import Path

# Resolve models folder path dynamically relative to file location
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'models'
DEFAULT_NCF_PATH = MODELS_DIR / 'ncf_model.pt'
DEFAULT_MAPPINGS_PATH = MODELS_DIR / 'ncf_mappings.pkl'

class RatingsDataset(Dataset):
    def __init__(self, df, user2idx, movie2idx):
        self.users = torch.LongTensor([user2idx[u] for u in df['user_id']])
        self.movies = torch.LongTensor([movie2idx[m] for m in df['movie_id']])
        self.ratings = torch.FloatTensor(df['rating'].values)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


class NCF(nn.Module):
    """
    Neural Collaborative Filtering.
    
    Architecture:
    - Separate embedding layers for users and items
    - Concatenated embeddings fed into MLP
    - Output: predicted rating (regression)
    
    Why NCF: Can capture non-linear user-item interactions that SVD misses.
    Trade-off: Slower to train, needs more data, less interpretable.
    """
    def __init__(self, n_users, n_items, n_factors=64, layers=[128, 64, 32]):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        mlp_layers = []
        input_size = n_factors * 2
        for layer_size in layers:
            mlp_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = layer_size
        mlp_layers.append(nn.Linear(input_size, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user, item):
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        x = torch.cat([u_emb, i_emb], dim=-1)
        # Scale output to [1, 5] using sigmoid and linear transformation
        return torch.sigmoid(self.mlp(x).squeeze(dim=-1)) * 4 + 1


def train_ncf(train_df, n_epochs=10, batch_size=2048, lr=0.001, n_factors=64, layers=[128, 64, 32]):
    """Full NCF training loop with loss tracking."""
    
    # Create user and movie index mappings
    users = train_df['user_id'].unique()
    movies = train_df['movie_id'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    movie2idx = {m: i for i, m in enumerate(movies)}
    
    dataset = RatingsDataset(train_df, user2idx, movie2idx)
    # Use 0 workers on Windows to prevent potential multiprocessing issues
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training NCF on device: {device}")
    
    model = NCF(len(users), len(movies), n_factors=n_factors, layers=layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_users, batch_movies, batch_ratings in loader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_ratings = batch_ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_users, batch_movies)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        rmse = np.sqrt(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | RMSE: {rmse:.4f}")
    
    # Return trained model, mappings, and losses
    return model, user2idx, movie2idx, train_losses


class Prediction:
    def __init__(self, est):
        self.est = est

class NCFWrapper:
    """
    NCFWrapper wraps the PyTorch model and lookup dictionaries.
    Implements a Surprise-like predict API and a fast batch prediction API.
    """
    def __init__(self, model, user2idx, movie2idx, device):
        self.model = model
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.device = device
        
    def predict(self, user_id, movie_id):
        # Fallback if user_id or movie_id is not in index map (cold start)
        if user_id not in self.user2idx or movie_id not in self.movie2idx:
            return Prediction(3.5)
            
        u_idx = self.user2idx[user_id]
        m_idx = self.movie2idx[movie_id]
        
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.LongTensor([u_idx]).to(self.device)
            m_tensor = torch.LongTensor([m_idx]).to(self.device)
            pred = self.model(u_tensor, m_tensor).item()
            
        return Prediction(pred)
        
    def predict_ncf_batch(self, user_id, movie_ids):
        """Batch predict rating for unseen items. Returns list of (movie_id, rating)."""
        if user_id not in self.user2idx:
            return [(m, 3.5) for m in movie_ids]
            
        u_idx = self.user2idx[user_id]
        
        valid_movie_ids = []
        valid_m_idxs = []
        invalid_predictions = []
        
        for m in movie_ids:
            if m in self.movie2idx:
                valid_movie_ids.append(m)
                valid_m_idxs.append(self.movie2idx[m])
            else:
                invalid_predictions.append((m, 3.5))
                
        if not valid_movie_ids:
            return invalid_predictions
            
        self.model.eval()
        with torch.no_grad():
            # Send in chunks to prevent memory issues if movie_ids is massive
            chunk_size = 50000
            preds = []
            for i in range(0, len(valid_movie_ids), chunk_size):
                chunk_m = valid_m_idxs[i:i+chunk_size]
                u_tensor = torch.LongTensor([u_idx] * len(chunk_m)).to(self.device)
                m_tensor = torch.LongTensor(chunk_m).to(self.device)
                chunk_preds = self.model(u_tensor, m_tensor).cpu().numpy().tolist()
                preds.extend(chunk_preds)
            
        valid_predictions = list(zip(valid_movie_ids, preds))
        return valid_predictions + invalid_predictions


def save_ncf_model(model, user2idx, movie2idx, model_path=None, mappings_path=None):
    if model_path is None:
        model_path = str(DEFAULT_NCF_PATH)
    if mappings_path is None:
        mappings_path = str(DEFAULT_MAPPINGS_PATH)
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(mappings_path), exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), model_path)
    
    # Save index mappings
    with open(mappings_path, 'wb') as f:
        pickle.dump({'user2idx': user2idx, 'movie2idx': movie2idx}, f)
        
    print(f"NCF model weights saved to {model_path}")
    print(f"NCF mappings saved to {mappings_path}")


def load_ncf_model(n_users=None, n_items=None, n_factors=32, layers=[64, 32], model_path=None, mappings_path=None):
    if model_path is None:
        model_path = str(DEFAULT_NCF_PATH)
    if mappings_path is None:
        mappings_path = str(DEFAULT_MAPPINGS_PATH)
        
    with open(mappings_path, 'rb') as f:
        maps = pickle.load(f)
    user2idx = maps['user2idx']
    movie2idx = maps['movie_id2idx'] if 'movie_id2idx' in maps else maps['movie2idx']
    
    if n_users is None:
        n_users = len(user2idx)
    if n_items is None:
        n_items = len(movie2idx)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NCF(n_users, n_items, n_factors, layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return NCFWrapper(model, user2idx, movie2idx, device)
