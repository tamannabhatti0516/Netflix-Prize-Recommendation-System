import pandas as pd
import numpy as np
import os
from pathlib import Path

# Resolve base directories dynamically so paths work from notebooks, dashboard, and CLI
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

def parse_netflix_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single Netflix Prize combined_data file.
    Returns DataFrame with columns: [movie_id, user_id, rating, date]
    """
    records = []
    current_movie_id = None
    
    print(f"Opening file: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(',')
                user_id = int(parts[0])
                rating = int(parts[1])
                date = parts[2]
                records.append((current_movie_id, user_id, rating, date))
    
    df = pd.DataFrame(records, columns=['movie_id', 'user_id', 'rating', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df['rating'] = df['rating'].astype(np.int8)
    df['movie_id'] = df['movie_id'].astype(np.int16)
    return df


def load_all_data(raw_dir: str = None, sample_size: int = 5_000_000) -> pd.DataFrame:
    """
    Load and parse all 4 combined_data files.
    Concatenate, then take a stratified random sample of `sample_size` rows.
    Saves processed sample to parquet for fast reloading.
    """
    if raw_dir is None:
        raw_dir = str(DEFAULT_RAW_DIR)
        
    os.makedirs(str(DEFAULT_PROCESSED_DIR), exist_ok=True)
    processed_path = os.path.join(str(DEFAULT_PROCESSED_DIR), 'ratings_sample.parquet')
    
    # If already processed, load from cache
    if os.path.exists(processed_path):
        print(f"Loading from cached parquet: {processed_path}")
        return pd.read_parquet(processed_path)
    
    print("Parsing raw files (this takes a few minutes)...")
    dfs = []
    for i in range(1, 5):
        filepath = os.path.join(raw_dir, f'combined_data_{i}.txt')
        if os.path.exists(filepath):
            print(f"  Parsing combined_data_{i}.txt ...")
            df_part = parse_netflix_file(filepath)
            dfs.append(df_part)
            print(f"  → {len(df_part):,} records")
        else:
            print(f"  Warning: File not found: {filepath}")
    
    if not dfs:
        raise FileNotFoundError(f"No combined_data_*.txt files found in {raw_dir}")
        
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(df_full):,}")
    
    # Stratified sample — preserve rating distribution
    print(f"Taking stratified sample of {sample_size:,} records...")
    df_sample = df_full.groupby('rating', group_keys=False).apply(
        lambda x: x.sample(frac=sample_size/len(df_full), random_state=42)
    ).reset_index(drop=True)
    
    # If sample is slightly off due to rounding, trim
    if len(df_sample) != sample_size:
        df_sample = df_sample.sample(min(sample_size, len(df_sample)), random_state=42).reset_index(drop=True)
    
    df_sample.to_parquet(processed_path, index=False)
    print(f"Saved {len(df_sample):,} rows to {processed_path}")
    
    return df_sample


def load_movie_titles(raw_dir: str = None) -> pd.DataFrame:
    """
    Load movie_titles.csv with columns: [movie_id, year, title]
    Handles encoding issues gracefully.
    """
    if raw_dir is None:
        raw_dir = str(DEFAULT_RAW_DIR)
        
    filepath = os.path.join(raw_dir, 'movie_titles.csv')
    print(f"Loading movie titles from: {filepath}")
    
    movies = pd.read_csv(
        filepath,
        encoding='latin-1',
        header=None,
        names=['movie_id', 'year', 'title'],
        on_bad_lines='skip'
    )
    movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce')
    movies = movies.dropna(subset=['movie_id'])
    movies['movie_id'] = movies['movie_id'].astype(int)
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    return movies


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Temporal train-test split.
    All ratings before the 80th percentile date → train
    All ratings after → test
    """
    # Use 80th percentile as cutoff
    cutoff = df['date'].quantile(0.80)
    print(f"Splitting data temporally with cutoff date: {cutoff}")
    train = df[df['date'] <= cutoff].reset_index(drop=True)
    test = df[df['date'] > cutoff].reset_index(drop=True)
    
    # Only keep test users and movies that appear in train (cold-start filter)
    train_users = set(train['user_id'].unique())
    train_movies = set(train['movie_id'].unique())
    test = test[
        test['user_id'].isin(train_users) & 
        test['movie_id'].isin(train_movies)
    ].reset_index(drop=True)
    
    os.makedirs(str(DEFAULT_PROCESSED_DIR), exist_ok=True)
    train_path = os.path.join(str(DEFAULT_PROCESSED_DIR), 'train.parquet')
    test_path = os.path.join(str(DEFAULT_PROCESSED_DIR), 'test.parquet')
    
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Train users: {train['user_id'].nunique():,} | Test users: {test['user_id'].nunique():,}")
    return train, test
