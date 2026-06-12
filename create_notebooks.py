import nbformat as nbf
import os
from pathlib import Path

# Resolve notebook directory
PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
os.makedirs(str(NOTEBOOKS_DIR), exist_ok=True)

# -------------------------------------------------------------
# Notebook 1: EDA
# -------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("# 🎬 01 — Exploratory Data Analysis\nThis notebook loads the Netflix Prize dataset sample (5,000,000 ratings) and generates key summary statistics and visualizations covering rating distributions, user activity, movie popularity, and temporal trends."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
from src.data_pipeline import load_all_data, load_movie_titles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load dataset and titles
df = load_all_data(sample_size=5_000_000)
movies = load_movie_titles()
df_merged = df.merge(movies, on='movie_id', how='left')

print("Sampled data shape:", df.shape)
print("Merge data shape:", df_merged.shape)
print(df_merged.head())"""),
    nbf.v4.new_markdown_cell("## 2.2 Dataset Summary Stats"),
    nbf.v4.new_code_cell("""total_ratings = len(df)
unique_users = df['user_id'].nunique()
unique_movies = df['movie_id'].nunique()
min_date = df['date'].min()
max_date = df['date'].max()
mean_rating = df['rating'].mean()
median_rating = df['rating'].median()
std_rating = df['rating'].std()
sparsity = 1 - (total_ratings / (unique_users * unique_movies))

stats = pd.DataFrame({
    'Metric': ['Total Ratings', 'Unique Users', 'Unique Movies', 'Date Range Start', 'Date Range End', 'Mean Rating', 'Median Rating', 'Std Deviation', 'Sparsity'],
    'Value': [total_ratings, unique_users, unique_movies, str(min_date.date()), str(max_date.date()), f"{mean_rating:.4f}", int(median_rating), f"{std_rating:.4f}", f"{sparsity*100:.4f}%"]
})
print(stats.to_string(index=False))"""),
    nbf.v4.new_markdown_cell("## 2.3 Rating Distribution Plot"),
    nbf.v4.new_code_cell("""os.makedirs('../reports/figures/', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall distribution
rating_counts = df['rating'].value_counts().sort_index()
axes[0].bar(rating_counts.index, rating_counts.values, color='#E50914', edgecolor='black')
axes[0].set_title('Overall Rating Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rating (1-5 Stars)')
axes[0].set_ylabel('Number of Ratings')
for i, v in enumerate(rating_counts.values):
    axes[0].text(i+1, v + 10000, f'{v/1e6:.2f}M', ha='center', fontsize=9)

# Average rating per user (distribution)
user_avg = df.groupby('user_id')['rating'].mean()
axes[1].hist(user_avg, bins=50, color='#221F1F', edgecolor='white')
axes[1].set_title('Distribution of Average Ratings per User', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Rating')
axes[1].set_ylabel('Number of Users')

plt.tight_layout()
plt.savefig('../reports/figures/rating_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    nbf.v4.new_markdown_cell("## 2.4 User Activity Analysis"),
    nbf.v4.new_code_cell("""ratings_per_user = df.groupby('user_id').size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ratings_per_user, bins=100, color='#E50914', log=True)
axes[0].set_title('Ratings per User (Log Scale)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Ratings')
axes[0].set_ylabel('Number of Users (log)')

# Cumulative: what % of users contribute X% of ratings
sorted_counts = ratings_per_user.sort_values(ascending=False)
cumulative = sorted_counts.cumsum() / sorted_counts.sum() * 100
user_percentile = np.arange(1, len(sorted_counts)+1) / len(sorted_counts) * 100
axes[1].plot(user_percentile, cumulative, color='#E50914', linewidth=2)
axes[1].axhline(80, color='gray', linestyle='--', label='80% of ratings')
axes[1].set_title('User Activity: Cumulative Rating Contribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Top X% of Users')
axes[1].set_ylabel('% of Total Ratings')
axes[1].legend()

plt.tight_layout()
plt.savefig('../reports/figures/user_activity.png', dpi=150, bbox_inches='tight')
plt.show()

top_10_contrib = cumulative.iloc[int(len(cumulative)*0.1)]
users_lt5 = (ratings_per_user < 5).sum()
users_lt5_pct = (ratings_per_user < 5).mean() * 100
print(f"Top 10% of users contribute: {top_10_contrib:.1f}% of ratings")
print(f"Users with < 5 ratings: {users_lt5:,} ({users_lt5_pct:.1f}%)")"""),
    nbf.v4.new_markdown_cell("## 2.5 Movie Popularity Trends"),
    nbf.v4.new_code_cell("""ratings_per_movie = df.groupby('movie_id').size().reset_index(name='rating_count')
top_movies = ratings_per_movie.nlargest(20, 'rating_count').merge(movies, on='movie_id')

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(top_movies['title'].str[:40], top_movies['rating_count'], color='#E50914')
ax.set_title('Top 20 Most Rated Movies', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Ratings')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/top_movies.png', dpi=150, bbox_inches='tight')
plt.show()

# Long tail: distribution of ratings per movie
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ratings_per_movie['rating_count'], bins=200, log=True, color='#221F1F', edgecolor='white')
ax.set_title('Long Tail: Ratings per Movie Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Ratings per Movie')
ax.set_ylabel('Number of Movies (log)')
plt.tight_layout()
plt.savefig('../reports/figures/long_tail.png', dpi=150, bbox_inches='tight')
plt.show()

pct_under_100 = (ratings_per_movie['rating_count'] < 100).mean() * 100
print(f"Movies with fewer than 100 ratings: {pct_under_100:.1f}%")"""),
    nbf.v4.new_markdown_cell("## 2.6 Temporal Analysis"),
    nbf.v4.new_code_cell("""df['year_month'] = df['date'].dt.to_period('M')
monthly = df.groupby('year_month').size()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

monthly.plot(ax=axes[0], color='#E50914', linewidth=1.5)
axes[0].set_title('Monthly Rating Volume Over Time', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Number of Ratings')

df['year_rated'] = df['date'].dt.year
yearly_avg = df.groupby('year_rated')['rating'].mean()
yearly_avg.plot(ax=axes[1], marker='o', color='#E50914', linewidth=2)
axes[1].set_title('Average Rating by Year', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Rating')
axes[1].set_ylim(1, 5)

plt.tight_layout()
plt.savefig('../reports/figures/temporal_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    nbf.v4.new_markdown_cell("## 2.7 Sparsity Visualization"),
    nbf.v4.new_code_cell("""n_sample_users = 200
n_sample_movies = 200

sample_users = df['user_id'].value_counts().head(n_sample_users).index
sample_movies = df['movie_id'].value_counts().head(n_sample_movies).index

matrix_sample = df[
    df['user_id'].isin(sample_users) & df['movie_id'].isin(sample_movies)
].pivot_table(index='user_id', columns='movie_id', values='rating')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(matrix_sample.notna(), cmap='Reds', cbar=False, ax=ax,
            xticklabels=False, yticklabels=False)
ax.set_title(f'User-Item Interaction Matrix\\n(Top {n_sample_users} users x Top {n_sample_movies} movies)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Movies')
ax.set_ylabel('Users')
plt.tight_layout()
plt.savefig('../reports/figures/sparsity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    nbf.v4.new_markdown_cell("## 2.8 Save EDA Insights to JSON"),
    nbf.v4.new_code_cell("""os.makedirs('../reports/results/', exist_ok=True)
eda_insights = {
    "total_ratings_sampled": len(df),
    "unique_users": int(df['user_id'].nunique()),
    "unique_movies": int(df['movie_id'].nunique()),
    "sparsity_pct": float(round(sparsity * 100, 4)),
    "mean_rating": float(round(df['rating'].mean(), 4)),
    "median_rating": float(df['rating'].median()),
    "std_rating": float(round(df['rating'].std(), 4)),
    "date_range_start": str(df['date'].min()),
    "date_range_end": str(df['date'].max()),
    "users_with_lt5_ratings_pct": float(round(users_lt5_pct, 2)),
    "movies_with_lt100_ratings_pct": float(round(pct_under_100, 2)),
}

with open('../reports/results/eda_insights.json', 'w') as f:
    json.dump(eda_insights, f, indent=2)

print(json.dumps(eda_insights, indent=2))""")
]

# -------------------------------------------------------------
# Notebook 2: Preprocessing
# -------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# 🎬 02 — Data Preprocessing\nFilters out low activity users (<5 ratings) and rare movies (<10 ratings) to ensure Collaborative Filtering density. Then performs a temporal split (80/20) and stores the train/test sets to Parquet format."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
from src.data_pipeline import load_all_data, load_movie_titles, train_test_split_temporal

# Load cached stratified sample
df = load_all_data(sample_size=5_000_000)
movies = load_movie_titles()

# Keep active users and movies
user_counts = df['user_id'].value_counts()
valid_users = user_counts[user_counts >= 5].index
df = df[df['user_id'].isin(valid_users)].reset_index(drop=True)

movie_counts = df['movie_id'].value_counts()
valid_movies = movie_counts[movie_counts >= 10].index
df = df[df['movie_id'].isin(valid_movies)].reset_index(drop=True)

print(f"After filtering: {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['movie_id'].nunique():,} movies")

# Perform temporal split (split based on 80th percentile date)
train, test = train_test_split_temporal(df)""")
]

# -------------------------------------------------------------
# Notebook 3: SVD Model
# -------------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("# 🎬 03 — SVD Matrix Factorization\nImplements hyperparameter tuning using GridSearchCV on a subset of the training set. Trains the final SVD model using the optimal configuration, and saves the weights as a pickle file."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
import pandas as pd
from surprise.model_selection import GridSearchCV
from surprise import SVD
from src.models.svd_model import train_svd, save_model, build_surprise_dataset, load_model
from src.evaluation import compute_rmse

# Read splits
train_df = pd.read_parquet('../data/processed/train.parquet')
test_df = pd.read_parquet('../data/processed/test.parquet')

# Grid search on a sample to optimize training speed
gs_sample = train_df.sample(min(100000, len(train_df)), random_state=42)
data = build_surprise_dataset(gs_sample)

param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [15, 20],
    'lr_all': [0.005, 0.007],
    'reg_all': [0.02, 0.05]
}

print("Running GridSearchCV for SVD...")
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=2, n_jobs=-1)
gs.fit(data)

print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
print(f"Best Params: {gs.best_params['rmse']}")

best_params = gs.best_params['rmse']

# Train final model on full trainset
final_svd = train_svd(train_df, **best_params)
save_model(final_svd, '../models/svd_model.pkl')""")
]

# -------------------------------------------------------------
# Notebook 4: CF Model
# -------------------------------------------------------------
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("# 🎬 04 — Collaborative Filtering\nTrains memory-based User-CF and Item-CF models using `scikit-surprise`. Uses sampling (200,000 rows) to keep training fast, memory-safe, and avoid Out Of Memory crashes on standard systems."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
import pandas as pd
from src.models.cf_model import train_user_cf, train_item_cf, save_cf_model
from src.models.svd_model import build_surprise_dataset

train_df = pd.read_parquet('../data/processed/train.parquet')

# For User-Based CF, sample users to make the user-user similarity matrix memory-safe
unique_users = train_df['user_id'].unique()
user_sample = pd.Series(unique_users).sample(n=min(8000, len(unique_users)), random_state=42)
user_cf_sample = train_df[train_df['user_id'].isin(user_sample)]

print("Training User-Based CF...")
user_cf = train_user_cf(user_cf_sample, k=20, similarity='cosine')
save_cf_model(user_cf, '../models/user_cf_model.pkl')

# For Item-Based CF, sample ratings randomly (item-item similarity matrix is small and memory-safe)
item_cf_sample = train_df.sample(min(200000, len(train_df)), random_state=42)

print("Training Item-Based CF...")
item_cf = train_item_cf(item_cf_sample, k=20, similarity='cosine')
save_cf_model(item_cf, '../models/item_cf_model.pkl')""")
]

# -------------------------------------------------------------
# Notebook 5: NCF Model
# -------------------------------------------------------------
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("# 🎬 05 — Neural Collaborative Filtering (NCF)\nTrains a custom PyTorch Deep Learning recommendation model. Includes embedding layers for users and items, fed into multi-layer perceptron (MLP) layers, and trains the network in regression mode."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.ncf_model import train_ncf, save_ncf_model

train_df = pd.read_parquet('../data/processed/train.parquet')

# Sample for neural model to run fast (~1-2 minutes)
ncf_sample = train_df.sample(min(1000000, len(train_df)), random_state=42)

print("Starting NCF training...")
model, user2idx, movie2idx, losses = train_ncf(
    ncf_sample,
    n_epochs=5,
    batch_size=2048,
    lr=0.001,
    n_factors=32,
    layers=[64, 32]
)

# Save weights and index mappings
save_ncf_model(model, user2idx, movie2idx, '../models/ncf_model.pt', '../models/ncf_mappings.pkl')

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses)+1), losses, marker='o', color='#E50914', linewidth=2)
plt.title('NCF Training Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.tight_layout()
plt.savefig('../reports/figures/ncf_loss.png', dpi=150, bbox_inches='tight')
plt.show()""")
]

# -------------------------------------------------------------
# Notebook 6: Evaluation
# -------------------------------------------------------------
nb6 = nbf.v4.new_notebook()
nb6.cells = [
    nbf.v4.new_markdown_cell("# 🎬 06 — Model Evaluation and Comparison\nLoads SVD, User-CF, Item-CF, and NCF models. Computes comparative metrics (RMSE, MAE, MAP@10, Precision@10, Recall@10) on a test subset and saves them as a JSON results file. Produces comparison plots."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from src.models.svd_model import load_model
from src.models.cf_model import load_cf_model
from src.models.ncf_model import load_ncf_model
from src.evaluation import compute_rmse, compute_mae, compute_map_at_k, compute_precision_recall_at_k
from src.data_pipeline import load_movie_titles

# Load splits and titles
train_df = pd.read_parquet('../data/processed/train.parquet')
test_df = pd.read_parquet('../data/processed/test.parquet')
movies = load_movie_titles()
all_movie_ids = train_df['movie_id'].unique().tolist()

# Load trained models
print("Loading all trained models...")
svd_model = load_model('../models/svd_model.pkl')
user_cf_model = load_cf_model('../models/user_cf_model.pkl')
item_cf_model = load_cf_model('../models/item_cf_model.pkl')
ncf_wrapper = load_ncf_model(n_factors=32, layers=[64, 32], model_path='../models/ncf_model.pt', mappings_path='../models/ncf_mappings.pkl')

# Subsample test set for fast evaluation of RMSE/MAE
eval_test = test_df.sample(min(100000, len(test_df)), random_state=42)
testset_surprise = list(zip(eval_test['user_id'], eval_test['movie_id'], eval_test['rating']))

results = {}

# Evaluate SVD
print("\\nEvaluating SVD...")
start_time = time.time()
svd_preds = svd_model.test(testset_surprise)
svd_rmse = compute_rmse(svd_preds)
svd_mae = compute_mae(svd_preds)
svd_prec, svd_rec = compute_precision_recall_at_k(svd_preds, k=10)
svd_map = compute_map_at_k(svd_model, eval_test, train_df, all_movie_ids, k=10)
results['SVD'] = {
    'RMSE': float(svd_rmse), 'MAE': float(svd_mae), 'MAP@10': float(svd_map),
    'Precision@10': float(svd_prec), 'Recall@10': float(svd_rec)
}

# Evaluate User-CF
print("\\nEvaluating User-CF...")
ucf_preds = user_cf_model.test(testset_surprise)
ucf_rmse = compute_rmse(ucf_preds)
ucf_mae = compute_mae(ucf_preds)
ucf_prec, ucf_rec = compute_precision_recall_at_k(ucf_preds, k=10)
ucf_map = compute_map_at_k(user_cf_model, eval_test, train_df, all_movie_ids, k=10)
results['User-CF'] = {
    'RMSE': float(ucf_rmse), 'MAE': float(ucf_mae), 'MAP@10': float(ucf_map),
    'Precision@10': float(ucf_prec), 'Recall@10': float(ucf_rec)
}

# Evaluate Item-CF
print("\\nEvaluating Item-CF...")
icf_preds = item_cf_model.test(testset_surprise)
icf_rmse = compute_rmse(icf_preds)
icf_mae = compute_mae(icf_preds)
icf_prec, icf_rec = compute_precision_recall_at_k(icf_preds, k=10)
icf_map = compute_map_at_k(item_cf_model, eval_test, train_df, all_movie_ids, k=10)
results['Item-CF'] = {
    'RMSE': float(icf_rmse), 'MAE': float(icf_mae), 'MAP@10': float(icf_map),
    'Precision@10': float(icf_prec), 'Recall@10': float(icf_rec)
}

# Evaluate NCF
print("\\nEvaluating NCF...")
ncf_preds_list = []
# Subsample for NCF inference speed
ncf_eval_sample = eval_test.head(10000)
for uid, iid, r in zip(ncf_eval_sample['user_id'], ncf_eval_sample['movie_id'], ncf_eval_sample['rating']):
    ncf_preds_list.append((uid, iid, r, ncf_wrapper.predict(uid, iid).est, None))
    
ncf_rmse = compute_rmse(ncf_preds_list)
ncf_mae = compute_mae(ncf_preds_list)
ncf_prec, ncf_rec = compute_precision_recall_at_k(ncf_preds_list, k=10)
ncf_map = compute_map_at_k(ncf_wrapper, ncf_eval_sample, train_df, all_movie_ids, k=10)
results['NCF'] = {
    'RMSE': float(ncf_rmse), 'MAE': float(ncf_mae), 'MAP@10': float(ncf_map),
    'Precision@10': float(ncf_prec), 'Recall@10': float(ncf_rec)
}

# Display Table
results_df = pd.DataFrame(results).T
print(results_df.round(4).to_string())

# Save comparisons
os.makedirs('../reports/results/', exist_ok=True)
with open('../reports/results/model_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
results_df['RMSE'].plot(kind='bar', ax=axes[0], color='#E50914', title='RMSE (lower is better)')
results_df['MAP@10'].plot(kind='bar', ax=axes[1], color='#221F1F', title='MAP@10 (higher is better)')
for ax in axes:
    ax.set_xlabel('Model')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('../reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()""")
]

# -------------------------------------------------------------
# Notebook 7: Recommendations Analysis
# -------------------------------------------------------------
nb7 = nbf.v4.new_notebook()
nb7.cells = [
    nbf.v4.new_markdown_cell("# 🎬 07 — Recommendations Analysis\nVisualizes and examines recommendation recommendations for users. Identifies 5 success cases (with taste analyses), 3 failure cases (with diagnostic notes), and computes popularity bias on the top recommendations across 100 power users."),
    nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
import pandas as pd
import numpy as np
from src.models.svd_model import load_model
from src.data_pipeline import load_movie_titles
from src.recommender import get_top_k_recommendations, get_user_history, explain_recommendation
from collections import Counter

# Load data
train_df = pd.read_parquet('../data/processed/train.parquet')
movies = load_movie_titles()
svd_model = load_model('../models/svd_model.pkl')
all_movie_ids = train_df['movie_id'].unique().tolist()

# Find power users
power_users = train_df['user_id'].value_counts().head(100).index.tolist()

# SUCCESS CASES (5 Users)
print("SUCCESS CASES (5 Users):")
for idx, user_id in enumerate(power_users[:5]):
    print(f"\\n{'='*60}")
    print(f"USER {user_id} - Top Rated Movies in History:")
    history = get_user_history(user_id, train_df, movies, n=10)
    print(history.to_string(index=False))
    
    print(f"\\nTop-10 Recommendations for User {user_id}:")
    seen = set(train_df[train_df['user_id'] == user_id]['movie_id'])
    recs = get_top_k_recommendations(svd_model, user_id, all_movie_ids, seen, movies, k=10)
    print(recs.to_string(index=False))
    
    print(f"\\nTaste Analysis & Explainability:")
    print(explain_recommendation(user_id, recs['movie_id'].iloc[0], svd_model, train_df, movies))

# FAILURE CASES (3 Users)
print(f"\\n{'='*60}")
print("DIAGNOSING FAILURE CASES:")
few_ratings_users = train_df['user_id'].value_counts().tail(5).index.tolist()
for user_id in few_ratings_users[:3]:
    print(f"\\nUSER {user_id} (Low Activity - Ratings Count = {len(train_df[train_df['user_id'] == user_id])}):")
    history = get_user_history(user_id, train_df, movies, n=5)
    print(history.to_string(index=False))
    
    seen = set(train_df[train_df['user_id'] == user_id]['movie_id'])
    recs = get_top_k_recommendations(svd_model, user_id, all_movie_ids, seen, movies, k=5)
    print("Recommendations:")
    print(recs.to_string(index=False))
    print("Diagnosis: With very few ratings, the model relies heavily on global biases rather than personalized tastes. It recommends highly popular films with generic positive ratings. A content-based approach or active user preference elicitation is required to solve this.")

# POPULARITY BIAS
print(f"\\n{'='*60}")
print("POPULARITY BIAS ANALYSIS:")
rec_movie_ids = []
for user_id in power_users[:100]:
    seen = set(train_df[train_df['user_id'] == user_id]['movie_id'])
    recs = get_top_k_recommendations(svd_model, user_id, all_movie_ids, seen, movies, k=10)
    rec_movie_ids.extend(recs['movie_id'].tolist() if 'movie_id' in recs.columns else [])

rec_freq = Counter(rec_movie_ids)
print("\\nMost frequently recommended movies across 100 power users:")
top_rec_movies = pd.DataFrame(rec_freq.most_common(20), columns=['movie_id', 'times_recommended'])
top_rec_movies = top_rec_movies.merge(movies, on='movie_id')
print(top_rec_movies[['title', 'times_recommended']].to_string(index=False))""")
]

# Write out all notebooks
notebooks = {
    '01_EDA.ipynb': nb1,
    '02_Preprocessing.ipynb': nb2,
    '03_SVD_Model.ipynb': nb3,
    '04_CF_Model.ipynb': nb4,
    '05_NCF_Model.ipynb': nb5,
    '06_Evaluation_Comparison.ipynb': nb6,
    '07_Recommendations_Analysis.ipynb': nb7
}

for name, nb in notebooks.items():
    path = NOTEBOOKS_DIR / name
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {path}")
