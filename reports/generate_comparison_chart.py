import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load or define the results
results = {
    "SVD": {
        "RMSE": 0.9626,
        "MAE": 0.7532,
        "MAP@10": 0.0102
    },
    "User-CF": {
        "RMSE": 1.0703,
        "MAE": 0.9028,
        "MAP@10": 0.0016
    },
    "Item-CF": {
        "RMSE": 1.0699,
        "MAE": 0.8742,
        "MAP@10": 0.0011
    },
    "NCF": {
        "RMSE": 1.0442,
        "MAE": 0.8204,
        "MAP@10": 0.0006
    }
}

results_df = pd.DataFrame(results).T

# Save plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot RMSE
results_df['RMSE'].plot(kind='bar', ax=axes[0], color='#E50914', title='RMSE (lower is better)')
# Plot MAP@10
results_df['MAP@10'].plot(kind='bar', ax=axes[1], color='#221F1F', title='MAP@10 (higher is better)')

for ax in axes:
    ax.set_xlabel('Model')
    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
print("Regenerated reports/figures/model_comparison.png successfully!")
