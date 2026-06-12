# 🎬 Netflix Prize Recommendation System
**Cult Open Projects 2026 — Problem Statement 1**

A complete, end-to-end personalized movie recommendation engine built on the Netflix Prize Dataset (100M+ ratings). Implements SVD Matrix Factorization, User-Based CF, Item-Based CF, and Neural Collaborative Filtering with a fully interactive Streamlit dashboard.

## Results Summary

| Model | RMSE | MAP@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|
| SVD (Matrix Factorization) | 0.9626 | 0.0102 | 0.5087 | 0.5306 |
| User-Based CF | 1.0703 | 0.0016 | 0.5798 | 0.6728 |
| Item-Based CF | 1.0699 | 0.0011 | 0.5416 | 0.5909 |
| NCF (Neural) | 1.0442 | 0.0006 | 0.4591 | 0.4653 |

## Setup

### 1. Clone and install
```bash
git clone [<your-repo-url>](https://github.com/tamannabhatti0516/Netflix-Prize-Recommendation-System.git)
cd netflix-recommendation-system
pip install -r requirements.txt
```

### 2. Run preprocessing and notebooks
The project contains raw text files in `./data/raw/` (unzipped Netflix Prize dataset). To create notebooks, preprocess data, train models, and run predictions, run:
```bash
# 1. Programmatically generate all Jupyter notebooks
python create_notebooks.py

# 2. Run the notebook execution pipeline (takes ~5 minutes, memory-safe)
python run_notebooks.py
```

### 3. Launch dashboard
To start the Netflix-themed Streamlit dashboard, run:
```bash
cd app
streamlit run app.py
```

### 4. Generate report & presentation
To generate the competition technical report and presentation PowerPoint slides, run:
```bash
cd reports
python generate_report.py
python generate_presentation.py
```

## Project Structure
```
netflix-recommendation-system/
│
├── data/
│   ├── raw/                        # Downloaded Kaggle files land here
│   ├── processed/
│   │   ├── ratings_sample.parquet  # Cleaned sample for fast iteration
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── movie_titles_clean.csv
│
├── src/
│   ├── data_pipeline.py            # Parsing, cleaning, sampling
│   ├── feature_engineering.py      # Derived features, temporal features
│   ├── evaluation.py               # RMSE, MAP@10, Precision@K, Recall@K
│   ├── recommender.py              # Top-K generation function
│   └── models/
│       ├── svd_model.py            # SVD Matrix Factorization
│       ├── cf_model.py             # User-Based and Item-Based CF
│       └── ncf_model.py            # Neural Collaborative Filtering (PyTorch)
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_SVD_Model.ipynb
│   ├── 04_CF_Model.ipynb
│   ├── 05_NCF_Model.ipynb
│   ├── 06_Evaluation_Comparison.ipynb
│   └── 07_Recommendations_Analysis.ipynb
│
├── app/
│   ├── app.py                      # Streamlit dashboard
│   └── assets/
│       └── style.css
│
├── reports/
│   ├── figures/                    # All saved plots (PNG)
│   ├── results/                    # Saved metrics JSON
│   ├── technical_report.html       # W3C-styled HTML report
│   ├── technical_report.pdf        # Generated PDF Technical Report
│   ├── presentation.pptx           # 8-slide PowerPoint Presentation
│   ├── generate_report.py          # HTML report compiler
│   └── generate_presentation.py    # Widescreen PPTX builder
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Evaluation Methodology
- **Train-Test Split:** Temporal split at 80th percentile date.
- **MAP@10 Relevance Threshold:** rating &ge; 3.5 (as per competition spec).
- **Sample Size:** 5,000,000 ratings (stratified sample of full dataset).
- **Random Seed:** 42 throughout.
