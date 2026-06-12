import os
import sys
import time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

notebooks = [
    '01_EDA.ipynb',
    '02_Preprocessing.ipynb',
    '03_SVD_Model.ipynb',
    '04_CF_Model.ipynb',
    '05_NCF_Model.ipynb',
    '06_Evaluation_Comparison.ipynb',
    '07_Recommendations_Analysis.ipynb'
]

# Configure execution kernel and timeout
ep = ExecutePreprocessor(timeout=900, kernel_name='python3')

print("Starting execution of recommendation system notebooks...")
overall_start = time.time()

for nb_name in notebooks:
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"\n=======================================================")
    print(f"Running: {nb_name}...")
    start_time = time.time()
    
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook, setting the execution path to notebooks/
        ep.preprocess(nb, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        duration = time.time() - start_time
        print(f"Success! {nb_name} finished in {duration:.1f} seconds.")
    except Exception as e:
        print(f"Error executing {nb_name}: {e}")
        print("Stopping execution pipeline.")
        sys.exit(1)

total_duration = time.time() - overall_start
print(f"\nAll notebooks executed successfully in {total_duration/60:.1f} minutes!")
