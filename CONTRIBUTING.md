# Contributing to Divar Real Estate Analysis

Thank you for your interest in this project! This guide explains how to set up your environment and contribute.

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kouroshkhn/divar-real-estate-analysis.git
cd divar-real-estate-analysis

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download the dataset (requires ~1 GB disk space)
python -c "
from datasets import load_dataset
ds = load_dataset('divarofficial/real_estate_ads')
ds['train'].to_csv('data/divar_real_estate_ads.csv', index=False)
print('Dataset saved to data/divar_real_estate_ads.csv')
"

# 5. Launch JupyterLab
jupyter lab
```

---

## 📓 Running Notebooks

Run notebooks **strictly in this order** — each stage produces files required by the next:

| Step | Notebook | Output File(s) |
|---|---|---|
| 1 | `01_eda.ipynb` | `data/divar_tehran.csv` |
| 2 | `02_data_cleaning.ipynb` | `data/divar_tehran_tabdil.pkl` → `num_filled.pkl` → `nonnum_filled_final.pkl` |
| 3 | `03_clustering.ipynb` | `data/divar_tehran_clustered.pkl` |
| 4 | `04_price_modeling.ipynb` | In-memory results |
| 5 | `05_text_classification.ipynb` | In-memory results |

> **Note on Notebook 05 (LSTM / Task B):** This notebook was originally run on Google Colab for GPU access. If running locally, ensure you have a CUDA-capable GPU or be prepared for long CPU training times. The `drive.mount()` cells are commented out — set file paths to `../data/` for local use.

---

## 🧑‍💻 Code Style

- Use **4 spaces** for indentation in Python code cells
- Keep markdown cells concise with clear section headings
- Prefer `pandas` vectorized operations over Python loops
- Comment non-obvious transformations with a brief explanation

---

## 🐛 Reporting Issues

Please open a [GitHub Issue](https://github.com/kouroshkhn/divar-real-estate-analysis/issues) with:
1. A clear description of the problem
2. The notebook and cell where the issue occurs
3. Your Python and library versions (`pip freeze`)

---

## 📬 Contact

For questions about the analysis methodology, contact us via the institution:
[D-Learn Data Processing & Analysis School](https://D-learn.ir)
