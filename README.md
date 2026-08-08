# Divar Real Estate Analysis 🏠

> **Comprehensive data science analysis of 1M+ Iranian real estate listings from [Divar](https://divar.ir)**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview

This project applies the full data science pipeline to a large-scale Persian real-estate dataset from **Divar.ir**, one of Iran's largest classifieds platforms. The dataset contains **1,000,000+ property listings** with 60 features, covering residential, commercial, and rental properties across all major Iranian cities.

> **Institution:** [D-Learn Data Processing & Analysis School](https://D-learn.ir) (مدرسه پردازش و تحلیل داده دقیقه)
>
> **Completed:** Autumn 2025 (1404 Iranian calendar)

---

## 🗂️ Repository Structure

```
divar-real-estate-analysis/
├── README.md                          ← You are here
├── requirements.txt                   ← All Python dependencies
├── .gitignore                         ← Git ignore rules
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb         ← Full cleaning pipeline (3 stages)
│   ├── 03_clustering.ipynb            ← K-Means clustering + PCA
│   ├── 04_price_modeling.ipynb        ← Price prediction models
│   └── 05_text_classification.ipynb   ← Persian NLP + text classification
│
├── data/
│   └── README.md                      ← Dataset download instructions
│
├── reports/
│   └── final_report_en.md             ← Comprehensive English technical report
│
└── images/                            ← Plots and figures (from notebooks)
```

> 📄 **PDF Reports & Assets** (Final Persian PDF + Presentation Slides) are available in the [v1.0 Release](https://github.com/kouroshkhn/divar-real-estate-analysis/releases/tag/v1.0).

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Hugging Face — Divar Real Estate Ads](https://huggingface.co/datasets/divarofficial/real_estate_ads) |
| **Rows** | ~1,000,000+ listings |
| **Columns** | 60 features |
| **Language** | Persian / Farsi |
| **Coverage** | All major Iranian cities |

> ⚠️ **The raw data is not included** in this repository due to its size (≥ 1 GB). See [`data/README.md`](data/README.md) for download instructions.

### Key Features

| Category | Features |
|---|---|
| **Location** | `city_slug`, `neighborhood_slug`, `location_latitude`, `location_longitude` |
| **Property type** | `cat2_slug` (listing type), `cat3_slug` (property type) |
| **Financial** | `price_value`, `rent_value`, `credit_value` |
| **Physical** | `building_size`, `land_size`, `rooms_count`, `floor`, `construction_year` |
| **Amenities** | `has_parking`, `has_elevator`, `has_warehouse`, `has_balcony`, … |
| **Text** | `title`, `description` (Persian free text) |
| **Metadata** | `user_type` (individual vs. agency), `created_at_month` |

---

## 🔬 Analysis Pipeline

Run the notebooks in order (01 → 05). Each notebook produces intermediate files used by the next.

### 01 — Exploratory Data Analysis [`01_eda.ipynb`](notebooks/01_eda.ipynb)

- Load the full 1M+ row dataset
- Inspect data types, missing value rates, and unique values per column
- Classify all 60 columns into 5 groups: Continuous Numeric, Discrete Numeric, Boolean, Categorical, Text
- Focus analysis on **Tehran** (largest and most complete subset)
- Initial correlation heatmap — reveals the need for deep cleaning before reliable correlations emerge
- **Output:** `data/divar_tehran.csv` (Tehran-only subset)

### 02 — Data Cleaning [`02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb)

A three-stage cleaning pipeline:

#### Part A — Text & Type Standardisation
- Detect and remove **8,790 duplicate records** from 981,675 high-quality rows
  - Strategy: separate "high-quality" records (with key numeric fields filled) from low-quality ones; deduplicate only within the high-quality subset
- Normalise **Persian/Arabic character variants** and digits to standard form
- Convert all `object`-typed boolean columns to proper Python `bool`
- Convert floor, year, and capacity columns from `object` to `int`/`float`
- **Output:** `data/divar_tehran_tabdil.pkl`

#### Part B — Numeric Column Imputation & Outlier Removal
- Drop columns with **>70% missing values** across all transaction types
- Remove unrealistic placeholder values (`price_value = 999,999,999,999,999`, `building_size = 10,000,000`, etc.)
- Apply **log1p + IQR** outlier fencing grouped by property type (`cat3_slug`) to handle right-skewed price/size distributions
- Hierarchical group-median imputation: neighbourhood → category → global
- **Output:** `data/divar_tehran_num_filled.pkl`

#### Part C — Non-Numeric Column Imputation
- `user_type`: fill NaN with mode (most frequent value)
- `cat*_slug` columns: fill NaN with `"missing"` (preserved as own category in OHE)
- All `has_*` boolean features: fill NaN with `False` ("silence = absence" principle)
- Final validation: zero NaN in all key columns
- **Output:** `data/divar_tehran_nonnum_filled_final.pkl`

### 03 — Clustering [`03_clustering.ipynb`](notebooks/03_clustering.ipynb)

Goal: group Tehran properties by structural and financial characteristics.

#### Key Results

| Metric | Value |
|---|---|
| **Algorithm** | K-Means |
| **Optimal K** | **8 clusters** (selected via Elbow method + cross-tabulation) |
| **Properties clustered** | ~833,000 |
| **Features used** | 11 (physical + financial + amenities) |
| **PCA components** | 8 (explain **90.22%** of variance) |
| **Cluster stability** | **96–100%** on 7 of 8 clusters |

#### Cluster Descriptions (K = 8)

| Cluster | Dominant profile |
|---|---|
| 0 | Mid-range residential apartments for sale |
| 1 | Residential apartments for rent |
| 2 | Mixed residential — small units |
| 3 | Commercial properties |
| 4 | Large rental villas with high `land_size` |
| 5 | Rental apartments (deposit/rent model) |
| 6 | High-value sale properties (large `building_size`, high `price_value`) |
| 7 | Small commercial / office rentals |

**Dimensionality reduction:** Direct 11D → 2D (PCA) retains only 42.73% variance (clusters overlap visually). Reducing to 8D first retains 90.22% and re-clustering at K=8 shows 96–100% stability on 7 of 8 clusters. Cluster 3 had 15.6% migration to Cluster 2 — these two are structurally similar in PCA space.

- **Output:** `data/divar_tehran_clustered.pkl`

### 04 — Price Modeling [`04_price_modeling.ipynb`](notebooks/04_price_modeling.ipynb)

**Scope:** Residential **sale** listings only (`cat2_slug == 'residential-sell'`) · **Target:** `price_value`

#### Feature Engineering
Two engineered features added beyond the raw columns:
- **`elevator_floor_effect`**: floor × 1.5 (if elevator present) or floor × 0.7 (no elevator) — captures that high floors without elevator access are less desirable
- **`neighborhood_avg_price`**: mean sale price per `neighborhood_slug` from training data — strong proxy for location prestige

#### Models Evaluated

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline — fast, interpretable |
| **KNN Regressor** (k=5) | Distance-based, no distributional assumption |
| **Decision Tree** | Captures non-linear splits |
| **MLP Neural Net** | 2 hidden layers (64, 32 units), 500 epochs |

- **Split:** 60% train / 20% validation / 20% test
- **Preprocessing:** `StandardScaler` for numeric, `OneHotEncoder` for categorical
- **Tuning:** `RandomizedSearchCV` on best-performing model
- **Price band classification:** listings are classified as Over-valued / Normal / Under-valued based on the gap between predicted and listed price

### 05 — Text Classification [`05_text_classification.ipynb`](notebooks/05_text_classification.ipynb)

**Language:** Persian (Farsi) · **Input:** `description` + `title` columns

#### NLP Pipeline
1. Concatenate `description` + `title`
2. Remove URLs, HTML, non-Persian punctuation
3. Normalise character variants with **[Hazm](https://github.com/sobhe/hazm)** normaliser
4. Tokenise → Stem → Remove stopwords (all via Hazm)
5. Feature extraction: TF-IDF, Bag-of-Words, Word2Vec (100-dim)

#### Two Classification Tasks

**Task A — Property Type (`cat3_slug`)**

| Model | Feature | Accuracy | Weighted F1 |
|---|---|---|---|
| Logistic Regression | TF-IDF | **0.829** | **0.826** |
| Random Forest | TF-IDF | — | — |
| Naïve Bayes | BoW | — | — |

**Winner: Logistic Regression + TF-IDF**

**Task B — Advertiser Type (`user_type`)**

This model was also used to **impute ~700,000 missing `user_type` values** in the dataset.

| Model | Accuracy | Weighted F1 | Minority-class Recall |
|---|---|---|---|
| LSTM + Embedding layer | **0.89** | **0.90** | **0.81** |

**Winner: LSTM with learned Embedding layer**

> **Colab note:** Part B was originally run on Google Colab for GPU access. The `drive.mount()` cells are kept but commented out. To run locally, set file paths to `../data/`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kouroshkhn/divar-real-estate-analysis.git
cd divar-real-estate-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from [Hugging Face](https://huggingface.co/datasets/divarofficial/real_estate_ads) and place it as `data/divar_real_estate_ads.csv`.

### 5. Run the notebooks in order

```
01_eda.ipynb  →  02_data_cleaning.ipynb  →  03_clustering.ipynb
             →  04_price_modeling.ipynb
             →  05_text_classification.ipynb
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn`, `missingno` | Visualization |
| `scikit-learn` | Preprocessing, clustering, regression, metrics |
| `tensorflow` / `keras` | LSTM model for text classification |
| `gensim` | Word2Vec embeddings |
| `hazm` | Persian NLP (tokenisation, stemming, stopwords) |
| `joblib`, `tqdm` | Utilities |

---

## 📋 Intermediate Files

Each notebook produces checkpoint files used by the next stage:

| File | Produced by | Used by |
|---|---|---|
| `divar_tehran.csv` | `01_eda.ipynb` | `02_data_cleaning.ipynb` |
| `divar_tehran_tabdil.pkl` | `02_data_cleaning.ipynb` (Part A) | `02_data_cleaning.ipynb` (Part B) |
| `divar_tehran_num_filled.pkl` | `02_data_cleaning.ipynb` (Part B) | `02_data_cleaning.ipynb` (Part C) |
| `divar_tehran_nonnum_filled_final.pkl` | `02_data_cleaning.ipynb` (Part C) | `03_clustering.ipynb`, `04_price_modeling.ipynb` |
| `divar_tehran_clustered.pkl` | `03_clustering.ipynb` | Optional downstream use |

---

## 📄 Reports

- [**Final Report (English)**](reports/final_report_en.md) — Comprehensive technical report in English Markdown
- [**Final Report (Persian PDF)**](https://github.com/kouroshkhn/divar-real-estate-analysis/releases/download/v1.0/final_report_fa.pdf) — Full technical report in Farsi
- [**Presentation Slides (PDF)**](https://github.com/kouroshkhn/divar-real-estate-analysis/releases/download/v1.0/Presentation.pdf) — Project presentation deck

---

## 📜 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgements

- Dataset: [Hugging Face — Divar Real Estate Ads](https://huggingface.co/datasets/divarofficial/real_estate_ads)
- Persian NLP: [Hazm library](https://github.com/sobhe/hazm)
- Institution: [D-Learn Data Processing & Analysis School](https://D-learn.ir)
