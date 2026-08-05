# Data Directory

## Overview

This directory is intentionally kept **empty** (except for this README) in the public repository because the raw dataset is large (≥ 1 million rows, 60 columns) and exceeds GitHub's recommended file size limit.

---

## Dataset: Divar Real Estate Ads (`divar_real_estate_ads.csv`)

| Property | Value |
|---|---|
| Source | [Hugging Face — divar_real_estate_ads](https://huggingface.co/datasets/divar_real_estate_ads) |
| Rows | ~1,000,000+ real estate listings |
| Columns | 60 features (see below) |
| Language | Persian / Farsi |

### Key Columns

| Column | Description |
|---|---|
| `cat2_slug` | Listing type (residential rent, commercial sale, …) |
| `cat3_slug` | Property type (villa, apartment, land, …) |
| `city_slug`, `neighborhood_slug` | Location identifiers |
| `price_value` | Total price or price per m² |
| `rent_value` | Monthly rent amount |
| `credit_value` | Security deposit (Rahn) |
| `building_size` | Floor area in m² |
| `land_size` | Land area in m² |
| `rooms_count` | Number of bedrooms |
| `floor` | Floor number |
| `total_floors_count` | Total floors in the building |
| `construction_year` | Year built |
| `user_type` | Advertiser type: individual or agency |
| `description`, `title` | Free-text fields (Persian) |
| `has_parking`, `has_elevator`, … | Binary amenity flags |

---

## How to Use

1. Download the dataset from Hugging Face (link above) **or** request the CSV file from the project authors.
2. Place the raw file here as:
   ```
   data/divar_real_estate_ads.csv
   ```
3. The notebooks also use several intermediate `.pkl` checkpoints that are produced by running the notebooks in order:

| File | Produced by | Used by |
|---|---|---|
| `divar_tehran.csv` | `01_eda.ipynb` | `02_data_cleaning.ipynb` |
| `divar_tehran_tabdil.pkl` | `02_data_cleaning.ipynb` (Stage A) | `02_data_cleaning.ipynb` (Stage B) |
| `divar_tehran_num_filled.pkl` | `02_data_cleaning.ipynb` (Stage B) | `02_data_cleaning.ipynb` (Stage C) |
| `divar_tehran_nonnum_filled_final.pkl` | `02_data_cleaning.ipynb` (Stage C) | `03_clustering.ipynb`, `04_price_modeling.ipynb` |
| `divar_tehran_clustered.pkl` | `03_clustering.ipynb` | (optional downstream) |

> **Tip:** Run the notebooks in order (01 → 05) so that each intermediate file is available for the next stage.

---

## Large File Note

If you have a file over 100 MB, use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "data/*.csv"
git lfs track "data/*.pkl"
git add .gitattributes
```
