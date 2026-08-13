# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `reports/final_report_en.md` — comprehensive English technical report covering all 5 pipeline stages
- `CONTRIBUTING.md` — local setup guide, notebook order, code style, issue reporting
- `CHANGELOG.md` — this file
- `datasets` library added to `requirements.txt` for direct HuggingFace download
- `scipy`, `Pillow` added to `requirements.txt` (previously implicit dependencies)
- HuggingFace dataset badge in `README.md`
- GitHub release badge in `README.md`
- **Key Results at a Glance** summary table added to `README.md`
- `age_of_building` feature engineering documented in `README.md` notebook 04 section
- LSTM row added to Task A classification table in `README.md`
- Python download snippet added to `data/README.md`
- `.gitignore` extended with `*.keras`, `*.npz`, `wandb/`, `mlruns/`, `.mlflow/`
- Version upper bounds added to all `requirements.txt` entries
- **Limitations** section added to `reports/final_report_en.md`
- **Results Summary** table added to top of `reports/final_report_en.md`

### Changed
- `requirements.txt` reorganised into labelled groups (Core / Visualization / ML / NLP / etc.)
- `data/README.md` download section restructured as Option A (Python) and Option B (Manual)
- `README.md` notebook 05 description updated with Colab compatibility note

---

## [1.0.0] — 2026-08-05

### Added
- Initial public release
- 5 Jupyter notebooks covering the full data science pipeline:
  - `01_eda.ipynb` — Exploratory Data Analysis
  - `02_data_cleaning.ipynb` — 3-stage cleaning pipeline
  - `03_clustering.ipynb` — K-Means + PCA clustering
  - `04_price_modeling.ipynb` — Price regression models
  - `05_text_classification.ipynb` — Persian NLP + LSTM classification
- `README.md` with project overview and pipeline description
- `requirements.txt` with all Python dependencies
- `.gitignore` for data files, model weights, and editor files
- `data/README.md` with dataset download instructions
- Persian technical report (`final_report_fa.pdf`) — available in [v1.0 Release](https://github.com/kouroshkhn/divar-real-estate-analysis/releases/tag/v1.0)

[Unreleased]: https://github.com/kouroshkhn/divar-real-estate-analysis/compare/v1.0...HEAD
[1.0.0]: https://github.com/kouroshkhn/divar-real-estate-analysis/releases/tag/v1.0
