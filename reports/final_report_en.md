# Divar Real Estate Data Analysis — Technical Report

> **Comprehensive Data Science & Machine Learning Pipeline on 1M+ Real Estate Listings**  
> **Authors:** Khodaei, Hosseini, Taherinia  
> **Institution:** D-Learn Data Processing & Analysis School (*مدرسه پردازش و تحلیل داده دقیقه*)  
> **Dataset Source:** [Divar Real Estate Ads (Hugging Face)](https://huggingface.co/datasets/divarofficial/real_estate_ads)  

---

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Dataset Overview](#2-dataset-overview)
  - [Dataset Structure](#dataset-structure)
  - [Business Feature Categorization](#business-feature-categorization)
- [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [Initial Exploration](#initial-exploration)
  - [Feature Categorization by Data Type](#feature-categorization-by-data-type)
  - [Feature Relationship & Correlation Discovery](#feature-relationship--correlation-discovery)
- [4. Data Cleaning & Preparation](#4-data-cleaning--preparation)
  - [Handling Duplicates](#handling-duplicates)
  - [Text & Data Type Standardization](#text--data-type-standardization)
- [5. Property Listing Clustering](#5-property-listing-clustering)
  - [Preprocessing & Preparation](#preprocessing--preparation)
  - [Handling Missing Values & Outliers](#handling-missing-values--outliers)
  - [Feature Selection, Scaling & Encoding](#feature-selection-scaling--encoding)
  - [K-Means Execution & Optimal $K$ Selection](#k-means-execution--optimal-k-selection)
  - [Dimensionality Reduction & Visualization (PCA)](#dimensionality-reduction--visualization-pca)
- [6. Price Modeling & Regression Analysis](#6-price-modeling--regression-analysis)
  - [Subset Selection & Financial Cleaning](#subset-selection--financial-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Data Splitting & ColumnTransformer Pipeline](#data-splitting--columntransformer-pipeline)
  - [Baseline Modeling & Hyperparameter Tuning](#baseline-modeling--hyperparameter-tuning)
  - [Neural Network (MLP) Performance Analysis](#neural-network-mlp-performance-analysis)
  - [Price Band Categorization & Spatial Insights](#price-band-categorization--spatial-insights)
- [7. Text Classification & Persian NLP](#7-text-classification--persian-nlp)
  - [Dataset Preparation & Persian NLP Pipeline (Hazm)](#dataset-preparation--persian-nlp-pipeline-hazm)
  - [Feature Extraction & Vectorization](#feature-extraction--vectorization)
  - [Task A: Property Type Classification (`cat3_slug`)](#task-a-property-type-classification-cat3_slug)
  - [LSTM Architecture & Deep Learning Training Dynamics](#lstm-architecture--deep-learning-training-dynamics)
  - [Task B: Advertiser Type Classification (`user_type`) & Imputation](#task-b-advertiser-type-classification-user_type--imputation)
- [8. Conclusion & Strategic Applications](#8-conclusion--strategic-applications)

---

## 1. Introduction

This report provides a comprehensive data science evaluation of the **Divar Real Estate Listings** dataset (`real_estate_ads`), collected from Divar, Iran's leading classifieds platform. The primary objectives are to evaluate data quality, discover market patterns, build property price estimation models, segment property types via clustering, and leverage Persian Natural Language Processing (NLP) to classify unstructured text descriptions.

The analysis spans the complete data science pipeline:
1. **Exploratory Data Analysis (EDA)** across 60 multi-modal features.
2. **Robust Multi-Stage Cleaning** handling missing values, structural anomalies, and extreme outliers.
3. **Unsupervised Clustering (K-Means & PCA)** to uncover latent market sub-segments.
4. **Supervised Price Estimation (Regression)** comparing linear, tree-based, distance-based, and neural network algorithms.
5. **Persian Text Processing & NLP Classification (Hazm & LSTM)** for property type prediction and large-scale missing value imputation.

---

## 2. Dataset Overview

The dataset consists of **1,000,000+ real estate listings** across Iran, captured in 60 feature columns. It reflects live market behavior, user pricing dynamics, geographic preferences, and transaction terms across Iranian real estate markets.

### Business Feature Categorization

#### 1. Categorization
- `cat2_slug`: Primary listing category (e.g., residential sale, residential rent, commercial sale).
- `cat3_slug`: Specific property type (e.g., apartment, villa, land/plot, shop/office).

#### 2. Location
- `city_slug`, `neighborhood_slug`: City and neighborhood identifiers.
- `location_latitude`, `location_longitude`: Geographic coordinates.
- `location_radius`: Radius of accuracy for location pin.

#### 3. Listing Details
- `created_at_month`: Listing creation timestamp (month level).
- `user_type`: Advertiser category (`personal` / individual vs. `real-estate-agent` / agency).
- `title`, `description`: Free-text Persian title and listing body.

#### 4. Financial Information
- **Rent Details:**
  - `rent_mode`: Rental structure (e.g., monthly).
  - `rent_value`: Monthly rent amount in IRR / Tomans.
  - `rent_to_single`: Boolean flag indicating if rental allows single tenants.
  - `rent_type`: Usage designation (residential vs. commercial/office).
- **Sale Price Details:**
  - `price_mode`: Pricing model (total price vs. price per square meter).
  - `price_value`: Total sale price or unit price.
- **Mortgage / Deposit Details:**
  - `credit_mode`: Deposit structure mode.
  - `credit_value`: Mortgage / security deposit (`rahn`) amount.
- **Transformed Fields:** `transformable_credit`, `transformable_rent`, `transformed_credit`, `transformed_rent`, `rent_credit_transform`.

#### 5. Property Specifications
- **Dimensions:** `building_size` (built area in $m^2$), `land_size` (land plot area in $m^2$).
- **Deed Specifications:** `deed_type` (e.g., single-owner deed, six-barnd), `has_business_deed` (commercial deed flag).
- **Building Structure:** `floor` (unit floor number), `rooms_count` (bedroom count), `total_floors_count` (total building stories), `unit_per_floor` (units per floor).
- **Age & Renovation:** `construction_year` (solar Hijri year of build), `is_rebuilt` (renovation flag).

#### 6. Amenities & Facilities
- **Utilities:** `has_water`, `has_electricity`, `has_gas`.
- **Climate Control:** `has_heating_system`, `has_cooling_system`, `has_warm_water_provider`.
- **Core Facilities:** `has_elevator`, `has_parking`, `has_warehouse`, `has_balcony`.
- **Luxury Features:** `has_pool`, `has_jacuzzi`, `has_sauna`.
- **Building Details:** `building_direction` (north/south facing), `floor_material` (ceramic, parquet, stone), `has_barbecue`, `has_security_guard`, `has_restroom`.

#### 7. Short-Term Rental Information
- `property_type`: Environment classification (coastal, forest, urban).
- `regular_person_capacity`, `extra_person_capacity`: Standard and maximum guest capacities.
- `cost_per_extra_person`: Fee per extra occupant.
- `rent_price_on_regular_days`, `rent_price_on_special_days`, `rent_price_at_weekends`: Tiered daily rental pricing.

---

## 3. Exploratory Data Analysis (EDA)

### Initial Exploration
Initial data inspection revealed structural complexity:
- Mix of Persian text, Finglish, raw digits, mixed character encoding, and unformatted HTML/special symbols in `title` and `description`.
- `NaN` values prevalent across columns, particularly specialized short-term rental and transformable deposit fields.
- Extremely severe right-skewed outliers in numeric variables (e.g., synthetic placeholder values like `999,999,999,999,999`).

### Feature Categorization by Data Type

All 60 columns were classified into 5 operational data type categories:

| Group | Type | Count | Feature Names |
|---|---|---|---|
| **Group 1** | Continuous Numerical | 16 | `rent_value`, `price_value`, `credit_value`, `transformable_credit`, `transformable_rent`, `transformed_credit`, `transformed_rent`, `land_size`, `building_size`, `cost_per_extra_person`, `rent_price_on_regular_days`, `rent_price_on_special_days`, `rent_price_at_weekends`, `location_latitude`, `location_longitude`, `location_radius` |
| **Group 2** | Discrete Numerical | 7 | `regular_person_capacity`, `extra_person_capacity`, `construction_year`, `unit_per_floor`, `total_floors_count`, `rooms_count`, `floor` |
| **Group 3** | Boolean / Binary | 17 | `rent_to_single`, `has_business_deed`, `has_balcony`, `has_elevator`, `has_warehouse`, `has_parking`, `is_rebuilt`, `has_water`, `has_electricity`, `has_gas`, `has_security_guard`, `has_barbecue`, `has_pool`, `has_jacuzzi`, `has_sauna`, `rent_credit_transform`, `transformable_price` |
| **Group 4** | Categorical | 17 | `cat2_slug`, `cat3_slug`, `city_slug`, `neighborhood_slug`, `deed_type`, `building_direction`, `floor_material`, `property_type`, `rent_mode`, `credit_mode`, `price_mode`, `user_type`, `rent_type`, `has_warm_water_provider`, `has_heating_system`, `has_cooling_system`, `has_restroom` |
| **Group 5** | Text / Temporal | 3 | `title`, `description`, `created_at_month` |

### Feature Relationship & Correlation Discovery

Initial correlation matrices on uncleaned raw data produced spurious results due to unmitigated outliers and raw `NaN` handling. After preliminary filtering and log-transformation, clear statistical signals emerged:

#### Monthly Rent (`rent_value`) Key Correlations:
- **Strongest Positive Signals:** `location_latitude` ($r = 0.72$), `price_value` ($r = 0.65$), `credit_value` ($r = 0.65$).
- **Moderate Signals:** `land_size` ($r = 0.54$), `building_size` ($r = 0.50$), `rooms_count` ($r = 0.41$).
- **Weak Signals:** `construction_year` ($r = 0.10$), `floor` ($r = 0.17$), `location_longitude` ($r = 0.02$).

#### Total Sale Price (`price_value`) Key Correlations:
- **Strongest Positive Signals:** `building_size` ($r = 0.65$), `land_size` ($r = 0.65$), `rent_value` ($r = 0.65$), `credit_value` ($r = 0.60$), `location_latitude` ($r = 0.56$), `rooms_count` ($r = 0.55$).

#### Inter-Feature Interactions:
- `building_size` $\leftrightarrow$ `rooms_count`: $r = 0.76$ (Strong structural alignment).
- `land_size` $\leftrightarrow$ `building_size`: $r = 0.65$.
- `total_floors_count` $\leftrightarrow$ `unit_per_floor`: $r = 0.52$.
- `location_longitude`: Displays minimal linear correlation with price ($r \approx 0.02 - 0.10$), whereas `location_latitude` captures the strong North-South price gradient in Tehran.

---

## 4. Data Cleaning & Preparation

### Handling Duplicates

Naively checking exact row duplicates produced false negatives due to minor string discrepancies in text fields and timestamp variants (`created_at_month`).

#### Two-Tier Quality Strategy:
1. **High-Quality Records:** Listings with non-null values across core numerical fields (`price_value`, `building_size`, `rooms_count`).
2. **Low-Quality Records:** Listings missing fundamental financial/physical attributes (`NaN` in primary metrics).

Duplicate detection was restricted to the high-quality subset. Across **981,675 high-quality listings**, exactly **8,790 true duplicate listings** were identified and purged, keeping only the earliest valid posting.

### Text & Data Type Standardization

1. **Character & Digit Normalization:** Persian digits (`۰-۹`) and Arabic numbers (`٠-٩`) were converted to standard ASCII digits. Arabic characters (`ي`, `ك`) were mapped to Persian (`ی`, `ک`).
2. **Numeric Type Enforcement:** Object columns containing numerical strings with commas or Persian text were parsed, cleaned, and cast to `float64` or `int64`.
3. **Categorical Imputation:** Categorical columns with missing entries were assigned the explicit string `"not_specified"` or `"missing"`.
4. **Boolean Uniformity:** Facilities and utility columns stored as mixed object/string types were converted to unified `bool` (`True`/`False`), treating missing values as `False` under the *"absence of claim equals absence of feature"* rule.

---

## 5. Property Listing Clustering

### Preprocessing & Preparation

Clustering was targeted at discovering latent property categories within the Tehran real estate market using mathematical distance metrics ($K$-Means).

#### Missing Value Strategy:
- **High-Missingness Columns (>70%):** Columns like `rent_to_single`, `transformed_rent`, `transformed_credit`, `transformable_rent`, `transformable_credit`, `rent_credit_transform`, `has_sauna`, `has_pool`, `has_jacuzzi` were dropped or converted to binary indicator flags.
- **Land Size (`land_size`):** Inspected in conjunction with `building_size`. Retained primarily for house/villa property categories. Evaluated 27,462 records where $building\_size > land\_size$, resolving data entry swaps. Dropped 12 corrupt rows where both land and building dimensions were zero.
- **Construction Year (`construction_year`):** Missing values imputed using the 25th percentile ($Q_1$) to conservatively reflect older existing building stock.
- **Building Size (`building_size`):** Imputed via hierarchical group-medians conditioned on `rooms_count` and `cat3_slug`.

### Handling Missing Values & Outliers

Extremely large placeholder values were removed prior to model training:
- `price_value = 999,999,999,999,999`
- `building_size = 10,000,000`

To preserve valid luxury properties while removing entries corrupting $K$-Means centroid calculations, logarithmic transformation ($\log(1+x)$) was applied prior to computing Interquartile Range (IQR) fences:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$

IQR bounds were calculated independently within each property type group (`cat3_slug`). Low-variance features such as `location_radius` (ratio of max/min > 500 with zero variance across 99% of sample) were excluded.

### Feature Selection, Scaling & Encoding

- **Final Feature Vector (11 Features):** Physical dimensions (`building_size`, `land_size`, `rooms_count`, `construction_year`), Core amenities (`has_elevator`, `has_parking`, `has_warehouse`), Financial attributes (`price_value`, `rent_value`, `credit_value`), Location (`location_latitude`).
- **Standardization:** All continuous numerical features were scaled using `StandardScaler`:
  $$z = \frac{x - \mu}{\sigma}$$
  Boolean amenities retained $0/1$ representation.

### K-Means Execution & Optimal $K$ Selection

The $K$-Means algorithm was evaluated across $K \in [2, 10]$ using the Elbow Method (Within-Cluster Sum of Squares / Inertia) and Cross-Tabulation matrices.

```
Inertia / WCSS Curve (Elbow Analysis)
  Inertia
    ^
    |  * (K=2)
    |   \
    |    * (K=4)
    |     \
    |      *---* (Elbow at K=6 & K=8)
    |           \
    +-----------------------------------> K Clusters
```

#### Cluster Resolution Analysis:
- **$K=5$ vs $K=6$:** $K=5$ merged rental villas and rental apartments into a single ambiguous cluster. $K=6$ cleanly separated them.
- **$K=7$ vs $K=8$:** $K=8$ isolated commercial real estate and high-value luxury sales from standard residential listings.

#### Final Cluster Profiles ($K=8$):

| Cluster ID | Dominant Market Profile | Key Attributes |
|---|---|---|
| **Cluster 0** | Mid-Range Residential Sales | Medium size ($80-120 m^2$), standard urban price points |
| **Cluster 1** | Standard Residential Rentals | High `rent_value`, moderate deposit, $2$ rooms |
| **Cluster 2** | Small / Economy Residential Units | Compact size ($<60 m^2$), lower price per unit |
| **Cluster 3** | Commercial Properties & Shops | Commercial deed, high location accessibility |
| **Cluster 4** | Large Rental Villas | High `land_size`, private amenities, outer suburbs |
| **Cluster 5** | High-Deposit Rental Apartments | High `credit_value` (mortgage model), low monthly rent |
| **Cluster 6** | Luxury High-Value Sales | Large `building_size` ($>250 m^2$), top tier `price_value` |
| **Cluster 7** | Small Commercial / Office Rentals | Small office spaces, mixed commercial use |

### Dimensionality Reduction & Visualization (PCA)

#### Direct 2D Projection vs. Multi-Stage 8D PCA:
- **Direct 11D $\rightarrow$ 2D PCA:** Retained only **42.73%** of total variance. Visual scatter plots exhibited overlapping clusters, misrepresenting cluster separation quality.
- **PCA to 8 Components:** Retained **90.22%** of cumulative variance.

Re-clustering in the 8-component PCA space confirmed high cluster stability:
- **7 of 8 clusters** exhibited **96.0% – 100.0% sample stability** matching the original 11D space.
- Cluster 3 (Commercial) displayed 15.6% overlap with Cluster 2 (Small Units), reflecting mixed-use urban properties in central Tehran.

---

## 6. Price Modeling & Regression Analysis

### Subset Selection & Financial Cleaning

To model sale prices accurately, the dataset was filtered to **Tehran Residential Sales** (`cat2_slug == 'residential-sell'`), leaving **94,541 candidate listings**.

#### Outlier Truncation & Financial Fences:
- `rent_value`: Excluded $< 500,000$ and $> 200,000,000$ Tomans.
- `credit_value`: Excluded $< 50,000,000$ and $> 50,000,000,000$ Tomans.
- `price_value`: Truncated absolute extreme tail ($> 100,000,000,000$ IRR/Tomans) and applied log-space $1.5 \times \text{IQR}$ fences grouped by `cat3_slug`.
- Truncated top $1\%$ and bottom $1\%$ tails of target price distribution.
- **Final Cleaned Subset:** **84,586 records**.

### Feature Engineering

Three domain-specific engineered features were introduced:

1. **Building Age (`age_of_building`):**
   $$\text{age\_of\_building} = 1403 - \text{construction\_year}$$
   Provides a direct metric for physical depreciation.

2. **Elevator Floor Interaction Effect (`elevator_floor_effect`):**
   High floor units without elevators suffer severe market penalties, whereas elevator access transforms high floors into premium penthouse space:
   $$\text{elevator\_floor\_effect} = \begin{cases} \text{floor} \times 1.5 & \text{if } \text{has\_elevator} = \text{True} \\ \text{floor} \times 0.7 & \text{if } \text{has\_elevator} = \text{False} \end{cases}$$

3. **Neighborhood Price Proxy (`neighborhood_avg_price`):**
   Calculated as the target-encoded median sale price per `neighborhood_slug` derived exclusively from the training fold to prevent data leakage.

4. **Logarithmic Target Transformation:**
   To equalize error variance across price tiers:
   $$y = \log(1 + \text{price\_value})$$

### Data Splitting & ColumnTransformer Pipeline

- **Data Split:** $80\%$ Training / $20\%$ Testing. The Training set was further split $80/20$ into Train ($64\%$) and Validation ($16\%$).
- **Preprocessing Pipeline (`ColumnTransformer`):**
  - Continuous Numeric: `StandardScaler`.
  - Categorical Features: `OneHotEncoder(handle_unknown='ignore')`.

### Baseline Modeling & Hyperparameter Tuning

Four regression architectures were evaluated on the validation dataset:

| Model Architecture | Base $R^2$ (Val) | Tuned $R^2$ (Val) | Best Hyperparameters / Notes |
|---|---|---|---|
| **Linear Regression** | $0.770$ | $0.770$ | Standard OLS baseline; fast, linear coefficients |
| **KNN Regressor** ($k=5$) | **$0.830$** | **$0.835$** | Weighted by distance (`weights='distance'`), $k=7$ |
| **Decision Tree Regressor** | $0.760$ | $0.770$ | Tuned via `RandomizedSearchCV`: `max_depth=10`, `min_samples_split=20` |
| **MLP Regressor** (Neural Net) | $< 0.000$ (Negative) | $< 0.000$ (Negative) | Architectures: $(64,)$, $(64, 32)$, $\text{max\_iter}=200$, $\text{ReLU}$ |

`RandomizedSearchCV` was executed over a statistically representative 2,000-sample fold to optimize tree depth and neural network hyperparameters under memory constraints.

### Neural Network (MLP) Performance Analysis

The Multi-Layer Perceptron (`MLPRegressor`) persistently failed to achieve positive $R^2$ scores across various hidden layer configurations, activation functions ($\text{ReLU}, \text{tanh}$), and regularization terms ($\alpha$).

#### Diagnostic Causes:
1. **High-Dimensional Feature Sparsity:** One-Hot Encoding generated sparse categorical matrices, causing backpropagation gradient dissipation across unpopulated input dimensions.
2. **Extreme Collinearity:** Physical features (`building_size`, `rooms_count`) and neighborhood target encoding caused ill-conditioned weight matrices during gradient descent.
3. **Sample Size Constraints during Search:** Micro-batch gradient optimization on non-convex real estate price manifolds required deeper hyperparameter grid search beyond the 2,000-sample tuning subset. Distance-based ($K$-NN) and tree ensembles proved vastly superior for structured tabular listings.

### Price Band Categorization & Spatial Insights

Using the top-performing regression model ($K$-NN / Tuned Decision Tree), the ratio of actual listing price to model-predicted price was computed for each listing:
$$\text{Price Ratio} = \frac{\text{Actual Listed Price}}{\text{Model Predicted Price}}$$

Listings were categorized into three operational valuation bands:
- **Under-Valued (Bargain Files):** $\text{Price Ratio} < 0.8$ (Listed $>20\%$ below model estimate).
- **Fairly Valued (Normal Market):** $0.8 \le \text{Price Ratio} \le 1.2$.
- **Over-Valued (Inflated Listings):** $\text{Price Ratio} > 1.2$ (Listed $>20\%$ above model estimate).

```
Price Valuation Distribution (Test Set)
  Listings Count
    ^
 9k |        +-----------------+
    |        |                 |
    |        |  Fairly Valued  |
 2k |  +-----+  (Normal ~9k)   +-----+
 1.8k|  |Under|                 |Over |
    |  |(~2k)|                 |(1.8k|
    +--+-----+-----------------+-----+------> Valuation Category
```

#### Test Set Valuation Breakdown:

| Valuation Category | Listing Count | Avg. Built Size ($m^2$) | Avg. Actual Price (Billion IRR/Tomans) | Top Represented Neighborhoods |
|---|---|---|---|---|
| **Under-Valued** | ~2,000 | $94 m^2$ | $4.25$ | Suburbs, Southern districts, Motivated sellers |
| **Fairly Valued** | ~9,000 | $93 m^2$ | $8.99$ | Poonak, Jeyhoun, Sazman-e-Barnameh |
| **Over-Valued** | ~1,800 | $98 m^2$ | $10.00$ | Poonak, Chitgar Lake, Salsabil |

#### Key Spatial Findings:
- Properties across all three valuation bands exhibited nearly identical average built areas ($93 - 98 m^2$).
- Over-valued properties were listed at an average of **10.0 Billion Tomans** vs **4.25 Billion Tomans** for under-valued listings of identical footprint.
- High seller markup concentrated in speculative high-demand growth zones (e.g., Chitgar Lake area).

---

## 7. Text Classification & Persian NLP

### Dataset Preparation & Persian NLP Pipeline (Hazm)

Unstructured textual descriptions (`title` and `description`) contain high-density semantic signals regarding property specifics, seller urgency, and deal structures.

#### Persian Cleaning & Pipeline (`Hazm` Library Integration):
1. **String Concatenation:** `title` and `description` merged into unified document column `not_cleaned`.
2. **Noise Reduction:** Custom regular expressions purged emojis, HTML fragments, non-Persian character sets, English alphanumeric codes, punctuation, and isolated digits.
3. **Persian Text Normalization (`Hazm Normalizer`):** Mapped Arabic characters (`ي`, `ك`) to Persian (`ی`, `ک`), standardizing half-spaces (`نیم‌فاصله`).
4. **Tokenization (`Hazm WordTokenizer`):** Parsed text streams into token sets.
5. **Stemming (`Hazm Stemmer`):** Reduced inflected variants to root words (e.g., `طبقات` $\rightarrow$ `طبقه`, `آپارتمان‌ها` $\rightarrow$ `آپارتمان`).
6. **Stopword Removal:** Filtered high-frequency, low-information Persian grammatical terms using a curated stopword set.

```
Raw Text ---> Regex Clean ---> Hazm Normalize ---> Tokenize ---> Stem ---> Stopword Filter ---> Final Vector
```

**Final NLP Dataset Size:** **999,945 clean Persian text documents**.

### Feature Extraction & Vectorization

Three text representation approaches were constructed:
1. **Bag of Words (BoW):** `CountVectorizer` constrained to top $10,000$ n-gram features.
2. **TF-IDF Vectorization:** `TfidfVectorizer` constrained to top $10,000$ features with inverse document frequency weighting:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$$
3. **Word2Vec Embeddings:** Trained Skip-Gram model ($\text{vector\_size}=100$, $\text{min\_count}=2$, $\text{window}=5$). Document vectors computed via mean token embedding aggregation.
4. **Dense Sequential Identifiers:** Keras `Tokenizer` mapping tokens to integer index sequences, padded to uniform length $N=100$ for deep neural architectures.

---

### Task A: Property Type Classification (`cat3_slug`)

Classifying listings into 16 fine-grained property types based exclusively on textual descriptions ($40\%$ random sample $= 400,000$ listings; $80/20$ Train/Test split).

#### Classical Model Performance Comparison:

| Classifier Architecture | Vectorizer Feature | Accuracy | Weighted $F_1$-Score |
|---|---|---|---|
| **Logistic Regression** | BoW | $0.825$ | $0.823$ |
| **Logistic Regression** | **TF-IDF** | **$0.829$** | **$0.826$** |
| Naïve Bayes (Multinomial) | BoW | $0.749$ | $0.755$ |
| Naïve Bayes (Multinomial) | TF-IDF | $0.729$ | $0.715$ |
| Random Forest (Base) | BoW / TF-IDF | $0.495$ | $0.413$ |
| Random Forest (Tuned + Class Weights) | BoW / TF-IDF | $0.698$ | $0.716$ |
| Word2Vec + Classifier | Mean Embedding | $0.778$ | $0.774$ |

**Classical Baseline Winner:** **Logistic Regression + TF-IDF** achieved **$82.9\%$ Accuracy** and **$0.826$ Weighted $F_1$-Score**.

---

### LSTM Architecture & Deep Learning Training Dynamics

To capture word order dependencies and semantic context, a Deep Recurrent Neural Network (LSTM) was constructed:

```
Input Tokens (Length=100)
       |
[Embedding Layer] (Input Dim=10,000, Output Dim=128)
       |
[LSTM Layer] (128 Units, Dropout=0.2, Recurrent Dropout=0.2)
       |
[Dense Layer] (64 Units, ReLU)
       |
[Output Softmax] (16 Classes)
```

- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam` ($\text{lr} = 0.001$)
- **Batch Size:** $32$ | **Epochs:** $10$

```
LSTM Learning Curves across Epochs
  Accuracy / Loss
    1.0 +----------------------------------------------+ (Val Acc ~0.84)
        |  *---*---*---*---*---*---*---*---*---* (Train Acc -> 0.89)
    0.5 |  #---#---# (Val Loss Min ~0.44 at Epoch 3)
        |       \   /---/---/---/---/ (Val Loss Increases -> 0.54)
    0.0 +----------------------------------------------+
        1   2   3   4   5   6   7   8   9   10  Epochs
```

#### Epoch Training Dynamics:
- **Epoch 1:** Training Acc = $0.73$, Val Acc = $0.83$, Val Loss = $0.80$.
- **Epoch 2–3:** Val Acc reached **$0.84$ ($84.0\%$)**, Val Loss minimized at **$0.44$**.
- **Epoch 4–10:** Training Acc climbed to $0.89$ (Train Loss dropped to $0.28$), but Val Loss expanded to $0.54$, signaling onset of mild memorization / overfitting.
- **Optimal Early Stopping Point:** Epoch $3 - 4$.

#### Performance Metric:
The LSTM achieved **$84.0\%$ multi-class accuracy** across 16 categories, outperforming the baseline random guess benchmark of $6.25\%$ ($1/16$) and surpassing classical TF-IDF Logistic Regression.

---

### Task B: Advertiser Type Classification (`user_type`) & Imputation

#### Problem Formulation:
Out of 999,945 total listings, only **288,877 listings** contained valid `user_type` labels, while **$>700,000$ listings were unlabelled (`NaN`)**. A supervised model was trained on the labelled subset to impute the unlabelled listings.

#### Class Imbalance & Weighting:
The labelled subset exhibited heavy imbalance:
- **Real Estate Agencies (`real-estate-agent`):** $102,043$ listings ($88.4\%$, Majority Class).
- **Personal / Individual Sellers (`personal`):** $13,348$ listings ($11.6\%$, Minority Class).
- **Imbalance Ratio:** $\approx 7.64 : 1$.

To prevent the model from defaulting to predicting the majority class, explicit loss penalty weights were computed via `compute_class_weight`:
$$W_{\text{majority}} = 0.57 \quad \mid \quad W_{\text{minority}} = 4.32$$

#### Binary LSTM Architecture:
- Structure: `Embedding(10,000, 128)` $\rightarrow$ `LSTM(64)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(1, Sigmoid)`.
- Loss: `binary_crossentropy` with `class_weight` penalties applied during `fit()`.

#### Model Evaluation on Unseen Test Fold:

| Target Class | Class Weight | Precision | Recall | $F_1$-Score | Sample Count |
|---|---|---|---|---|---|
| **Personal (`personal`)** | $4.32$ | $0.53$ | **$0.81$** | $0.64$ | ~2,670 |
| **Agency (`real-estate-agent`)** | $0.57$ | $0.97$ | $0.91$ | $0.94$ | ~20,400 |
| **Macro Average** | — | $0.75$ | $0.86$ | $0.79$ | ~23,070 |
| **Weighted Average** | — | **$0.92$** | **$0.89$** | **$0.90$** | ~23,070 |

#### Imputation Outcome:
Applying the trained binary LSTM classifier to the $>700,000$ unlabelled entries successfully generated high-confidence predictions, stored in a new field `predicted_user_type`. This completed the dataset without discarding valuable unlabelled samples.

---

## 8. Conclusion & Strategic Applications

### Key Technical Achievements
1. **Data Pipeline Robustness:** Successfully cleaned and standardized a multi-modal 1M+ listing Persian dataset, converting missing entries into meaningful signals.
2. **Market Segmentation:** Discovered 8 stable, interpretable property clusters using $K$-Means and 8-component PCA ($90.22\%$ variance retained, $>96\%$ cluster stability).
3. **Automated Valuation:** Built distance and tree-based price regression models achieving $R^2 = 0.835$, successfully categorizing listings into Over-valued, Fairly-valued, and Under-valued bands.
4. **Persian NLP & Deep Learning:** Deployed Hazm NLP normalization with an LSTM neural network, achieving $84.0\%$ accuracy on 16-class property classification and $0.90$ Weighted $F_1$ on advertiser classification, enabling mass missing-value imputation for $>700,000$ listings.

### Industrial & Business Applications

```
+-----------------------------------------------------------------------------------+
|                           BUSINESS IMPACT & USE CASES                             |
+-----------------------------------+-----------------------------------------------+
| Automated Valuation Models (AVM)  | Real-time estimated property pricing for      |
|                                   | buyers, sellers, and mortgage lenders.        |
+-----------------------------------+-----------------------------------------------+
| Pricing Anomaly & Fraud Detection | Automated alerts for severely over-priced or  |
|                                   | suspicious under-priced listings.             |
+-----------------------------------+-----------------------------------------------+
| User Intelligence & Ad Filtering  | Inferring advertiser persona (Agency vs Owner)|
|                                   | to enhance search ranking algorithms.         |
+-----------------------------------+-----------------------------------------------+
```

---
*Report finalized and verified for technical accuracy.*
