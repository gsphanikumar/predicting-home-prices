# Predicting Home Sale Prices

**Phani Kumar G S**

## 1. Problem Statement

When a home goes on the market, how do you know if the asking price is fair? Buyers, sellers, real estate agents, and lenders all need an objective answer to that question — but arriving at one manually requires expertise, time, and local market knowledge that most people don't have.

This project builds a computer model that can look at the measurable characteristics of a home (its size, age, quality, neighborhood, garage, bathrooms, and dozens of other details) and produce an accurate estimate of what that home should sell for. The model learns by studying nearly 1,500 past home sales in Ames, Iowa, where the actual sale prices are already known.

**Goals:**
- Predict the sale price of a residential home as accurately as possible
- Identify which features of a home matter most to its price
- Group homes into natural market segments (Budget, Mid-Range, Premium, Luxury) to enable peer-group comparisons

**Challenges:**
- The raw data has 19 columns with missing information that must be handled carefully
- Sale prices range from $35,000 to $755,000 — a very wide spread — with a small number of very expensive homes pulling the average upward
- There are roughly 80 different home characteristics, many of which are correlated with each other, making it hard for simple models to separate their individual effects

**Potential benefits:**
- Buyers can verify whether a home is fairly priced before making an offer
- Sellers and agents can set realistic listing prices
- Lenders can make faster, more consistent appraisals
- The same approach generalizes to any housing market with similar data

## 2. What the Model Predicts

This project uses **supervised learning** — a type of machine learning where the model is trained on historical examples that include both the inputs (home features) and the correct answer (the actual sale price). The model learns the relationship between the two and can then estimate prices for homes it has never seen before.

The goal is to predict a specific number (sale price in dollars), not just a category like "cheap" or "expensive."

In addition to price prediction, the model identifies natural groupings of homes without being told in advance what those groups should be.

**Expected outputs:**
- A dollar-value sale price estimate for any home, given its characteristics
- A market segment label (Budget/Starter, Mid-Range, Premium, or Luxury) for that home

## 3. Data Acquisition

**Source:** The Ames Housing Dataset, fetched directly from OpenML (a public machine learning data repository) via the scikit-learn Python library.

**What's in the data:**
- **1,460 residential property sales** in Ames, Iowa
- **80 features per property**, covering lot size, neighborhood, year built, overall condition, above-ground living area, basement area, garage size, number of bathrooms, and many more
- **Target variable:** `SalePrice` — the actual price the home sold for, in US dollars

**Why this dataset:**
The Ames dataset is a realistic snapshot of a real housing market. It contains the kinds of data quality problems found in practice — missing values, a mix of numbers and text fields, and strong correlations between features. This makes it a rigorous test of an end-to-end data science pipeline, not just a toy example.

**Key data characteristics discovered during exploration:**

| Stat | Value |
|------|-------|
| Number of homes | 1,458 (after removing 2 extreme outliers) |
| Price range | $34,900 — $755,000 |
| Median sale price | ~$163,000 |
| Average sale price | ~$181,000 |
| Features with missing data | 19 columns |

**What the data tells us visually:**

- Sale prices are right-skewed: most homes cluster between $100K–$250K, but a small number sell for $400K–$755K, pulling the average higher than the median.
- The single strongest numerical predictor is **Overall Quality rating** (a 1–10 scale): homes rated 9–10 typically sell for 2–3× the price of homes rated 4–5.
- **Neighborhood** has an enormous effect: the most expensive neighborhoods (NridgHt, NoRidge, StoneBr) have median prices 2–3× those of the least expensive neighborhoods.
- **Total square footage** has a near-linear relationship with price — more space consistently means higher prices.
- **Newer homes** (built after 1980) command noticeably higher prices on average.

## 4. Data Preparation

Raw data is rarely ready to use directly. Before training any model, the data was cleaned and transformed in three stages.

### Handling Missing Values

Of the 19 columns with missing data, most were missing for a logical reason — the property simply did not have that feature. For example:
- A missing `PoolQC` (pool quality) value just means the home has no pool
- A missing `GarageType` means there is no garage

For these columns, missing values were filled with `"None"` (for text fields) or `0` (for numbers). For the one genuinely missing numerical column (`LotFrontage`, the street width), the missing values were filled with the median value across all homes.

Two homes with unusually large floor areas but very low sale prices (likely data entry errors) were removed, as they would distort the model's learning.

**Result:** Zero missing values, 1,458 clean records.

### Feature Engineering

Eight new features were created by combining existing ones, because combined measures often predict price better than individual columns:

| New Feature | How It's Calculated | Why It Helps |
|---|---|---|
| TotalSF | Basement + 1st floor + 2nd floor area | Single measure of overall size |
| TotalBathrooms | Full + half bathrooms (all floors) | Captures total bathroom count |
| HouseAge | Year sold minus year built | Newer homes are worth more |
| YrsSinceRemodel | Year sold minus last remodel year | Recent updates boost value |
| HasPool | 1 if pool exists, 0 if not | Binary flag for pool presence |
| HasGarage | 1 if garage exists, 0 if not | Binary flag for garage presence |
| HasBasement | 1 if basement exists, 0 if not | Binary flag for basement |
| HasFireplace | 1 if fireplace exists, 0 if not | Binary flag for fireplace |

`TotalSF` turned out to be the single strongest predictor in the entire dataset (correlation of 0.83 with sale price).

### Preparing for Modeling

- **Target transformation:** Sale prices were log-transformed (a mathematical adjustment) to correct for the right-skewed distribution. This makes the modeling math more reliable. All reported dollar errors are converted back to regular dollars for interpretation.
- **Encoding text fields:** Categorical columns (like Neighborhood or Exterior Material) were converted into numerical format using one-hot encoding, expanding the dataset from 80 features to 267.
- **Feature selection:** Features with very weak correlation to price (|r| < 0.05) were dropped, reducing the set to 182 informative features.
- **Scaling:** Features were normalized so that no single variable dominates simply by having large numerical values (e.g., square footage in the thousands vs. a binary 0/1 flag).
- **Train/test split:** The data was randomly divided — 80% (1,166 homes) for training the model, and 20% (292 homes) held back as an unseen test set to evaluate real-world performance.

## 5. Modeling

Seven different modeling approaches were tested, progressing from simple to complex:

**1. Multiple Linear Regression (Baseline)**
The simplest possible approach: assume that each feature has a fixed, additive effect on price. Easy to understand and explain. Serves as the benchmark every other model must beat.

**2. Ridge Regression**
A variation of linear regression that adds a mathematical penalty for large coefficients, preventing any single feature from having an outsized influence. Keeps all features but shrinks the ones that matter less.

**3. Lasso Regression**
Similar to Ridge, but this penalty actually zeros out the least useful features entirely — performing automatic feature selection. Produced a model using only 104 of the 182 features, with nearly the same accuracy as more complex approaches.

**4. Elastic Net**
A blend of Ridge and Lasso penalties, combining the strengths of both.

**5. Decision Tree**
Splits the data into progressively smaller groups based on yes/no questions about home features (e.g., "Is TotalSF > 2,500?"), arriving at price estimates for each leaf group. Intuitive and fast, but prone to memorizing the training data (overfitting). Best tree depth of 6 was selected using cross-validation.

**6. Gradient Boosting (equivalent to XGBoost)**
Builds hundreds of small decision trees in sequence, each one correcting the errors of the previous ones. This ensemble approach consistently delivers the highest accuracy on structured tabular data and is the industry standard for housing valuation tasks. Used 400 trees with depth 4, learning rate 0.05.

**7. Neural Network (Multi-Layer Perceptron)**
Loosely inspired by the human brain, a neural network learns non-linear patterns through multiple layers of weighted connections. Architecture: 256 → 128 → 64 neurons. Despite extensive tuning, neural networks tend to underperform on datasets this small (under 5,000 rows); gradient boosting remains superior in this range.

**Market Segmentation (K-Means Clustering)**
Separately from price prediction, K-Means clustering was used to group homes into 4 natural market segments based on their structural characteristics, using PCA (a dimensionality-reduction technique) to simplify the feature space first. A K-Nearest Neighbors classifier was then trained to assign new listings to a segment.

## 6. Model Evaluation

### How Performance Is Measured

Two metrics are used to evaluate each model:

- **R² (R-squared):** The percentage of price variation explained by the model. An R² of 1.0 would be a perfect predictor; 0.0 means the model is no better than always predicting the average price.
- **RMSE (Root Mean Squared Error):** The typical size of a prediction error, in log-price units. This is converted to dollars for business interpretation.

### Results

| Model | Test R² | Test RMSE ($) | Median Error % |
|---|---|---|---|
| Baseline OLS (Linear Regression) | 0.888 | $21,878 | 6.4% |
| Ridge Regression | 0.887 | $23,060 | 6.5% |
| **Lasso Regression** | **0.898** | **$21,543** | **6.1%** |
| Elastic Net | 0.897 | $21,721 | 6.0% |
| Decision Tree | 0.759 | $39,798 | 9.1% |
| **Gradient Boosting** | **0.898** | **$20,585** | **6.3%** |
| Neural Network (MLP) | — | $189,598 | 29.7% |

### Which Model Won — and Why

**Gradient Boosting** achieved the best overall accuracy, with a test R² of 0.898 and a median prediction error of about 6.3%. On a typical $163,000 home, this translates to a median prediction error of roughly **$10,000** — within the normal range of price negotiation, making it commercially viable as a pricing guide.

**Lasso Regression** is essentially tied with Gradient Boosting in accuracy (confirmed by 5-fold cross-validation, which showed their performance ranges overlap). Lasso's advantage is interpretability: it produces a transparent list of coefficients that can be audited, which matters in regulatory and appraisal contexts.

**The Neural Network failed** on this dataset — its errors were nearly 10× larger than the best models. This is a well-known limitation: neural networks need far more data (typically 50,000+ examples) to outperform gradient boosting on structured tabular problems.

### Recommended Model for Each Use Case

| Use Case | Best Model | Reason |
|---|---|---|
| Automated pricing tool (AVM) | Gradient Boosting | Highest accuracy, handles complex non-linear patterns |
| Regulatory / lender appraisal | Lasso | Fully auditable, nearly identical accuracy |
| Quick prototype / internal tool | Lasso | Fast training, no sensitivity to scaling choices |
| Market segment assignment | KNN on PCA | Assigns new listings to a tier without retraining |

### What Drives Home Prices Most

The top three value drivers, consistent across all models:

1. **Total Square Footage** (all floors + basement combined) — the single most predictive feature
2. **Overall Quality Rating** — a 1–10 subjective quality score assigned by the appraiser
3. **Neighborhood** — location creates the largest categorical price spread; top neighborhoods (NridgHt, NoRidge, StoneBr) have median prices 2–3× those of lower-tier areas

### Market Segments Identified

K-Means clustering identified 4 distinct property tiers:

| Segment | Description | Typical Price Range |
|---|---|---|
| Budget / Starter | Older, smaller homes with basic finishes | Below ~$130K |
| Mid-Range | Average condition, typical Ames suburban home | ~$130K–$185K |
| Premium | Newer construction, larger lots, better quality | ~$185K–$260K |
| Luxury | High-end finishes, top neighborhoods, large footprint | Above ~$260K |

The model is most accurate within the Budget/Starter and Mid-Range tiers, where homes are more similar to each other. Luxury homes vary more in ways that are harder to capture from standard listing data alone.

## Project Notebooks

- [EDA and Baseline Model](ames_housing_eda.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gsphanikumar/predicting-home-prices/blob/main/ames_housing_eda.ipynb)
- [Full Analysis, Model Comparison & Recommendations](ames_housing_full_analysis.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gsphanikumar/predicting-home-prices/blob/main/ames_housing_full_analysis.ipynb)
