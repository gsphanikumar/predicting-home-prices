### Predicting Home Sale Prices

**Phani Kumar G S**

#### Executive summary

This project builds and evaluates machine learning models to predict residential home sale prices using the Ames Housing Dataset. Through exploratory data analysis, feature engineering, and a progression of modeling approaches, from linear regression to tree-based and ensemble methods, the project identifies the key structural and locational drivers of home value and determines which modeling approach delivers the best predictive performance.

#### Rationale
Home price prediction has direct practical value for buyers, sellers, real estate agents, and lenders. Understanding which features drive price enables better investment decisions, more accurate appraisals, and fairer lending. Beyond real estate, this problem is a canonical end-to-end supervised learning exercise that tests the full data science pipeline: cleaning messy real-world data, engineering meaningful features, and rigorously comparing models.

#### Research Question
Can a machine learning model accurately predict residential home sale prices and classify properties into meaningful market segments based on structural and locational features, and which modeling approach delivers the best predictive performance?

#### Data Sources
**Ames Housing Dataset** — fetched via scikit-learn's `fetch_openml`.

- 1,460 residential property sales in Ames, Iowa
- ~80 features per property (lot size, neighborhood, year built, overall condition, square footage, garage, basement, and more)
- Target variable: `SalePrice` (continuous, in USD)

The dataset contains realistic data quality challenges — 19 columns with missing values, a mix of numerical and categorical variables, and significant multicollinearity — making it well-suited for an end-to-end machine learning exercise.

#### Methodology
The project works through five stages:

1. **Data Loading** — fetch from OpenML, inspect shape, types, and summary statistics
2. **Missing Value Analysis** — identify 19 columns with missing data; distinguish "feature absent" from genuine gaps
3. **EDA** — correlation heatmap, scatter plots of top numerical predictors, box plots by OverallQual and Neighborhood, median price trend by year built
4. **Data Cleaning and Feature Engineering** — targeted imputation, outlier removal, and 8 engineered features (HouseAge, TotalSF, TotalBathrooms, YrsSinceRemodel, HasPool, HasGarage, HasBasement, HasFireplace)
5. **Modeling** — multiple linear regression with one-hot encoding and log-transformed target as a baseline; regularized regression (Lasso, Ridge, Elastic Net); tree-based models (Decision Tree, XGBoost); and K-Means clustering for market segmentation

#### Results
**Data Quality:** The dataset has 19 columns with missing values. Most (PoolQC, Alley, Fence, GarageType, BsmtQual, etc.) indicate the property does not have that feature. After imputation and removal of 2 outliers, the dataset is fully clean.

**Target Variable:** SalePrice is right-skewed (skewness ~1.88), ranging from $34,900 to $755,000 with a median near $163,000. A log transformation normalizes the distribution (skewness drops to ~0.12) and is used as the target in all models.

**Key Price Drivers:**

| Feature | Correlation with SalePrice |
|---------|---------------------------|
| OverallQual | ~0.79 |
| TotalSF (engineered) | ~0.78 |
| GrLivArea | ~0.71 |
| GarageCars | ~0.64 |
| GarageArea | ~0.62 |
| TotalBsmtSF | ~0.61 |

Neighborhood is the most impactful categorical variable — top neighborhoods (NridgHt, NoRidge, StoneBr) have median prices 2–3x those of lower-tier neighborhoods. Newer homes (built after 1980) command significantly higher prices on average.

**Baseline Model — Multiple Linear Regression:**

| Metric | Train | Test |
|--------|-------|------|
| RMSE (log scale) | 0.0863 | 0.1372 |
| R² Score | 0.9527 | 0.8884 |

The model explains ~86% of price variance on unseen data. The gap between train and test R² indicates mild overfitting — expected for plain linear regression on a high-dimensional one-hot encoded feature matrix.

#### Next steps
- Apply regularized regression: Lasso, Ridge, Elastic Net with cross-validated hyperparameter tuning
- Visualize gradient descent convergence
- Train and evaluate tree-based models: Decision Tree, XGBoost
- Explore a simple neural network
- Segment the market using PCA + K-Means clustering
- Build a KNN classifier for segment prediction
- Conduct a final model comparison across all approaches

#### Outline of project

- [EDA and Baseline Model](ames_housing_eda.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gsphanikumar/predicting-home-prices/blob/main/ames_housing_eda.ipynb)

