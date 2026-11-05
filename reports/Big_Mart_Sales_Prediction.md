# Big Mart Sales Prediction

## 1. Project Information

Project Name: Big Mart Sales Prediction

Project Type: Supervised Learning / Regression

Status: Development

Date Started: 2026-10-25

Last Updated: 2026-10-25

Author: Samuel Ablordeppey

### Project Description

An end-to-end machine learning project for predicting product sales across Big Mart stores using regression models, Python, scikit-learn, and the Big Mart Sales dataset.

A machine learning regression project that predicts sales for individual products at different stores using product attributes and store characteristics. Includes data cleaning, exploratory data analysis, feature engineering, regression model comparison, hyperparameter tuning, and model evaluation.

### Objective

Build and evaluate regression models capable of predicting product sales using product-level attributes, product visibility, product type, pricing information, and store characteristics.

### Expected Outcome

A trained and evaluated regression model that accepts product and store characteristics and produces an estimated sales value for a product at a particular store.

## 2. Problem Definition

### Real-World Problem

Retail stores need to understand and estimate product sales to support inventory management, demand planning, product placement, and business decision-making. Sales may vary according to product characteristics, prices, visibility, store type, store location, and other factors.

### Machine Learning Problem

Use historical product and store data to learn the relationship between product attributes, store characteristics, and recorded sales, then use the learned relationship to predict sales for unseen observations.

### Problem Type

- Supervised Learning
- Regression

### Target Variable

Name: `Item_Outlet_Sales`

Type: Continuous numerical

Description: Sales of a particular product at a particular retail outlet, as recorded in the dataset.

Units: Monetary units as represented in the dataset. The dataset documentation should be consulted to confirm the currency and interpretation of the sales values.

### Features

The input features are listed in the Data Dictionary section below.

## 3. Requirements & Constraints

### Functional Requirements

1. Load the Big Mart Sales dataset.
2. Inspect the dataset structure, data types, and data quality.
3. Clean and preprocess the data.
4. Perform exploratory data analysis (EDA).
5. Handle missing values and investigate potential data-quality issues.
6. Engineer useful features where appropriate.
7. Encode categorical variables and scale numerical variables where necessary.
8. Split the dataset into training and testing sets.
9. Train regression models.
10. Evaluate model performance using appropriate regression metrics.
11. Compare regression algorithms.
12. Tune promising models using cross-validation and hyperparameter optimization.
13. Interpret feature importance and model predictions.
14. Save the selected trained model and preprocessing pipeline.
15. Generate predictions for new observations.

### Technical Requirements

1. Python
2. Pandas
3. NumPy
4. Scikit-learn
5. Matplotlib
6. Seaborn
7. Joblib
8. XGBoost, LightGBM, or CatBoost, where appropriate

### Constraints

- The dataset contains approximately 8,523 observations and 12 columns in the commonly distributed training dataset, including the target variable.
- Missing values may occur in features such as `Item_Weight` and `Outlet_Size` and must be investigated and handled appropriately.
- Numerical and categorical variables require appropriate preprocessing.
- Categorical features include `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`.
- The target variable, `Item_Outlet_Sales`, may be right-skewed and contain high-sales observations.
- Some categorical values may represent the same category under different spellings or abbreviations, requiring consistent standardization.
- `Item_Visibility` may contain zero values that require investigation to determine whether they represent genuine observations or data-quality issues.
- Relationships between product attributes, store characteristics, and sales may be nonlinear.
- Model performance depends on data quality, feature relevance, and the evaluation strategy.
- The dataset represents a particular retail dataset and may not generalize to every retailer, market, or time period.
- The dataset does not necessarily provide sufficient temporal information to support genuine time-series forecasting.
- Data leakage must be prevented during preprocessing, feature engineering, feature selection, hyperparameter tuning, and model evaluation.
- Any target transformation, such as a logarithmic transformation of `Item_Outlet_Sales`, must be handled correctly when converting predictions back to the original scale.

## 4. Dataset

### Dataset Information

Dataset Name: BigMart Sales Data

Project Dataset Name: `big_mart_sales.csv`

Source: Kaggle

Dataset URL: [https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

Version: Dataset version available from the linked source at download time

Number of Rows: Approximately 8,523 in the commonly distributed training dataset

Number of Columns: 12, including the target variable

File Format: CSV

### Dataset Files

```
data/
 ├── raw/
 │    └── big_mart_sales.csv
 ├── external/
 └── processed/
```

- `raw/`: Original downloaded dataset, preserved without modifications.
- `external/`: Additional datasets, if required for supplementary analysis.
- `processed/`: Cleaned or transformed datasets generated during preprocessing.

### Data Dictionary

| #  | Feature                     | Type                     | Description                                             | Unit / Values                    | Role                 |
| -- | --------------------------- | ------------------------ | ------------------------------------------------------- | -------------------------------- | -------------------- |
| 1  | `Item_Identifier`           | Categorical (Identifier) | Unique identifier for each product                      | Product codes                    | Feature / Identifier |
| 2  | `Item_Weight`               | Numerical                | Weight of the product                                   | Dataset units                    | Feature              |
| 3  | `Item_Fat_Content`          | Categorical              | Fat-content classification of the product               | Low Fat, Regular, and variants   | Feature              |
| 4  | `Item_Visibility`           | Numerical                | Recorded visibility measure of the product in the store | Dataset-defined measure          | Feature              |
| 5  | `Item_Type`                 | Categorical              | Category to which the product belongs                   | Product categories               | Feature              |
| 6  | `Item_MRP`                  | Numerical                | Maximum Retail Price of the product                     | Monetary units                   | Feature              |
| 7  | `Outlet_Identifier`         | Categorical (Identifier) | Unique identifier for the retail outlet                 | Outlet codes                     | Feature              |
| 8  | `Outlet_Establishment_Year` | Numerical (Integer)      | Year the retail outlet was established                  | Year                             | Feature              |
| 9  | `Outlet_Size`               | Categorical              | Size classification of the retail outlet                | Small, Medium, High              | Feature              |
| 10 | `Outlet_Location_Type`      | Categorical              | Tier classification of the outlet's location            | Tier 1, Tier 2, Tier 3           | Feature              |
| 11 | `Outlet_Type`               | Categorical              | Type of retail outlet                                   | Grocery Store, Supermarket types | Feature              |
| 12 | `Item_Outlet_Sales`         | Numerical (Continuous)   | Sales of a product at a particular outlet               | Monetary units                   | Target               |

### Feature and Target Summary

Input Features (X):

```
["Item_Identifier",    "Item_Weight",    "Item_Fat_Content",    "Item_Visibility",    "Item_Type",    "Item_MRP",    "Outlet_Identifier",    "Outlet_Establishment_Year",    "Outlet_Size",    "Outlet_Location_Type",    "Outlet_Type"]
```

Target Variable (y):

```
"Item_Outlet_Sales"
```

### Initial Modeling Considerations

- **Numerical features:** `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Outlet_Establishment_Year`
- **Categorical features**: `Item_Identifier`, `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
- **Target:** `Item_Outlet_Sales`
- **Feature engineering:** Created `item_visibility_ratio` to measure product visibility relative to the average visibility within each outlet..
- **Categorical encoding:** One-hot encoding is a suitable starting point for nominal categorical features. Consider how to handle high-cardinality identifiers such as `Item_Identifier` and `Outlet_Identifier`.
- **Feature scaling:** Standardization is useful for scale-sensitive models, such as Linear Regression with regularization, Ridge, Lasso, Elastic Net, and Support Vector Regression.
- **Initial regression model:** Linear Regression.
- **Candidate models:** Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost, LightGBM, CatBoost, and Support Vector Regression.
- **Evaluation metrics:** MAE, RMSE, and R².
- **Validation strategy:** Train-test split and cross-validation, with feature engineering and preprocessing fitted within a scikit-learn pipeline to prevent data leakage.
- **Model interpretation:** Analyze feature importance and the relationship between product pricing, outlet characteristics, and predicted sales.

The aim is to compare a simple baseline with more advanced models and select a model based on reliable validation and test-set performance, not merely its training score.

### Data Split

**Training:** 80%

**Validation:** Used during model development

**Testing:** 20%

## 5. Environment & Dependencies

Operating System: Windows 11

Python Version: 3.12

Main Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

### Virtual Environment Configuration

Environment Tool: `venv`

Environment Directory: `.venv/`

Create Environment:

```
python -m venv .venv
```

Activate Environment (Windows PowerShell):

```
.\.venv\Scripts\Activate.ps1
```

Deactivate Environment:

```
deactivate
```

Upgrade pip:

```
python -m pip install --upgrade pip
```

### Installation

Activate virtual environment above then

```
python -m pip install -r requirements.txt
```

### Reproducibility

Random Seed: 42

Environment: Python virtual environment (`.venv`)

Dependency Management: `requirements.txt`

Development Dataset Version: Version 0.1.0

## 6. Data Understanding

### Initial Inspection

The dataset was inspected for:

- Number of observations
- Number of features
- Data types
- Missing values
- Duplicate records
- Unique values
- Numerical distributions
- Categorical variables
- Target distribution

### Initial Observations

1. The dataset contains both numerical and categorical features.
2. No features contain missing values.
3. One duplicate row was identified.
4. Some features show low multicollinearity.

## 7. Data Cleaning

### Missing Values

**Problem:** No columns contain missing values.

**Detection:** Pandas isnull() analysis.

**Treatment:** Numerical variables were imputed using appropriate statistical values, while categorical variables were handled according to their meaning and missing-value behavior.

**Reason:** The model requires a complete numerical feature matrix.

### Duplicates

**Detected:** No significant duplicate observations were identified.

**Action:** Dataset was checked before modeling.

**Reason:** Duplicate observations can distort model training.

### Invalid Values

**Problem:** Some variables contained values requiring interpretation before modeling.

**Action:** Values were inspected and corrected or transformed where necessary.

**Reason:** Invalid values can introduce incorrect relationships into the model.

### Outliers

**Detection:** Statistical analysis and visualization.

**Treatment:** Investigated individually rather than automatically removing all outliers.

**Reason:** Extreme values may represent legitimate observations in the dataset.

### Data Leakage

**Potential leakage:** Features that directly reveal the target or information unavailable at prediction time.

**Detection:** Feature review and train/test preprocessing separation.

**Action:** Leakage-prone information was excluded.

## 8. Exploratory Data Analysis

### Univariate Analysis
The distributions of numerical variables were examined using histograms and descriptive statistics.

### Bivariate Analysis
Relationships between individual features and the target feature were examined using scatter plots, box plots, and correlation analysis.

### Multivariate Analysis
Feature relationships and correlations were examined to identify:

- Strong predictors
- Redundant features
- Multicollinearity
- Potential interactions

### Important Findings

1. Some features have a stronger relationship with the target variable than others.
2. Several numerical features are weakly correlated with one another.
3. The target distribution may exhibit skewness due to extreme values.

## 9. Feature Engineering


### Features Created

| Feature                 | Method / Formula                                          | Reason                                                                           |
| ----------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `item_visibility_ratio` | `Item_Visibility / average_item_visibility_within_outlet` | Measure product visibility relative to the average visibility within its outlet. |

### Features Removed

| Feature                  | Reason                                        |
| ------------------------ | --------------------------------------------- |
| Highly redundant feature | Reduce unnecessary redundancy                 |

### Numerical Transformation
- No Log transformation
- No Power transformation
- No Polynomial transformation

### Categorical Transformation
- One-Hot Encoding

### Scaling

**Method:** `StandardScaler`

**Reason:** Scale numerical features to comparable ranges and support preprocessing consistency.

### Feature Selection

**Methods Used**

- Correlation analysis
- Multicollinearity analysis
- RFECV
- Model-based selection


### Selected Features

```text
Item_Fat_Content
Item_Visibility
Item_MRP
Outlet_Establishment_Year
Outlet_Size
Outlet_Location_Type
Outlet_Type
item_visibility_ratio
````

### Removed Features

```
Item_Weight
Item_Identifier
Item_Type
Outlet_Identifier
```


## 10. Assumptions

### Data Assumptions
1. Historical relationships between features and insurance charges provide useful information for predicting future observations.
2. Training data is sufficiently representative of the population for which predictions will be made.
3. Feature values are measured consistently and accurately.

### Linear Regression Model Assumptions
1. A linear relationship exists between predictors and the target, including any engineered interaction terms.
2. Residuals are approximately independent.
3. Residual variance is reasonably constant (homoscedasticity).
4. Residuals are approximately normally distributed for statistical inference.
5. Severe multicollinearity among predictors should be assessed and managed.

## 11. Baseline Model

**Baseline:** Basic Linear Regression using the initial processed features.

### Purpose
Establish a reference performance level before applying additional feature engineering and feature selection.

### Results

| RMSE        | MAE        | R²       | CV Mean RMSE | CV RMSE Std |
|-------------|------------|----------|--------------|-------------|
| 1067.010252 | 790.864339 | 0.581117 | 1145.359411  | 15.206287   |


## 12. Model Selection

### Candidate Models

| Model                                | Reason for Testing                                            |
| ------------------------------------ | ------------------------------------------------------------- |
| Linear Regression                    | Establish an interpretable linear baseline                    |
| Ridge Regression                     | Investigate L2 regularization                                 |
| Lasso Regression                     | Investigate L1 regularization and feature selection           |
| Elastic Net Regression               | Combine L1 and L2 regularization                              |
| Random Forest Regression             | Capture nonlinear relationships and feature interactions      |
| Gradient Boosting Regression         | Improve predictive accuracy by sequentially correcting errors |
| XGBoost Regression                   | Test optimized gradient boosting with regularization          |
| LightGBM Regression                  | Test efficient gradient boosting with leaf-wise tree growth   |
| K-Nearest Neighbors (KNN) Regression | Test prediction based on similarities between observations    |
| Support Vector Regression (SVR)      | Model nonlinear relationships using kernels                   |

### Model Selection Criteria

- Predictive performance
- Generalization
- Interpretability
- Computational cost
- Model assumptions

## 13. Model Training

**Models:** Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, XGBoost, LightGBM, KNN, and SVR.

**Features:** Original numerical and categorical features, engineered features, and interaction terms.

**Training Data:** Training dataset.

**Preprocessing:** Missing-value handling, feature scaling, and one-hot encoding within a scikit-learn pipeline.

**Hyperparameter Tuning:** Cross-validation with model-specific parameter grids.

**Random State:** 42.

### Training Process

1. Split the data.
2. Fit preprocessing transformations on training data.
3. Transform validation/test data.
4. Train the regression model.
5. Generate predictions.
6. Evaluate performance.

## 14. Hyperparameter Tuning

**Method:** Grid search with cross-validation.

**Cross-Validation:** 5-fold cross-validation.

### Parameters Tested

Model-specific hyperparameters for Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, XGBoost, LightGBM, KNN, and SVR.

### Best Parameters

Selected the best hyperparameters for each model based on cross-validation performance. Linear Regression retained its default configuration.


## 15. Experiment Tracking

| Experiment | Model | Features | Parameters | Metric | Result (RMSE) | Notes |
|---|---|---|---|---|---:|---|
| 1 | Linear Regression | Selected | `fit_intercept=True` | Tuned CV RMSE | 1,145.36 | Baseline |
| 2 | Ridge Regression | Selected | `alpha=0.1` | Tuned CV RMSE | 1,145.36 | L2 regularization |
| 3 | Lasso Regression | Selected | `alpha=1.0`, `max_iter=20000` | Tuned CV RMSE | 1,145.61 | L1 regularization and feature selection |
| 4 | Elastic Net Regression | Selected | `alpha=0.1`, `l1_ratio=0.8`, `max_iter=20000` | Tuned CV RMSE | 1,153.47 | Combined L1 and L2 regularization |
| 5 | Random Forest Regression | Selected | `max_depth=10`, `min_samples_leaf=1`, `n_estimators=200` | Tuned CV RMSE | 1,113.28 | Nonlinear relationships |
| 6 | Gradient Boosting Regression | Selected | `learning_rate=0.03`, `max_depth=3`, `n_estimators=200` | Tuned CV RMSE | 1,092.50 | Strong tuned CV performance |
| 7 | XGBoost Regression | Selected | Tuned parameters | Tuned CV RMSE | 1,091.53 | Best tuned CV performance |
| 8 | LightGBM Regression | Selected | Tuned parameters | Tuned CV RMSE | 1,094.50 | Best tuned test R² |
| 9 | KNN Regression | Selected | `n_neighbors=11`, `p=2`, `weights='uniform'` | Tuned CV RMSE | 1,137.11 | Similarity-based predictions |
| 10 | Support Vector Regression (SVR) | Selected | `C=100`, `epsilon=0.2`, `kernel='rbf'` | Tuned CV RMSE | 1,131.73 | Nonlinear kernel-based regression |

### Experiment Notes

- All models were evaluated using the same train-test split and 5-fold cross-validation approach.
- Hyperparameters were tuned using model-specific parameter grids and cross-validation.
- XGBoost achieved the lowest tuned cross-validation RMSE of **1,091.53**.
- LightGBM achieved the best tuned test R² of **0.616087**, with an RMSE of **1,021.50** and MAE of **716.02**.
- Model selection should prioritize cross-validation performance, using test-set metrics for final evaluation.

---

## 16. Model Evaluation & Consolidated Regression Model Comparison

### Baseline Model Results: Linear Regression

| RMSE | MAE | R² | CV Mean RMSE | CV RMSE Std |
|---:|---:|---:|---:|---:|
| 1,067.01 | 790.86 | 0.581117 | 1,145.36 | 15.21 |

### Consolidated Regression Model Comparison

| Model | Initial CV RMSE | Initial CV RMSE Std | Initial Test RMSE | Initial Test MAE | Initial Test R² | Tuned CV RMSE | Tuned Test RMSE | Tuned Test MAE | Tuned Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 1,191.07 | 13.48 | 1,106.72 | 770.09 | 0.549363 | 1,091.53 | 1,020.89 | 718.55 | 0.616547 |
| Gradient Boosting | 1,095.89 | 13.42 | 1,033.59 | 720.48 | 0.606945 | 1,092.50 | 1,028.11 | 718.49 | 0.611102 |
| LightGBM | 1,128.32 | 17.15 | 1,049.41 | 728.96 | 0.594825 | 1,094.50 | 1,021.50 | 716.02 | 0.616087 |
| Random Forest | 1,164.88 | 6.74 | 1,090.05 | 758.34 | 0.562829 | 1,113.28 | 1,041.15 | 727.93 | 0.601177 |
| SVR | 1,672.76 | 23.90 | 1,561.23 | 1,163.83 | 0.103214 | 1,131.73 | 1,043.15 | 725.23 | 0.599639 |
| KNN | 1,198.70 | 13.36 | 1,122.03 | 777.55 | 0.536804 | 1,137.11 | 1,057.27 | 735.34 | 0.588728 |
| Ridge | 1,145.37 | 15.15 | 1,067.21 | 790.96 | 0.580962 | 1,145.36 | 1,067.03 | 790.87 | 0.581101 |
| Linear Regression | 1,145.36 | 15.21 | 1,067.01 | 790.86 | 0.581117 | 1,145.36 | 1,067.01 | 790.86 | 0.581117 |
| Lasso | 1,145.61 | 14.90 | 1,068.69 | 791.98 | 0.579800 | 1,145.61 | 1,068.69 | 791.98 | 0.579800 |
| Elastic Net | 1,338.18 | 11.86 | 1,269.14 | 966.48 | 0.407380 | 1,153.47 | 1,077.92 | 795.48 | 0.572504 |

### Best Parameters

| Model             | Best Parameters                                                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| XGBoost           | `model__colsample_bytree`: 1.0, `model__learning_rate`: 0.1, `model__max_depth`: 2, `model__n_estimators`: 100, `model__subsample`: 1.0                           |
| LightGBM          | `model__colsample_bytree`: 1.0, `model__learning_rate`: 0.03, `model__max_depth`: 3, `model__n_estimators`: 200, `model__num_leaves`: 15, `model__subsample`: 0.8 |
| Gradient Boosting | `learning_rate=0.03`, `max_depth=3`, `n_estimators=200`                                                                                                           |
| KNN               | `n_neighbors=11`, `p=2`, `weights='uniform'`                                                                                                                      |
| Random Forest     | `max_depth=10`, `min_samples_leaf=1`, `n_estimators=200`                                                                                                          |
| Ridge             | `alpha=0.1`                                                                                                                                                       |
| Lasso             | `alpha=1.0`, `max_iter=20000`                                                                                                                                     |
| Linear Regression | `fit_intercept=True`                                                                                                                                              |
| Elastic Net       | `alpha=0.1`, `l1_ratio=0.8`, `max_iter=20000`                                                                                                                     |
| SVR               | `C=100`, `epsilon=0.2`, `kernel='rbf'`                                                                                                                            |

### Evaluation Summary

- **Best cross-validation model:** XGBoost, with a tuned CV RMSE of **1,091.53**.
- **Best test-set RMSE:** XGBoost, with a tuned test RMSE of **1,020.89**.
- **Best test-set MAE:** LightGBM, with a tuned test MAE of **716.02**.
- **Best test-set R²:** XGBoost, with a tuned test R² of **0.616547**.
- **Baseline:** Linear Regression achieved a test R² of **0.581117**.
- **Effect of tuning:** XGBoost, LightGBM, Random Forest, SVR, KNN, and Elastic Net improved their CV RMSE after tuning. Gradient Boosting, Ridge, Lasso, and Linear Regression retained similar or slightly different CV performance.
- **Final selection:** XGBoost is preferred based on its lowest tuned CV RMSE and test RMSE. LightGBM is a strong alternative because it achieved the lowest test MAE.


**Output file:** `outputs/tables/final_model-models_comparism.csv`

## 17. Error Analysis

### Major Errors

1. Evaluate prediction errors using RMSE, MAE, and R².
2. Some predictions may substantially overestimate or underestimate actual values.
3. Residual plots help examine error distributions, systematic patterns, and unequal error variance.

### Possible Causes

1. Limited or missing predictive features.
2. Nonlinear relationships between features and the target variable.
3. Outliers or unusual observations.
4. Feature interactions that the model does not fully capture.
5. Differences between the training and test data distributions.

### Improvements

1. Investigate observations with the largest absolute residuals.
2. Improve feature engineering and evaluate useful feature interactions.
3. Examine residual plots for systematic patterns and unequal error variance.
4. Compare models using cross-validation and appropriate evaluation metrics.
5. Investigate whether target transformation or alternative models improve predictive performance.

## 18. Model Interpretation

### Feature Importance
Permutation importance was used to evaluate how shuffling individual input features affected the selected model's predictive performance.

### Explainability
**Methods:** Permutation importance and regression coefficients, where applicable.

### Key Findings
1. Feature importance helps identify features that contribute most to predictive performance.
2. Permutation importance measures the change in model performance when a feature is shuffled.
3. Features with low or negative permutation importance may provide limited useful predictive information.
4. Correlated features can influence the interpretation of individual feature importance scores.
5. Feature importance indicates predictive contribution, not necessarily causation.

**Note:** Confirm the leading features using the model's feature importance results before reporting specific findings.

## 19. Bias & Data Quality

### Potential Bias
The dataset may represent only a specific population, location, or period. Consequently, model performance may not generalize to different populations or future observations.

### Data Quality Issues
1. Missing values, where present, requiring appropriate treatment.
2. Numerical features with different scales.
3. Potential outliers and unusual observations.
4. Categorical features requiring appropriate encoding.
5. Duplicate records or inconsistent values, where present.
6. Potential differences between training data and future prediction data.

### Impact
These issues can affect model performance, interpretation, and generalization. The preprocessing pipeline helps handle missing values, feature scaling, and categorical encoding where required. However, preprocessing cannot eliminate dataset bias or guarantee accurate predictions on data outside the original distribution.

## 20. Final Model

**Model:** XGBoost Regressor  
**Version:** 1.0  
**Selection Criterion:** Lowest tuned cross-validation RMSE  
**Features:** Selected original, engineered, and encoded features

### Final Configuration

```text
Model: XGBRegressor
Hyperparameters: Best parameters selected through cross-validation
Preprocessing: Scikit-learn Pipeline
Features: Selected features and engineered interactions
Cross-Validation: 5-fold
Random State: 42
```


### Final Performance

| Metric                    | Initial Model | Tuned Model   |
| ------------------------- | ------------- | ------------- |
| Cross-Validation RMSE     | 1,191.07      | 1,091.53      |
| Cross-Validation RMSE Std | 13.48         | Not available |
| Test RMSE                 | 1,106.72      | 1,020.89      |
| Test MAE                  | 770.09        | 718.55        |
| Test R²                   | 0.549363      | 0.616547      |

**Validation Note:** XGBoost was selected because it achieved the lowest tuned cross-validation RMSE of **1,091.53** among the ten candidate models. Tuning reduced its cross-validation RMSE by approximately **8.36%** and improved its test RMSE, MAE, and R². The final model explains approximately **61.65%** of the variation in Big Mart product sales in the test set.

### Feature Importance

Permutation importance was used to evaluate how shuffling individual input features affected the selected model's predictive performance. Higher values indicate a greater deterioration in model performance after shuffling.

| Rank | Feature | Importance | Importance Std |
|---|---|---|---|
| 1 | To be confirmed | Not available | Not available |
| 2 | To be confirmed | Not available | Not available |
| 3 | To be confirmed | Not available | Not available |

**Feature Importance Note:** Insert the actual leading features and their values from the final model's permutation importance results before drawing conclusions. Because permutation importance was calculated using the complete pipeline and original test features, the results represent original input features rather than separate engineered features.

### Final Validation Summary

The tuned XGBoost Regressor was selected based on its lowest cross-validation RMSE. It achieved a test R² of **0.616547**, RMSE of **1,020.89**, and MAE of **718.55**. LightGBM achieved a slightly lower test MAE of **716.02** and a test RMSE of **1,021.50**, while Gradient Boosting achieved a test RMSE of **1,028.11**.

These results apply to the test data used in this project. Performance on new data may differ.


## 21. Model Persistence

**Method:** Joblib

```text
models/
 └── best_regression_model.joblib
```

### Required Preprocessing

The saved pipeline contains the preprocessing required to transform incoming data before generating predictions.


## 22. Prediction / Inference

### Input Requirements

The final XGBoost Regressor requires the same raw input features used during model training. The fitted preprocessing pipeline handles the required transformations, including missing-value imputation, categorical encoding, and feature engineering.

**Expected Input Features**

```text
Item_Fat_Content
Item_Visibility
Item_MRP
Outlet_Establishment_Year
Outlet_Size
Outlet_Location_Type
Outlet_Type
````

The pipeline also generates the engineered feature:

\- `item_visibility_ratio`

Important: Supply the original input columns expected by the trained pipeline. Do not manually encode categorical features or create engineered features if these transformations are already included in the pipeline. Missing values should be handled according to the fitted preprocessing configuration.

## 23. Deployment (Future)

**Deployment Type:** Local application / API / Web application

**Technology:** FastAPI or Streamlit

### Deployment Structure

```text
deployment/
 ├── api/
 ├── app/
 └── config/
```

## 24. Testing (Future)

### Tests

- Data validation
- Preprocessing
- Model prediction
- Missing input handling
- Invalid input handling
- Edge cases
- Prediction output

### Results

**Tests Passed:** N/A

**Tests Failed:** N/A

## 25. Problems & Solutions

| Problem | Cause | Solution |
|---|---|---|
| Missing values | Incomplete observations | Appropriate imputation or removal where justified |
| Categorical features | Non-numerical values | Suitable categorical encoding |
| Multicollinearity | Correlated predictors | Feature analysis and regularization where appropriate |
| Skewed variables | Asymmetric distributions | Transformation where appropriate |
| Large residuals | Difficult-to-predict observations | Residual analysis and error investigation |
| Model overfitting | Excessive model complexity or sensitivity to training data | Cross-validation, regularization, and hyperparameter tuning |
| Inconsistent model performance | Differences between validation and test results | Compare cross-validation and test metrics |
| Data leakage | Information from validation or test data influencing training | Use appropriate data splitting and pipelines |
| Model reproducibility | Different settings or preprocessing between runs | Fixed random seeds, consistent pipelines, and saved model artifacts |
| Poor generalization | Differences between training data and real-world data | Evaluate on representative, unseen data |

## 26. Lessons Learned

### Machine Learning

1. Data quality and preprocessing influence model performance.
2. Cross-validation provides a more reliable basis for model selection than a single validation result.
3. Hyperparameter tuning can improve cross-validation performance without improving every test metric.
4. Feature importance helps identify influential predictors but does not establish causation.
5. Residual analysis helps reveal systematic errors and observations the model predicts poorly.
6. Different algorithms perform differently on the same dataset.
7. Evaluation metrics should be selected according to the problem and project objectives.

### Technical

1. Scikit-learn pipelines help keep preprocessing consistent during training and prediction.
2. Preprocessing and feature selection must be fitted without leaking information from validation or test data.
3. Saving the fitted pipeline allows predictions to be generated without retraining.
4. Permutation importance helps assess how individual input features contribute to predictive performance.
5. Consistent evaluation procedures make model comparisons more meaningful.

### Project Development

1. Documentation makes experiments, results, and decisions easier to track.
2. Separating data, source code, models, and outputs improves project organization.
3. Reproducibility should be considered from the beginning.
4. Comparing initial and tuned models helps explain the effects of hyperparameter tuning.
5. Recording model configurations and evaluation metrics supports future improvements.

## 27. Limitations

1. Model performance depends on the quality, completeness, and representativeness of the dataset.
2. The selected model may not generalize equally well to different populations, locations, or time periods.
3. Hyperparameter tuning does not guarantee improvements across every evaluation metric.
4. Test results may not fully represent performance on future, unseen data.
5. Correlated features can affect the interpretation of permutation importance.
6. Feature importance describes predictive contribution, not causal relationships.
7. Predictions may be less reliable for observations that differ substantially from the training data.
8. Model performance depends on the relevance and availability of the input features.

## 28. Future Improvements

1. Evaluate the model on additional representative datasets or an appropriate future-period holdout.
2. Investigate observations with the largest absolute prediction errors.
3. Explore additional feature engineering and meaningful feature interactions.
4. Evaluate the stability of feature importance across cross-validation folds.
5. Compare additional hyperparameter configurations using a controlled validation process.
6. Assess alternative preprocessing methods and target transformations where appropriate.
7. Develop an API for serving predictions.
8. Build a user-facing interface for entering input details and obtaining predictions.
9. Monitor prediction accuracy and data quality if the model is deployed.
10. Document model versions, dependencies, and retraining procedures for future maintenance.

## 29. Final Results

### Summary

The project developed an end-to-end machine learning workflow to predict medical insurance charges using the Medical Cost Personal Dataset. The workflow included data preprocessing, feature engineering, model comparison, cross-validation, hyperparameter tuning, model evaluation, residual diagnostics, feature-importance analysis, and model persistence.

### Final Model

**Model:** XGBoost Regressor  
**Version:** 1.0  
**Selection Criterion:** Lowest tuned cross-validation RMSE  
**Main Metric:** Root Mean Squared Error (RMSE)

| Metric | Final Result |
|---|---:|
| Tuned Cross-Validation RMSE | 1,091.53 |
| Test RMSE | 1,020.89 |
| Test MAE | 718.55 |
| Test R² | 0.616547 |

### Key Findings

1. Data preprocessing and feature engineering prepared the input features for model training.
2. XGBoost achieved the lowest tuned cross-validation RMSE among the ten evaluated models.
3. The final model achieved a test R² of **0.616547**, explaining approximately 61.65% of the variation in Big Mart product sales in the test set.
4. LightGBM achieved a slightly lower test MAE of 716.02, while XGBoost achieved the lowest test RMSE of 1,020.89.
5. Hyperparameter tuning improved XGBoost's cross-validation RMSE, test RMSE, test MAE, and test R² compared with its initial configuration.
6. Feature importance results should be used to identify influential input features, while recognizing that predictive importance does not establish causation.
7. The final model provides a predictive baseline for Big Mart sales, but its performance on new or unseen data may differ.


## 30. Project Structure

```text
house-price-prediction/
 │
 ├── data/
 │   ├── raw/
 │   ├── external/
 │   └── processed/
 │
 ├── notebooks/
 │
 ├── src/
 │   ├── data/
 │   ├── features/
 │   ├── models/
 │   ├── evaluation/
 │   └── visualization/
 │
 ├── models/
 ├── outputs/
 ├── tests/
 ├── config/
 ├── scripts/
 ├── docs/
 │
 ├── README.md
 ├── requirements.txt
 └── .gitignore
```

## 31. References

### Dataset

- [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) - Kaggle dataset used for medical insurance charge prediction.

### Documentation

- [Python Documentation](https://docs.python.org/3/) - Python language reference.
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data manipulation and analysis.
- [NumPy Documentation](https://numpy.org/doc/) - Numerical computing.
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - Preprocessing, pipelines, model training, evaluation, and hyperparameter tuning.
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) - XGBoost regression and model configuration.
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/) - LightGBM regression and model configuration.
- [Matplotlib Documentation](https://matplotlib.org/stable/) - Data visualization.
- [Seaborn Documentation](https://seaborn.pydata.org/) - Statistical data visualization.

### Research and Learning Resources

- [Scikit-learn Model Evaluation Guide](https://scikit-learn.org/stable/modules/model_evaluation.html) - Regression metrics and model evaluation.
- [Scikit-learn Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html) - Cross-validation strategies.
- [Scikit-learn Permutation Importance Guide](https://scikit-learn.org/stable/modules/permutation_importance.html) - Model interpretation using permutation importance.
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) - Gradient boosting concepts and implementation.

### Development Tools

- [Jupyter Documentation](https://docs.jupyter.org/) - Interactive notebooks.
- [Visual Studio Code Documentation](https://code.visualstudio.com/docs) - Development environment and Python workflows.
- [Git Documentation](https://git-scm.com/doc) - Version control and project tracking.

## 32. Progress Log

### LOG - STARTUP
#### 2026-10-25

>**Task:** Project setup
>**What I Did:** Defined the project objective, problem type, and success criteria; created the project repository and folder structure; configured the environment, dependencies, Git, .gitignore, documentation, and progress log; obtained and organized the dataset.
>**Result:** Project environment, repository, documentation, and dataset were successfully set up.
>**Problem:** None.
>**Decision:** Proceed with detailed data understanding and analysis.
>**Next Step:** Data Preparation & Cleaning.

---

### LOG - DATA PREPARATION & CLEANING
#### 2026-10-25

> **Task:** Data Preparation and Cleaning  
> **What I Did:** Loaded and inspected the dataset; examined its structure, dimensions, data types, missing values, duplicates, outliers, inconsistencies, and target variable; used reusable functions to support data inspection and cleaning.  
> **Result:** Data quality was assessed, and the dataset was prepared for further analysis where necessary.  
> **Problem:** Any unresolved data quality issues or limitations were identified for further investigation.  
> **Decision:** Applied appropriate cleaning operations based on the findings while preserving valid data and avoiding unnecessary information loss.  
> **Next Step:** Exploratory Data Analysis (EDA).

---

### LOG - EDA
#### 2026-10-25

> **Task:** Exploratory Data Analysis (EDA)  
> **What I Did:** Analyzed the target variable and feature distributions; examined relationships between features and the target, correlations, outliers, and potential feature interactions; documented key patterns and findings.  
> **Result:** Identified important data patterns, potential predictors, distribution characteristics, and relationships to guide subsequent preprocessing and modeling decisions.  
> **Problem:** Some distributions, correlations, and unusual observations required careful consideration before applying transformations or removing features.  
> **Decision:** Used the findings to guide feature engineering, transformations, encoding, and outlier handling based on their potential impact on model performance.  
> **Next Step:** Feature Engineering.

---
### LOG - FEATURE ENGINEERING
#### 2026-10-25

> **Task:** Feature Engineering  
> **What I Did:** Separated the features (`X`) from the target (`y`); created additional features where appropriate; evaluated potential feature interactions and transformations; preserved the original dataset.  
> **Result:** The engineered features were prepared for evaluation during preprocessing and model development.  
> **Problem:** Assessed whether the new features introduced redundancy, multicollinearity, or unnecessary complexity.  
> **Decision:** Retain useful features based on their relevance and impact on model performance.  
> **Next Step:** Data Preprocessing Pipeline.

---

### LOG - DATA PREPROCESSING PIPELINE
#### 2026-10-25

> **Task:** Build the Data Preprocessing Pipeline  
> **What I Did:** Identified and applied appropriate preprocessing techniques for numerical and categorical features; organized transformations into a reusable pipeline; ensured preprocessing was fitted on training data only.  
> **Result:** A consistent preprocessing pipeline was established for model training, evaluation, and prediction.  
> **Problem:** Preprocessing requirements varied by feature type and model, requiring appropriate technique selection and validation.  
> **Decision:** Retain preprocessing steps that support data quality and model performance while preventing data leakage.  
> **Next Step:** Modeling.


---

### LOG - MODELING
#### 2026-10-25

> **Task:** Model Development and Evaluation  
> **What I Did:** Established a baseline model; trained and evaluated candidate regression models; applied cross-validation and hyperparameter tuning; compared performance metrics and analyzed prediction errors.  
> **Result:** Candidate models were evaluated and compared to identify the most suitable model based on predictive performance and generalization.  
> **Problem:** Differences in model performance and prediction errors required careful evaluation to avoid overfitting and unreliable conclusions.  
> **Decision:** Select the most suitable model based on cross-validation results, test performance, prediction errors, and generalization.  
> **Next Step:** Model Finalization.

---

### LOG - MODEL FINALIZATION
#### 2026-10-25

> **Task**: Finalize the Selected Model  
> **What I Did**: Reviewed the model development process; confirmed the selected model and its configuration; documented evaluation findings; saved relevant model artifacts, code, configuration, and results.  
> **Result**: The modeling process was completed, and the final outputs were organized for reproducibility and future reference.  
> **Problem**: Documented any unresolved limitations, technical issues, or areas requiring further improvement.  
> **Decision**: Finalized the project outputs based on the available evidence and documented any limitations affecting their use.  
> **Next Step**: Experiment Tracking.

---

### LOG - EXPERIMENT TRACKING
#### 2026-10-25

>**Task:** Record and track experiments
>**What I Did:** Documented candidate models, feature engineering, preprocessing, hyperparameters, evaluation metrics, observations, and model-selection decisions.
>**Result:** Experiment history and the reasoning behind the final model selection were documented for reproducibility.
>**Problem:** None.
>**Decision:** Maintain experiment records and version-control relevant project changes.
>**Next Step:** Pre-deployment.

#### 2026-10-25

>**Task:** Update project progress and version control
>**What I Did:** Updated the progress log, committed project changes, and pushed the changes to GitHub.
>**Result:** Project progress was documented, and the changes were pushed to the remote repository.
>**Problem:** None.
>**Decision:** Continue preparing the project for potential deployment.
>**Next Step:** Pre-deployment.

---

### LOG - PRE-DEPLOYMENT
#### 2026-10-25

>**Task:** Prepare the project for deployment
>**What I Did:** Cleaned the codebase, organized notebooks and source files, removed unnecessary files, added appropriate tests, checked reproducibility and dependencies, updated environment configuration and `.gitignore`, and reviewed project documentation.
>**Result:** The project structure was cleaned and organized, and reproducibility and deployment preparation were reviewed.
>**Problem:** The README and complete reproduction instructions still required final review and completion.
>**Decision:** Complete the remaining documentation and verify that the project can be reproduced from a clean environment.
>**Next Step:** Complete project documentation.

---
### LOG - FINALIZE
#### 2026-10-25

>**Task:** Finalize the project
>**What I Did:** Reviewed the final project files, README, and documentation; checked Git status; committed and pushed final changes; prepared a version release; organized portfolio materials and archived project notes.
>**Result:** The project was finalized, documented, version-controlled, and published.
>**Problem:** None.
>**Decision:** Mark the project as complete.
>**Next Step:** Start the next project.