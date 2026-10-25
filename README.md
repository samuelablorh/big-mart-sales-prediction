# Big Mart Sales Prediction

A machine learning project that predicts product sales across Big Mart stores using the Big Mart Sales dataset. The project covers data preparation, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, evaluation, and feature importance analysis.

## Project Overview

**Objective:** Predict product sales (`Item_Outlet_Sales`) using product attributes and store characteristics.

**Dataset:** BigMart Sales Data, Kaggle.

**Problem type:** Supervised learning, Regression.

**Final model:** XGBoost Regressor, selected based on the lowest tuned cross-validation RMSE.

**Tools and technologies:**
- Python
- pandas and NumPy
- scikit-learn
- XGBoost and LightGBM
- Matplotlib and Seaborn
- Joblib
- VS Code and Jupyter notebooks
- Git and GitHub

## Dataset

The dataset contains approximately 8,523 observations and 12 columns, including the target variable `Item_Outlet_Sales`.

The features describe different aspects of products and retail outlets, including:
- Product identifiers and characteristics
- Product visibility
- Maximum Retail Price (MRP)
- Outlet establishment year
- Outlet size and location
- Outlet type

**Target variable:** `Item_Outlet_Sales`, the recorded sales of a product at a particular retail outlet.

**Dataset source:** [Kaggle: BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

Download the dataset from Kaggle and place the CSV file in the project's `data/raw/` directory using the filename expected by the notebooks: `big_mart_sales.csv`.

## Project Workflow

The project follows a typical machine learning workflow:

1. **Data loading**: Load the dataset into pandas DataFrames.
2. **Data inspection**: Examine the shape, data types, summary statistics, missing values, duplicates, and target distribution.
3. **Exploratory data analysis**: Investigate relationships between product and outlet characteristics and sales.
4. **Data cleaning**: Check duplicates, data quality, and numerical and categorical columns.
5. **Feature engineering**: Create `item_visibility_ratio` to measure product visibility relative to the average visibility within its outlet.
6. **Data splitting**: Separate training and testing data using an 80/20 split.
7. **Preprocessing**: Apply appropriate categorical encoding, numerical scaling where required, and preprocessing pipelines.
8. **Model training**: Train and compare multiple regression algorithms.
9. **Model evaluation**: Compare performance using 5-fold cross-validation and test-set metrics.
10. **Hyperparameter tuning**: Search for better model settings using GridSearchCV.
11. **Feature importance**: Use permutation importance to investigate the contribution of input features to predictive performance.
12. **Model selection**: Select the model with the lowest tuned cross-validation RMSE.

## Models Compared

The following regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regression (SVR)

The final model was selected based on its tuned cross-validation RMSE.

## Model Performance

The XGBoost Regressor was selected as the final model based on its lowest tuned cross-validation RMSE and its test-set performance.

### Initial Model

| Metric | Result |
|---|---:|
| Cross-validation RMSE | 1,191.07 |
| Cross-validation RMSE standard deviation | 13.48 |
| Test RMSE | 1,106.72 |
| Test MAE | 770.09 |
| Test R² | 0.549363 |

### Tuned Model

The selected hyperparameters were determined through GridSearchCV. Refer to the experiment results and saved model configuration for the exact parameter values.

| Metric | Result |
|---|---:|
| Cross-validation RMSE | 1,091.53 |
| Test RMSE | 1,020.89 |
| Test MAE | 718.55 |
| Test R² | 0.616547 |

### Interpreting the Results

- **RMSE:** Measures prediction error, penalizing larger errors more heavily. Lower is better.
- **MAE:** Measures the average absolute difference between predicted and actual product sales. Lower is better.
- **R²:** Measures the proportion of target variance explained by the model relative to a mean-prediction baseline. Higher is generally better.

Hyperparameter tuning reduced the XGBoost cross-validation RMSE by approximately 8.36%. The tuned model also improved its test-set RMSE, MAE, and R² compared with the initial model.

LightGBM achieved the lowest test-set MAE among the tuned models:

| Metric | Result |
|---|---:|
| Cross-validation RMSE | 1,094.50 |
| Test RMSE | 1,021.50 |
| Test MAE | 716.02 |
| Test R² | 0.616087 |

All error values are expressed in the same units as the dataset's `Item_Outlet_Sales` target.

## Feature Importance

Permutation importance was used to investigate which input features most affected the model's predictive performance.

The numerical feature-importance rankings and values are not included here because the final permutation-importance results were not provided in the project information.

**Important:** Permutation importance measures predictive usefulness, not causation. It does not prove that changing a feature will cause a particular change in product sales.

## Project Structure

The following is a suggested structure. Adjust the paths to match the files in your repository.

```text
big-mart-sales-prediction/
├── data/
│   ├── raw/
│   │   └── big_mart_sales.csv
│   ├── external/
│   └── processed/
├── notebooks/
│   └── big_mart_sales_prediction.ipynb
├── models/
│   └── best_regression_model.joblib
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
````

- `data/` — Dataset files or instructions for obtaining them.
- `notebooks/` — Notebooks containing analysis and experiments.
- `models/` — Saved model artifacts, if included.
- `outputs/tables/` — Generated evaluation tables and model comparison results.
- `src/` — Reusable Python modules, if the project uses them.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — Files and directories that Git should not track.
- `README.md` — Project documentation.

Only include directories and filenames that actually exist in the repository. Large datasets and generated model files may be better kept outside Git or managed using an appropriate data-versioning approach.

## Getting Started

### 1. Install Prerequisites

Install:

- Git
- Python 3.12 or another compatible Python version
- VS Code or JupyterLab

Verify that Python and Git are available:

```
python --version
git --version
```

### 2. Clone the Repository

Open a terminal and run:

```
git clone https://github.com/YOUR_USERNAME/big-mart-sales-prediction.git
cd big-mart-sales-prediction
```

Replace the example GitHub URL with the actual URL of your repository.

Cloning an existing repository already configures the remote named `origin`. You do not need to run `git init` or `git remote add origin` after cloning.

### 3. Create a Virtual Environment

Create a project-specific virtual environment:

```
python -m venv .venv
```

A virtual environment isolates this project's Python packages from other projects and the system-wide Python installation.

### 4. Activate the Virtual Environment

Windows PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```
source .venv/Scripts/activate
```

After activation, your terminal will usually display `(.venv)` at the beginning of the command prompt.

If PowerShell blocks activation because of its execution policy, you can use Git Bash or consult the official Python documentation for the appropriate activation guidance.

### 5. Verify the Environment

With the virtual environment activated, run:

```
python --version
where.exe python
python -m pip --version
```

On Windows, `where.exe python` should list the virtual environment's Python executable first.

The `python -m pip --version` command should also indicate that pip belongs to the virtual environment.

### 6. Install Dependencies

Upgrade pip within the active virtual environment:

```
python -m pip install --upgrade pip
```

Install the project's dependencies:

```
python -m pip install -r requirements.txt
```

Important: Always activate `.venv` before installing packages, and use `python -m pip` instead of a standalone `pip` command. This helps ensure that packages are installed into the intended Python environment.

If the repository does not yet contain `requirements.txt`, install the necessary packages:

```
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm jupyter
```

Then generate the dependency file:

```
python -m pip freeze > requirements.txt
```

Review the generated file before committing it. Ideally, it should reflect the packages required by the project rather than unrelated packages installed in the environment.

### 7. Open the Notebook

With the virtual environment activated, open the project in VS Code and select the `.venv` Python environment as the notebook kernel.

Alternatively, launch Jupyter:

```
python -m jupyter lab
```

Open `big_mart_sales_prediction.ipynb` and execute the cells in their intended order.

### 8. Obtain the Dataset

Download the dataset from the [Kaggle BigMart Sales Data page](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?utm_source=chatgpt.com).

Place the CSV file in `data/raw/` using the filename expected by the notebook: `big_mart_sales.csv`. If the dataset is not included in the repository, ensure that the download instructions and expected path are documented.

## Reproducibility

For consistent results:

- Use the documented Python environment and dependencies.
- Set the random seed to `42` where supported.
- Keep training and test data separate.
- Fit preprocessing steps using training data only.
- Use 5-fold cross-validation for model comparison and hyperparameter tuning.
- Evaluate the final selected model on held-out test data.
- Record the model parameters and evaluation metrics.

Results can vary if package versions, random seeds, data splits, or preprocessing procedures differ.

## Limitations and Future Improvements

Potential improvements include:

- Testing additional feature-engineering approaches and feature interactions.
- Investigating observations with the largest prediction errors.
- Examining residuals and prediction errors across different sales ranges.
- Comparing additional hyperparameter search strategies.
- Evaluating model stability across different data splits.
- Confirming the leading features using the final permutation-importance results.
- Saving the complete preprocessing and prediction pipeline.
- Building a simple interface or API for sales predictions.

The model's performance on the original test split does not guarantee equivalent performance on future sales data or other retail environments.

## References

- [Kaggle: BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?utm_source=chatgpt.com)
- [Python Documentation](https://docs.python.org/3/?utm_source=chatgpt.com)
- [Python ](https://docs.python.org/3/library/venv.html?utm_source=chatgpt.com)[`venv`](https://docs.python.org/3/library/venv.html?utm_source=chatgpt.com)[ Documentation](https://docs.python.org/3/library/venv.html?utm_source=chatgpt.com)
- [pip Documentation](https://pip.pypa.io/en/stable/?utm_source=chatgpt.com)
- [pandas Documentation](https://pandas.pydata.org/docs/?utm_source=chatgpt.com)
- [NumPy Documentation](https://numpy.org/doc/?utm_source=chatgpt.com)
- [scikit-learn Documentation](https://scikit-learn.org/stable/?utm_source=chatgpt.com)
- [XGBoost Documentation](https://xgboost.readthedocs.io/?utm_source=chatgpt.com)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/?utm_source=chatgpt.com)
- [Matplotlib Documentation](https://matplotlib.org/stable/?utm_source=chatgpt.com)
- [Seaborn Documentation](https://seaborn.pydata.org/?utm_source=chatgpt.com)

## Author

Samuel Ablordeppey

GitHub: [samuelablorh](https://github.com/samuelablorh?utm_source=chatgpt.com)