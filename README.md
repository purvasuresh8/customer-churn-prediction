# Customer Churn Prediction & Retention Analytics Platform

## Overview

An end-to-end machine learning and analytics platform designed to predict customer churn, identify retention drivers, classify churn risk, and generate actionable retention recommendations.

This project uses customer demographic and behavioral data from the IBM Telco Customer Churn dataset and applies machine learning techniques to predict customer attrition while providing business-oriented insights through an interactive analytics dashboard.

---

## Features

- Automated customer churn prediction
- Data validation and quality checks
- Data preprocessing and feature engineering
- Logistic Regression and Random Forest modeling
- Model evaluation and comparison
- Churn risk segmentation
- Retention strategy recommendations
- Interactive Streamlit dashboard
- Automated reporting and visualization

---

## Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- SHAP

### Visualization
- Matplotlib
- Seaborn
- Plotly
- Streamlit

### Data Source
- KaggleHub
- IBM Telco Customer Churn Dataset

---

## Project Architecture

```text
KaggleHub Dataset
        ↓
Data Validation
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Feature Engineered Dataset
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Pipeline
        ↓
Retention Recommendation Engine
        ↓
Interactive Dashboard
```

---

## Project Structure

```text
customer-churn-prediction/

README.md
requirements.txt

data/
├── raw/
├── processed/
└── predictions/

models/

reports/

visualizations/

src/
├── data_loader.py
├── data_validation.py
├── preprocessing.py
├── feature_engineering.py
├── run_feature_engineering.py
├── train_model.py
├── evaluate_model.py
├── predict.py
├── retention_recommendations.py
├── generate_report.py
├── model_explainability.py
├── dashboard.py
└── utils.py
```

---

## Dataset

### IBM Telco Customer Churn Dataset

Key attributes include:

- Customer demographic information
- Subscription information
- Contract type
- Billing method
- Monthly charges
- Total charges
- Churn status

Dataset Size:

```text
7043 customer records
21 original features
```

---

## Data Preprocessing

Implemented:

- Duplicate removal
- Missing value treatment
- Data type standardization
- Label encoding
- One-hot encoding
- Feature scaling

Output:

```text
data/processed/cleaned_data.csv
```

---

## Feature Engineering

Generated business-focused features:

- Customer tenure groups
- Spending segments
- Contract risk score
- Service count
- Customer value indicators
- Engagement scores

Output:

```text
data/processed/feature_engineered_data.csv
```

Features after engineering:

```text
40 total features
```

---

## Machine Learning Models

### Logistic Regression

Used as a baseline interpretable model.

### Random Forest

Used for non-linear relationship detection and feature importance analysis.

---

## Model Performance

### Logistic Regression

```text
Accuracy: 79.8%
Precision: 64.9%
Recall: 51.6%
F1 Score: 57.5%
ROC-AUC: 0.8397
```

### Random Forest

```text
Accuracy: 79.1%
Precision: 64.7%
Recall: 46.8%
F1 Score: 54.3%
ROC-AUC: 0.8360
```

### Best Performing Model

```text
Logistic Regression
```

Logistic Regression outperformed Random Forest on:

- Accuracy
- Recall
- F1 Score
- ROC-AUC

---

## Churn Prediction Pipeline

The prediction engine:

- Loads trained models
- Predicts churn probabilities
- Classifies risk levels
- Generates customer retention actions

Output:

```text
data/predictions/churn_predictions.csv
```

Example:

```text
Prediction | Probability | Risk

1          | 0.63        | Medium
0          | 0.04        | Low
1          | 0.68        | Medium
```

---

## Retention Recommendation Engine

Customers are segmented into:

### High Risk

- Immediate outreach
- Retention discounts
- Dedicated support

### Medium Risk

- Engagement campaigns
- Loyalty programs

### Low Risk

- Monitoring
- Regular promotions

Output:

```text
data/predictions/retention_plan.csv
```

---

## Dashboard

Built using Streamlit.

Features:

- Customer risk overview
- Churn probability analysis
- Retention recommendation insights
- Priority score visualization
- Interactive filtering
- Data export capability

Launch with:

```bash
streamlit run src/dashboard.py
```

---

# Challenges Encountered & Debugging Journey

This project was developed completely from scratch and required solving multiple real-world engineering and debugging challenges.

---

## 1. Dataset Parsing Failure

### Problem

The local dataset file accidentally contained Python code instead of actual CSV data.

Error:

```text
ParserError:
Expected 1 fields in line 6, saw 2
```

### Solution

- Removed corrupted file
- Switched to KaggleHub integration
- Automated dataset download and loading

### Result

Fully reproducible data ingestion workflow.

---

## 2. Dataset Management Problems

### Problem

Manual dataset placement caused repeated loading issues.

### Solution

Implemented KaggleHub dataset retrieval:

```python
kagglehub.dataset_download(
    "blastchar/telco-customer-churn"
)
```

### Result

No manual dataset management required.

---

## 3. Import Errors Across Modules

### Problem

Several modules attempted importing:

```python
load_data()
```

while the implementation contained:

```python
load_raw_data()
```

Error:

```text
ImportError:
cannot import name 'load_data'
```

### Solution

Standardized all data loading functions and imports.

---

## 4. Preprocessing Script Not Running

### Problem

Running:

```bash
python src/preprocessing.py
```

produced no output.

### Cause

The file only contained function definitions.

### Solution

Added:

```python
if __name__ == "__main__":
```

execution blocks.

### Result

Generated:

```text
cleaned_data.csv
```

---

## 5. Feature Engineering Pipeline Failure

### Problem

Feature engineering functions existed but were never executed.

### Solution

Created:

```text
run_feature_engineering.py
```

and automated saving.

### Result

Generated:

```text
feature_engineered_data.csv
```

---

## 6. Configuration Import Errors

### Problem

Training pipeline could not locate:

```python
config.py
```

Error:

```text
ModuleNotFoundError:
No module named 'config'
```

### Solution

Reorganized imports and project structure.

### Result

Centralized project configuration.

---

## 7. Model Evaluation Failure

### Problem

Evaluation crashed when passing:

```python
X.columns
```

Error:

```text
ValueError:
The truth value of a Index is ambiguous
```

### Solution

Converted feature names to Python lists and updated evaluation conditions.

### Result

Successful automated model evaluation.

---

## 8. Missing Directory Structure

### Problem

Models and reports could not be saved.

Error:

```text
FileNotFoundError:
models/logistic_regression.pkl
```

### Solution

Created required directories:

```text
models/
reports/
visualizations/
logs/
```

### Result

Successful model persistence.

---

## 9. Prediction Pipeline Failures

### Problem

Prediction scripts attempted to load models using undefined functions.

Error:

```text
NameError:
load_model is not defined
```

### Solution

Implemented standardized model loading using:

```python
joblib.load()
```

### Result

Predictions generated successfully.

---

## 10. Dashboard Syntax Errors

### Problem

Dashboard source code contained HTML tags:

```html
<br>
```

which caused:

```text
SyntaxError:
invalid syntax
```

### Solution

- Removed all HTML artifacts
- Cleaned Streamlit source code
- Revalidated dashboard execution

### Result

Successfully launched the interactive dashboard.

---

## Lessons Learned

During development, the following skills were strengthened:

- Data engineering
- ETL pipeline development
- Data validation
- Machine learning model development
- Feature engineering
- Debugging Python applications
- Dependency management
- Streamlit dashboard development
- Model deployment workflows
- End-to-end analytics platform design

---

## Final Results

| Metric | Logistic Regression | Random Forest |
|----------|----------|----------|
| Accuracy | 79.8% | 79.1% |
| Precision | 64.9% | 64.7% |
| Recall | 51.6% | 46.8% |
| F1 Score | 57.5% | 54.3% |
| ROC-AUC | 0.84 | 0.84 |

### Key Outcome

Logistic Regression achieved the strongest overall performance, reaching approximately **80% accuracy** and **0.84 ROC-AUC**, outperforming Random Forest on recall and F1-score while generating actionable customer retention recommendations through an interactive analytics dashboard.

---

## How to Run the Project

### 1. Clone Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Pipeline

```bash
python src/preprocessing.py
python src/run_feature_engineering.py
python src/train_model.py
python src/predict.py
python src/retention_recommendations.py
```

### 5. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Future Enhancements

- XGBoost implementation
- Hyperparameter tuning
- SHAP explainability dashboard
- Real-time prediction API
- Docker deployment
- Cloud hosting (AWS/Azure)
- CI/CD integration with GitHub Actions
- Automated PDF business reporting

---

## Author

**Purva Suresh**

Customer Churn Prediction & Retention Analytics Platform (Personal Project)

Built using Python, Machine Learning, Data Analytics, and Streamlit.
