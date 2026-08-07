# Customer Churn Prediction & Retention Analytics Platform

## Overview

An end-to-end machine learning and analytics platform designed to predict customer churn, identify key retention drivers, classify customer risk, and generate actionable retention recommendations.

The project uses the IBM Telco Customer Churn dataset and follows a complete machine learning workflow from data ingestion through preprocessing, feature engineering, model training, explainability, business recommendations, and dashboard visualization.

---

## Features

- Automated customer churn prediction
- Data validation and quality checks
- Data preprocessing and cleaning
- Feature engineering pipeline
- Logistic Regression and Random Forest modeling
- Model evaluation and comparison
- SHAP-based model explainability
- Churn driver analysis
- Customer risk segmentation
- Retention recommendation engine
- Interactive Streamlit dashboard
- Model persistence and deployment-ready architecture

---

## Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- SHAP (SHapley Additive Explanations)

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
SHAP Explainability
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
├── logistic_regression.pkl
└── random_forest.pkl

reports/
├── top_churn_drivers.csv
└── ...

visualizations/
├── logistic_regression_confusion_matrix.png
├── logistic_regression_roc_curve.png
├── random_forest_confusion_matrix.png
├── random_forest_feature_importance.png
├── random_forest_roc_curve.png
└── shap/
    ├── shap_summary.png
    ├── shap_feature_importance.png
    └── customer_0_waterfall.png

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

- Customer demographics
- Service subscriptions
- Contract information
- Billing information
- Monthly charges
- Total charges
- Customer churn status

Dataset Size:

```text
7,043 Customer Records
21 Original Features
```

---

## Data Validation

Implemented automated validation checks for:

- Missing values
- Duplicate records
- Invalid target values
- Invalid charge values
- Empty strings
- Customer identifier validation

This layer ensures data quality before model training.

---

## Data Preprocessing

Implemented:

- Duplicate removal
- Missing value handling
- Data type standardization
- Label encoding
- One-hot encoding
- Numerical feature scaling

Output:

```text
data/processed/cleaned_data.csv
```

---

## Feature Engineering

Created additional business-focused features:

- Tenure Group Classification
- Charge Segment Classification
- Average Monthly Spend
- Customer Value Indicators
- Service Count Metrics
- Customer Engagement Score

Output:

```text
data/processed/feature_engineered_data.csv
```

Final Dataset:

```text
7021 Records
40 Features
```

---

## Machine Learning Models

### Logistic Regression

Used as an interpretable baseline machine learning model.

### Random Forest

Used to capture more complex customer churn relationships and provide feature importance analysis.

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

Logistic Regression achieved the strongest overall results and outperformed Random Forest on:

- Accuracy
- Recall
- F1 Score
- ROC-AUC

---

## Model Explainability

To improve model transparency and business interpretability, SHAP (SHapley Additive Explanations) was used to explain churn predictions and identify the most influential customer churn drivers.

Generated Explainability Artifacts:

- SHAP Summary Plot
- SHAP Feature Importance Plot
- Customer-Level Waterfall Explanation
- Top Churn Drivers Report

Outputs:

```text
visualizations/shap/
├── shap_summary.png
├── shap_feature_importance.png
└── customer_0_waterfall.png

reports/
└── top_churn_drivers.csv
```

---

## Top Churn Drivers

SHAP analysis identified the most influential factors contributing to customer churn:

1. EngagementScore
2. InternetService_Fiber optic
3. Contract_Two year
4. TotalCharges
5. Contract_One year
6. tenure
7. PaperlessBilling
8. MultipleLines_Yes
9. PaymentMethod_Electronic check
10. StreamingMovies_Yes

These insights help prioritize customer retention strategies and support data-driven business decision-making.

---

## Churn Prediction Pipeline

The prediction engine:

- Loads trained models
- Generates churn predictions
- Calculates churn probabilities
- Assigns customer risk levels
- Produces recommendation-ready outputs

Output:

```text
data/predictions/churn_predictions.csv
```

Example:

```text
Prediction | Probability | Risk Level

1          | 0.63        | Medium
0          | 0.04        | Low
1          | 0.68        | Medium
```

---

## Retention Recommendation Engine

Customers are categorized into risk groups and assigned business actions.

### High Risk

- Immediate customer outreach
- Retention discounts
- Dedicated support intervention

### Medium Risk

- Engagement campaigns
- Loyalty incentives
- Proactive communication

### Low Risk

- Monitoring
- Regular marketing programs
- Periodic promotions

Output:

```text
data/predictions/retention_plan.csv
```

Example:

```text
Probability | Risk | Recommendation

0.88        | High   | Immediate retention outreach
0.63        | Medium | Engagement campaign
0.04        | Low    | Maintain engagement
```

---

## Dashboard

Built using Streamlit.

Dashboard Features:

- Customer Risk Overview
- KPI Metrics
- Risk Distribution Analysis
- Churn Probability Analysis
- Priority Score Visualization
- Retention Recommendation Analytics
- Customer Filtering
- Exportable Results

Launch Command:

```bash
streamlit run src/dashboard.py
```

---

# Challenges Encountered & Debugging Journey

This project was developed from scratch and required solving multiple real-world software engineering and machine learning challenges.

---

## 1. Dataset Parsing Failure

### Problem

The dataset file accidentally contained Python code instead of CSV content.

### Error

```text
ParserError:
Expected 1 fields in line 6, saw 2
```

### Solution

- Removed corrupted file
- Implemented KaggleHub integration
- Automated data loading

### Outcome

Established a reproducible data ingestion workflow.

---

## 2. Dataset Management Issues

### Problem

Manual dataset placement created repeated loading failures.

### Solution

Implemented automated retrieval using KaggleHub.

### Outcome

Removed manual dataset dependency.

---

## 3. Data Loader Import Issues

### Problem

Some scripts expected:

```python
load_data()
```

while the actual implementation provided:

```python
load_raw_data()
```

### Error

```text
ImportError:
cannot import name 'load_data'
```

### Solution

Standardized all loading functions.

---

## 4. Non-Executing Pipelines

### Problem

Preprocessing and feature engineering scripts contained only function definitions.

### Outcome

Scripts executed but produced no files.

### Solution

Added executable entry points:

```python
if __name__ == "__main__":
```

### Result

Generated:

```text
cleaned_data.csv
feature_engineered_data.csv
```

---

## 5. Configuration Import Errors

### Error

```text
ModuleNotFoundError:
No module named 'config'
```

### Solution

Refactored imports and project organization.

### Outcome

Centralized configuration-driven architecture.

---

## 6. Evaluation Pipeline Failure

### Error

```text
ValueError:
The truth value of a Index is ambiguous
```

### Solution

Converted Pandas Index objects into Python lists and improved evaluation logic.

### Outcome

Successful automated model evaluation.

---

## 7. Missing Directory Structure

### Error

```text
FileNotFoundError:
models/logistic_regression.pkl
```

### Solution

Created required output directories for:

```text
models/
reports/
visualizations/
logs/
```

### Outcome

Successful model persistence.

---

## 8. Prediction Pipeline Errors

### Error

```text
NameError:
load_model is not defined
```

### Solution

Standardized model loading using Joblib.

### Outcome

Generated churn prediction outputs successfully.

---

## 9. Dashboard Syntax Errors

### Error

```text
SyntaxError:
invalid syntax
```

### Root Cause

Streamlit source code contained HTML tags:

```html
<br>
```

### Solution

Removed HTML artifacts and validated dashboard code.

### Outcome

Successfully launched Streamlit dashboard.

---

## 10. SHAP Explainability Pipeline Issues

### Error

```text
AttributeError:
'float' object has no attribute 'rint'
```

### Root Cause

Mixed boolean and numeric feature data types caused SHAP compatibility issues.

### Solution

Converted all model features to numeric float values before explanation generation.

### Outcome

Successfully generated:

```text
shap_summary.png
shap_feature_importance.png
customer_0_waterfall.png
top_churn_drivers.csv
```

---

## Lessons Learned

This project provided hands-on experience in:

- Data Engineering
- ETL Pipeline Development
- Data Quality Validation
- Feature Engineering
- Machine Learning
- Model Explainability
- Business Analytics
- Streamlit Development
- Dependency Management
- Debugging Complex Python Applications
- End-to-End Analytics Solution Design

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

Logistic Regression achieved the strongest overall performance, reaching approximately 80% accuracy and 0.84 ROC-AUC while supporting actionable customer retention recommendations through a fully interactive analytics dashboard.

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute Pipeline

```bash
python src/preprocessing.py

python src/run_feature_engineering.py

python src/train_model.py

python src/model_explainability.py

python src/predict.py

python src/retention_recommendations.py
```

### Launch Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Future Enhancements

- SQL Analytics Layer
- XGBoost Modeling
- Hyperparameter Optimization
- PDF Executive Reporting
- Automated Report Scheduling
- Docker Containerization
- GitHub Actions CI/CD
- Cloud Deployment (AWS / Azure)

---

## Author

**Purva Suresh**

Customer Churn Prediction & Retention Analytics Platform

Built using Python, Machine Learning, Explainable AI, Data Analytics, and Streamlit.
