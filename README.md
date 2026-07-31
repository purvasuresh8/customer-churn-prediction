# Customer Churn Prediction Platform

## Overview

Customer churn can significantly impact business revenue and customer lifetime value. This project uses machine learning to identify customers at risk of churning and generate actionable retention insights from customer behavioral and transactional data.

The platform supports the complete machine learning workflow, including data validation, preprocessing, feature engineering, model training, evaluation, explainability, risk scoring, reporting, and retention recommendations.

---

## Business Objective

The goal of this project is to:

- Predict customer churn risk
- Identify high-risk customer segments
- Generate retention-focused insights
- Support data-driven decision-making
- Improve customer retention strategies

---

## Technologies Used

### Programming & Analytics
- Python
- SQL
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Data Visualization
- Matplotlib
- Seaborn

---

## Key Features

### Customer Churn Prediction
Developed predictive machine learning models to identify customers with elevated churn risk.

### Data Processing Pipeline
Built automated workflows for:

- Data ingestion
- Data validation
- Data preprocessing
- Feature engineering

### Model Evaluation
Implemented model comparison and performance evaluation using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Risk Scoring
Generated churn probability scores to prioritize customer retention efforts.

### Explainability & Reporting
Produced model explainability outputs and business-focused reports to translate analytical findings into actionable insights.

### Retention Recommendations
Generated customer retention recommendations based on predicted churn risk levels.

---

## Machine Learning Workflow

### 1. Data Collection & Validation

The pipeline validates incoming customer data to ensure:

- Data completeness
- Data consistency
- Data quality
- Schema integrity

Relevant modules:

- `data_loader.py`
- `data_validation.py`

---

### 2. Data Preprocessing

Preprocessing includes:

- Missing value handling
- Categorical encoding
- Feature scaling
- Data transformation

Relevant modules:

- `preprocessing.py`

---

### 3. Feature Engineering

Created predictive features using customer behavioral and transactional information.

Examples include:

- Customer tenure
- Purchase behavior
- Engagement metrics
- Account activity indicators

Relevant modules:

- `feature_engineering.py`
- `run_feature_engineering.py`

---

### 4. Model Development

Developed and evaluated machine learning models for customer churn prediction.

Models include:

- Logistic Regression
- Random Forest Classifier

Relevant modules:

- `train_model.py`

---

### 5. Model Evaluation

Compared model performance using multiple evaluation metrics and visualizations.

Relevant modules:

- `evaluate_model.py`

---

### 6. Explainability & Insights

Generated interpretable outputs to better understand model behavior and key churn drivers.

Relevant modules:

- `model_explainability.py`

---

### 7. Reporting & Recommendations

Generated business-focused outputs including:

- Churn risk reports
- Customer segmentation insights
- Retention recommendations

Relevant modules:

- `generate_report.py`
- `retention_recommendations.py`
- `dashboard.py`

---

## Repository Structure

```text
customer-churn-prediction
│
├── screenshots
│   ├── confusion_matrix.png
│   ├── customer_churn_distribution.png
│   ├── feature_importance_plot.png
│   └── model_comparison_chart.png
│
├── src
│   ├── dashboard.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── download_dataset.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── generate_report.py
│   ├── model_explainability.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── retention_recommendations.py
│   ├── run_feature_engineering.py
│   ├── train_model.py
│   └── util.py
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Visualizations

### Customer Churn Distribution

screenshots/customer_churn_distribution.png

### Confusion Matrix

screenshots/confusion_matrix.png

### Feature Importance

screenshots/feature_importance_plot.png

### Model Comparison

screenshots/model_comparison_chart.png

---

## Results

The platform successfully:

- Predicts customer churn risk
- Scores customers based on likelihood to churn
- Supports customer segmentation analysis
- Generates retention-focused recommendations
- Provides visual analytics for model evaluation and business reporting

---

## Future Improvements

Potential enhancements include:

- Hyperparameter optimization
- Additional ensemble learning models
- XGBoost implementation
- Real-time prediction API using FastAPI
- Cloud deployment
- Interactive web dashboard
- Automated model monitoring

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run feature engineering:

```bash
python src/run_feature_engineering.py
```

Train model:

```bash
python src/train_model.py
```

Evaluate model:

```bash
python src/evaluate_model.py
```

Generate predictions:

```bash
python src/predict.py
```

Generate report:

```bash
python src/generate_report.py
```

Launch dashboard:

```bash
python src/dashboard.py
```
