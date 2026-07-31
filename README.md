# Customer Churn Prediction Platform

## Overview

Developed a machine learning-powered analytics platform that predicts customer churn and generates retention insights using customer behavioral and transactional data. The project focuses on identifying at-risk customers, evaluating predictive models, and supporting data-driven retention strategies.

---

## Business Problem

Customer churn directly impacts revenue and customer lifetime value. The objective of this project is to identify customers most likely to leave and provide actionable insights that can support retention efforts.

---

## Technologies Used

- Python
- SQL
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Key Features

- Customer churn prediction
- Customer risk scoring
- Customer segmentation
- Retention analytics
- Automated data preprocessing pipeline
- Model evaluation and comparison
- Data visualization and reporting

---

## Project Workflow

### 1. Data Collection

Imported customer demographic, behavioral, and transaction data.

### 2. Data Preprocessing

- Missing value handling
- Duplicate removal
- Data quality validation
- Categorical feature encoding
- Feature scaling

### 3. Feature Engineering

Created predictive features including:

- Customer tenure
- Purchase frequency
- Average transaction value
- Support interaction metrics
- Engagement indicators

### 4. Model Development

Implemented multiple machine learning models:

- Logistic Regression
- Random Forest Classifier

### 5. Model Evaluation

Compared model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 6. Retention Analytics

Generated customer risk segments and churn probability scores to support retention planning.

---

## Results

The platform successfully identifies customers at elevated churn risk and provides quantitative insights that can support customer retention strategies.

Key outputs include:

- Customer churn probability scores
- High-risk customer identification
- Customer segmentation analysis
- Retention-focused reporting

---

## Repository Structure

customer-churn-prediction/

├── data/

├── notebooks/

├── src/

│   ├── data_preprocessing.py

│   ├── feature_engineering.py

│   ├── model_training.py

│   └── model_evaluation.py

├── reports/

├── requirements.txt

└── README.md

---

## Future Improvements

- XGBoost implementation
- Hyperparameter tuning
- Model deployment using FastAPI
- Real-time churn prediction API
- Interactive analytics dashboard
