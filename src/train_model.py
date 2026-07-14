from pathlib import Path
from data_loader import load_data
from download_dataset import download_dataset
from preprocessing import preprocess_data
from evaluate_model import evaluate_and_visualize


if not Path("data/raw/customer_churn.csv").exists():
    download_dataset()

df = load_data()

df = preprocess_data(df)

evaluate_and_visualize(
    logistic_model,
    X_test,
    y_test,
    "Logistic Regression",
    X.columns
)

evaluate_and_visualize(
    random_forest_model,
    X_test,
    y_test,
    "Random Forest",
    X.columns
)

print(df.head())
print(df.shape)
import joblib

joblib.dump(
    random_forest_model,
    "models/random_forest.pkl"
)

joblib.dump(
    logistic_model,
    "models/logistic_regression.pkl"
)

from config import (
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_FOREST_PARAMS,
    TEST_SIZE
)

rf_model = RandomForestClassifier(
    **RANDOM_FOREST_PARAMS
)
