import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    RANDOM_FOREST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS
)

from evaluate_model import evaluate_and_visualize


def main():

    # Load feature engineered dataset
    df = pd.read_csv(
        "data/processed/feature_engineered_data.csv"
    )

    print("Dataset Shape:", df.shape)

    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Logistic Regression
    logistic_model = LogisticRegression(
        **LOGISTIC_REGRESSION_PARAMS
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    # Random Forest
    random_forest_model = RandomForestClassifier(
        **RANDOM_FOREST_PARAMS
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    # Evaluate models
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

    # Save models
    joblib.dump(
        logistic_model,
        "models/logistic_regression.pkl"
    )

    joblib.dump(
        random_forest_model,
        "models/random_forest.pkl"
    )

    print("\nTraining Complete!")
    print("Models saved successfully.")


if __name__ == "__main__":
    main()
    