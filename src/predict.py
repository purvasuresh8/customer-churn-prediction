import pandas as pd
import joblib
from pathlib import Path


def assign_risk(prob):

    if prob >= 0.80:
        return "High"

    if prob >= 0.50:
        return "Medium"

    return "Low"


def main():

    model = joblib.load(
        "models/logistic_regression.pkl"
    )

    df = pd.read_csv(
        "data/processed/feature_engineered_data.csv"
    )

    X = df.drop(
        "Churn",
        axis=1
    )

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "Prediction": predictions,
        "ChurnProbability": probabilities
    })

    results["RiskLevel"] = results[
        "ChurnProbability"
    ].apply(assign_risk)

    Path(
        "data/predictions"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        "data/predictions/churn_predictions.csv",
        index=False
    )

    print("Predictions saved successfully.")
    print(results.head())


if __name__ == "__main__":
    main()
