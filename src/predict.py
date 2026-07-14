import os
import joblib
import pandas as pd


def load_model(model_path):
    """
    Load trained model.
    """

    return joblib.load(model_path)


def assign_risk_level(probability):
    """
    Assign customer risk category.
    """

    if probability >= 0.80:
        return "High"

    elif probability >= 0.50:
        return "Medium"

    return "Low"


def generate_recommendation(probability):
    """
    Generate retention strategy.
    """

    if probability >= 0.80:
        return "Offer retention discount and proactive outreach"

    elif probability >= 0.50:
        return "Enroll in loyalty campaign"

    return "Monitor customer activity"


def predict_churn(
    model_path,
    input_file,
    output_file
):
    """
    Predict customer churn.
    """

    model = load_model(model_path)

    df = pd.read_csv(input_file)

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)[:, 1]

    results = df.copy()

    results["ChurnPrediction"] = predictions

    results["ChurnProbability"] = probabilities

    results["RiskLevel"] = results[
        "ChurnProbability"
    ].apply(assign_risk_level)

    results["Recommendation"] = results[
        "ChurnProbability"
    ].apply(generate_recommendation)

    os.makedirs(
        "data/predictions",
        exist_ok=True
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"Predictions saved to {output_file}"
    )

    return results


if __name__ == "__main__":

    MODEL_PATH = (
        "models/random_forest.pkl"
    )

    INPUT_FILE = (
        "data/processed/new_customers.csv"
    )

    OUTPUT_FILE = (
        "data/predictions/churn_predictions.csv"
    )

    predictions = predict_churn(
        MODEL_PATH,
        INPUT_FILE,
        OUTPUT_FILE
    )

    print(
        predictions[
            [
                "ChurnPrediction",
                "ChurnProbability",
                "RiskLevel",
                "Recommendation"
            ]
        ].head()
    )
