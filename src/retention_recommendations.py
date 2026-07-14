import pandas as pd


def classify_risk(churn_probability):
    """
    Categorize customers by churn risk.
    """

    if churn_probability >= 0.80:
        return "High"

    elif churn_probability >= 0.50:
        return "Medium"

    return "Low"


def generate_recommendation(customer):
    """
    Generate personalized retention recommendation.
    """

    probability = customer["ChurnProbability"]
    tenure = customer.get("tenure", 0)
    monthly_charges = customer.get("MonthlyCharges", 0)
    contract = customer.get("Contract", "")

    # High risk customers
    if probability >= 0.80:

        if contract == "Month-to-month":
            return (
                "Offer contract upgrade discount and "
                "assign customer success representative"
            )

        if monthly_charges > 80:
            return (
                "Provide premium service discount "
                "or bundled package offer"
            )

        return (
            "Immediate outreach with retention incentive"
        )

    # Medium risk customers
    elif probability >= 0.50:

        if tenure < 12:
            return (
                "Enroll in onboarding and engagement campaign"
            )

        return (
            "Send loyalty rewards and personalized offers"
        )

    # Low risk customers
    return (
        "Maintain engagement through periodic promotions"
    )


def calculate_priority_score(churn_probability):
    """
    Convert churn probability into business priority.
    """

    return round(churn_probability * 100)


def generate_retention_plan(df):
    """
    Generate retention actions for all customers.
    """

    result = df.copy()

    result["RiskLevel"] = result[
        "ChurnProbability"
    ].apply(classify_risk)

    result["PriorityScore"] = result[
        "ChurnProbability"
    ].apply(calculate_priority_score)

    result["Recommendation"] = result.apply(
        generate_recommendation,
        axis=1
    )

    return result


def save_retention_plan(
    df,
    output_path="data/predictions/retention_plan.csv"
):
    """
    Save retention plan to CSV.
    """

    retention_df = generate_retention_plan(df)

    retention_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Retention plan saved to {output_path}"
    )

    return retention_df


if __name__ == "__main__":

    predictions_df = pd.read_csv(
        "data/predictions/churn_predictions.csv"
    )

    retention_plan = save_retention_plan(
        predictions_df
    )

    print(
        retention_plan[
            [
                "ChurnProbability",
                "RiskLevel",
                "PriorityScore",
                "Recommendation"
            ]
        ].head()
    )
