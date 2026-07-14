import pandas as pd
import numpy as np


def create_tenure_groups(df):
    """
    Group customers by tenure.
    """

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[
            "New",
            "Developing",
            "Established",
            "Loyal"
        ]
    )

    return df


def create_monthly_charge_segments(df):
    """
    Categorize customers based on monthly charges.
    """

    df["ChargeSegment"] = pd.qcut(
        df["MonthlyCharges"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    return df


def create_avg_monthly_spend(df):
    """
    Approximate average spend per month.
    """

    df["AvgMonthlySpend"] = (
        df["TotalCharges"] /
        (df["tenure"] + 1)
    )

    return df


def create_customer_value_flag(df):
    """
    Identify high-value customers.
    """

    median_charge = df["MonthlyCharges"].median()

    df["HighValueCustomer"] = np.where(
        df["MonthlyCharges"] > median_charge,
        1,
        0
    )

    return df


def create_service_count(df):
    """
    Count subscribed services.
    """

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    available_columns = [
        col for col in service_columns
        if col in df.columns
    ]

    df["ServiceCount"] = 0

    for col in available_columns:
        df["ServiceCount"] += (
            df[col]
            .astype(str)
            .str.contains("Yes")
            .astype(int)
        )

    return df


def create_contract_risk_score(df):
    """
    Assign risk based on contract type.
    """

    contract_scores = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    }

    if "Contract" in df.columns:
        df["ContractRiskScore"] = (
            df["Contract"]
            .map(contract_scores)
        )

    return df


def create_engagement_score(df):
    """
    Composite engagement score.
    """

    if "ServiceCount" not in df.columns:
        df = create_service_count(df)

    df["EngagementScore"] = (
        df["ServiceCount"] * 0.6
        + (df["tenure"] / df["tenure"].max()) * 10 * 0.4
    )

    return df


def engineer_features(df):
    """
    Master feature engineering pipeline.
    """

    df = create_tenure_groups(df)

    df = create_monthly_charge_segments(df)

    df = create_avg_monthly_spend(df)

    df = create_customer_value_flag(df)

    df = create_service_count(df)

    df = create_contract_risk_score(df)

    df = create_engagement_score(df)

    return df
