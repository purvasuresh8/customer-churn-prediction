import sqlite3
from pathlib import Path

import pandas as pd


DATABASE_FILE = "database/churn.db"

DATA_FILE = (
    "data/processed/"
    "feature_engineered_data.csv"
)


def create_database():

    Path("database").mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DATABASE_FILE
    )

    return conn


def load_data_to_database():

    conn = create_database()

    df = pd.read_csv(
        DATA_FILE
    )

    df.to_sql(
        "customers",
        conn,
        if_exists="replace",
        index=False
    )

    conn.commit()

    print(
        "✅ Customers table created."
    )

    return conn


def analyze_churn_distribution(conn):

    query = """
    SELECT
        Churn,
        COUNT(*) AS CustomerCount
    FROM customers
    GROUP BY Churn
    """

    result = pd.read_sql_query(
        query,
        conn
    )

    result.to_csv(
        "reports/churn_distribution_sql.csv",
        index=False
    )

    print("\nChurn Distribution")
    print(result)

    return result


def analyze_high_value_customers(conn):

    query = """
    SELECT
        HighValueCustomer,
        ROUND(
            AVG(Churn) * 100,
            2
        ) AS ChurnRate
    FROM customers
    GROUP BY HighValueCustomer
    """

    result = pd.read_sql_query(
        query,
        conn
    )

    result.to_csv(
        "reports/high_value_customer_sql.csv",
        index=False
    )

    print(
        "\nHigh Value Customer Analysis"
    )
    print(result)

    return result


def analyze_engagement_score(conn):

    query = """
    SELECT
        CASE

            WHEN EngagementScore < -0.5
            THEN 'Low'

            WHEN EngagementScore < 0.5
            THEN 'Medium'

            ELSE 'High'

        END AS EngagementGroup,

        COUNT(*) AS Customers,

        ROUND(
            AVG(Churn) * 100,
            2
        ) AS ChurnRate

    FROM customers

    GROUP BY EngagementGroup
    """

    result = pd.read_sql_query(
        query,
        conn
    )

    result.to_csv(
        "reports/engagement_analysis_sql.csv",
        index=False
    )

    print(
        "\nEngagement Analysis"
    )
    print(result)

    return result


def main():

    Path("reports").mkdir(
        parents=True,
        exist_ok=True
    )

    conn = load_data_to_database()

    analyze_churn_distribution(
        conn
    )

    analyze_high_value_customers(
        conn
    )

    analyze_engagement_score(
        conn
    )

    conn.close()

    print(
        "\n✅ SQL analytics completed."
    )


if __name__ == "__main__":
    main()
    