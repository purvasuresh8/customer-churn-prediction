import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler



def clean_data(df):
    """
    Clean the raw customer data while preserving
    original categories for feature engineering.
    """

    # Work on a copy so the original DataFrame is unchanged.
    df = df.copy()

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Convert TotalCharges to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Customers with zero tenure and missing charges
    # are treated as having zero total charges.
    zero_tenure = df["tenure"].eq(0)
    missing_charges = df["TotalCharges"].isna()

    df.loc[
        zero_tenure & missing_charges,
        "TotalCharges"
    ] = 0

    # Fill any other missing TotalCharges with the median.
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df



def encode_target(df):
    """
    Convert Churn column to binary.
    """

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


def encode_categorical_features(df):
    """
    Label encode binary categorical columns.
    One-Hot Encode multi-class columns.
    """

    binary_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    le = LabelEncoder()

    for col in binary_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    multi_class_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    available_columns = [
        col for col in multi_class_columns
        if col in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=available_columns,
        drop_first=True
    )

    return df


def scale_features(df):
    """
    Scale numerical columns.
    """

    scaler = StandardScaler()

    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    existing_cols = [
        col for col in numeric_columns
        if col in df.columns
    ]

    df[existing_cols] = scaler.fit_transform(
        df[existing_cols]
    )

    return df


def preprocess_data(df):
    """
    Complete preprocessing pipeline.
    """

    # Remove identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = clean_data(df)
    df = encode_target(df)
    df = encode_categorical_features(df)
    df = scale_features(df)

    return df

from pathlib import Path
from data_loader import load_raw_data

if __name__ == "__main__":

    # Load dataset
    df = load_raw_data()

    # Run preprocessing pipeline
    df = preprocess_data(df)

    # Create output directory
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    output_file = "data/processed/cleaned_data.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("Preprocessing completed successfully!")
    print(f"Saved file: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    