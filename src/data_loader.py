import pandas as pd
from pathlib import Path

from config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    FEATURE_ENGINEERED_FILE
)


def load_csv(file_path):
    """
    Load a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


def load_raw_data():
    """
    Load raw customer churn dataset.
    """

    return load_csv(RAW_DATA_FILE)


def load_cleaned_data():
    """
    Load cleaned dataset.
    """

    return load_csv(CLEANED_DATA_FILE)


def load_feature_engineered_data():
    """
    Load feature engineered dataset.
    """

    return load_csv(FEATURE_ENGINEERED_FILE)


def get_dataset_info(df):
    """
    Print dataset summary.
    """

    print("\nDataset Information")
    print("-" * 50)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict()
    }


if __name__ == "__main__":

    df = load_raw_data()

    get_dataset_info(df)

    print("\nFirst 5 Records:")
    print(df.head())
