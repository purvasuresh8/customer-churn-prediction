import os
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime


def create_directory(directory_path):
    """
    Create directory if it does not exist.
    """

    Path(directory_path).mkdir(
        parents=True,
        exist_ok=True
    )


def save_csv(df, file_path):
    """
    Save DataFrame to CSV.
    """

    create_directory(
        Path(file_path).parent
    )

    df.to_csv(
        file_path,
        index=False
    )

    print(
        f"CSV saved successfully: {file_path}"
    )


def load_csv(file_path):
    """
    Load CSV file.
    """

    return pd.read_csv(file_path)


def save_model(model, file_path):
    """
    Save trained model.
    """

    create_directory(
        Path(file_path).parent
    )

    joblib.dump(
        model,
        file_path
    )

    print(
        f"Model saved successfully: {file_path}"
    )


def load_model(file_path):
    """
    Load trained model.
    """

    return joblib.load(file_path)


def save_metrics(metrics, file_path):
    """
    Save evaluation metrics.
    """

    create_directory(
        Path(file_path).parent
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        file_path,
        index=False
    )

    print(
        f"Metrics saved successfully: {file_path}"
    )


def get_timestamp():
    """
    Return current timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def log_message(
    message,
    log_file="logs/pipeline.log"
):
    """
    Write message to log file.
    """

    create_directory(
        Path(log_file).parent
    )

    timestamp = get_timestamp()

    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"[{timestamp}] {message}\n"
        )

    print(
        f"[{timestamp}] {message}"
    )


def save_dataframe_summary(
    df,
    output_file
):
    """
    Save basic dataset summary.
    """

    summary = pd.DataFrame({
        "Column": df.columns,
        "DataType": df.dtypes.astype(str),
        "MissingValues": df.isnull().sum().values
    })

    save_csv(
        summary,
        output_file
    )


def print_section(title):
    """
    Print formatted console section.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def calculate_churn_rate(df, target_column="Churn"):
    """
    Calculate churn rate percentage.
    """

    churn_rate = (
        df[target_column].mean() * 100
    )

    return round(churn_rate, 2)


def ensure_project_directories(
        directories):
    """
    Create multiple directories.
    """

    for directory in directories:
        create_directory(directory)

    print(
        "All required directories created."
    )
