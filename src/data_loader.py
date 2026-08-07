import pandas as pd
from pathlib import Path
import kagglehub


def load_raw_data():
    path = kagglehub.dataset_download(
        "blastchar/telco-customer-churn"
    )

    print("Dataset location:", path)

    csv_file = (
        Path(path)
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(csv_file)

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
