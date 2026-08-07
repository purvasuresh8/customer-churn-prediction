import pandas as pd
from feature_engineering import engineer_features


def main():

    df = pd.read_csv(
        "data/processed/cleaned_data.csv"
    )

    df = engineer_features(df)

    # Convert engineered categorical features
    df = pd.get_dummies(
        df,
        columns=[
            "TenureGroup",
            "ChargeSegment"
        ],
        drop_first=True
    )

    output_file = (
        "data/processed/feature_engineered_data.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("Feature engineering completed!")
    print(f"Saved file: {output_file}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
    