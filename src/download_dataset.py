# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
from config import RAW_DATA_FILE

df.to_csv(
    RAW_DATA_FILE,
    index=False
)
# Create data directory
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Load dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# Save locally
output_path = "data/raw/customer_churn.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print("\nFirst 5 records:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
