from pathlib import Path
from data_loader import load_data
from download_dataset import download_dataset
from preprocessing import preprocess_data

if not Path("data/raw/customer_churn.csv").exists():
    download_dataset()

df = load_data()

df = preprocess_data(df)

print(df.head())
print(df.shape)
