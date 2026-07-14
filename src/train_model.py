from pathlib import Path
from data_loader import load_data
from download_dataset import download_dataset

if not Path("data/raw/customer_churn.csv").exists():
    download_dataset()

df = load_data()
