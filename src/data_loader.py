import pandas as pd

def load_data():
    return pd.read_csv("data/raw/customer_churn.csv")
