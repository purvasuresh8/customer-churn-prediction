from data_loader import load_data
from preprocessing import preprocess_data
from feature_engineering import engineer_features

df = load_data()

df = preprocess_data(df)

df = engineer_features(df)

df.to_csv(
    "data/processed/feature_engineered_data.csv",
    index=False
)

print(df.head())
print(df.shape)
