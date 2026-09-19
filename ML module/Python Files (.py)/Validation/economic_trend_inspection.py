# ==========================================
# ECONOMIC TREND DATASET INSPECTION
# ==========================================

import os
import pandas as pd

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Train.csv"
)

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)

print("=" * 70)
print("ECONOMIC TREND DATASET INSPECTION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names and Data Types:")
print(df.dtypes.to_string())

print("\nMissing-Value Summary:")
missing_values = df.isnull().sum()
missing_values = missing_values[
    missing_values > 0
].sort_values(ascending=False)

if len(missing_values) == 0:
    print("No missing values found.")
else:
    print(missing_values.to_string())

print("\nUnique-Value Summary:")
print(df.nunique().sort_values().to_string())

print("\nFirst Five Rows:")
print(df.head())

print("\nNumerical Columns:")
print(
    df.select_dtypes(
        include=["number"]
    ).columns.tolist()
)

print("\nCategorical Columns:")
print(
    df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)
