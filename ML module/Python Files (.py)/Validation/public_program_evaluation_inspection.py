# ==========================================
# PUBLIC PROGRAM EVALUATION DATASET INSPECTION
# ==========================================

import os
import pandas as pd

# ==========================================
# FILE PATH
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Train.csv"
)

# ==========================================
# CHECK FILE
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(train_file)

print("=" * 70)
print("PUBLIC PROGRAM EVALUATION DATASET INSPECTION")
print("=" * 70)

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes.to_string())

# Missing values
print("\nMissing-Value Summary:")
missing_values = df.isnull().sum()
missing_values = missing_values[
    missing_values > 0
].sort_values(ascending=False)

if len(missing_values) == 0:
    print("No missing values found.")
else:
    print(missing_values.to_string())

# Unique values
print("\nUnique-Value Summary:")
print(df.nunique().sort_values().to_string())

# First five rows
print("\nFirst Five Rows:")
print(df.head())

# Numerical columns
print("\nNumerical Columns:")
print(
    df.select_dtypes(
        include=["number"]
    ).columns.tolist()
)

# Categorical columns
print("\nCategorical Columns:")
print(
    df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)
