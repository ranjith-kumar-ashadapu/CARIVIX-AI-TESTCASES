# ==========================================
# ECONOMIC TREND TARGET VERIFICATION
# ==========================================

import os
import pandas as pd

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Train.csv"
)

target_column = "2025"

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)

print("=" * 70)
print("ECONOMIC TREND TARGET VERIFICATION")
print("=" * 70)

if target_column not in df.columns:
    raise ValueError(
        f"Target column was not found: {target_column}"
    )

print("\nSelected Target Column:")
print(target_column)

print("\nTarget Data Type:")
print(df[target_column].dtype)

print("\nTarget Summary:")
print(df[target_column].describe())

missing_target_count = df[target_column].isnull().sum()
total_records = len(df)

print("\nTarget Availability:")
print("Total Records:", total_records)
print(
    "Available Target Values:",
    total_records - missing_target_count
)
print(
    "Missing Target Values:",
    missing_target_count
)
print(
    "Missing Target Percentage:",
    f"{missing_target_count / total_records * 100:.2f}%"
)

print("\nFirst Ten Target Values:")
print(
    df[target_column]
    .head(10)
    .to_string(index=False)
)

leakage_features = [
    "2025",
    "Latest_Year_Value",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Latest_Value_Log",
    "Latest_Value_Outlier_Flag",
    "Historical_Average",
    "Historical_Median",
    "Historical_Minimum",
    "Historical_Maximum",
    "Historical_Standard_Deviation",
    "Historical_Range",
    "Percentage_Change",
    "Absolute_Change",
    "CAGR_Percentage",
    "Trend_Slope",
    "Latest_Value_Status",
    "Trend_Direction",
]

existing_leakage_features = [
    column
    for column in leakage_features
    if column in df.columns
]

print("\nFeatures to Exclude to Prevent Target Leakage:")
for column in existing_leakage_features:
    print("-", column)

print("\nHistorical Year Features Available as Inputs:")
historical_years = [
    str(year)
    for year in range(1981, 2025)
    if str(year) in df.columns
]

for column in historical_years:
    print("-", column)

print("\n" + "=" * 70)
print("TARGET VERIFICATION COMPLETED")
print("=" * 70)

print("Target Variable:", target_column)
print("Problem Type: Regression")
print(
    "Objective: Predict the 2025 economic value "
    "using historical values and non-leaking metadata."
)
print("=" * 70)
