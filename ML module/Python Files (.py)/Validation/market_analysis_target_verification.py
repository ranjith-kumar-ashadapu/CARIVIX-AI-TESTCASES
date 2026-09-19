# ==========================================
# MARKET ANALYSIS TARGET VERIFICATION
# ==========================================

import os
import pandas as pd

# ==========================================
# FILE PATH
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Market_Analysis_Data_Splits"
    r"\Market_Analysis_Train.csv"
)

# ==========================================
# TARGET VARIABLE
# ==========================================

target_column = "2025 [YR2025]"

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)

print("=" * 70)
print("MARKET ANALYSIS TARGET VERIFICATION")
print("=" * 70)

# ==========================================
# CHECK TARGET COLUMN
# ==========================================

if target_column not in df.columns:
    raise ValueError(
        f"Target column was not found: {target_column}"
    )

print("\nSelected Target Column:")
print(target_column)

print("\nTarget Data Type:")
print(df[target_column].dtype)

# ==========================================
# TARGET SUMMARY
# ==========================================

print("\nTarget Summary:")
print(df[target_column].describe())

# ==========================================
# MISSING TARGET VALUES
# ==========================================

missing_target_count = df[target_column].isnull().sum()
total_records = len(df)
available_target_count = total_records - missing_target_count

print("\nTarget Availability:")
print("Total Records:", total_records)
print("Available Target Values:", available_target_count)
print("Missing Target Values:", missing_target_count)
print(
    "Missing Target Percentage:",
    f"{missing_target_count / total_records * 100:.2f}%"
)

# ==========================================
# TARGET VALUE PREVIEW
# ==========================================

print("\nFirst Ten Target Values:")
print(df[target_column].head(10).to_string(index=False))

# ==========================================
# IDENTIFY POSSIBLE LEAKAGE FEATURES
# ==========================================

leakage_features = [
    "2025 [YR2025]",
    "Latest_Year_Value",
    "Latest_Value_Log",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Absolute_Change",
    "Percentage_Change",
    "CAGR_Percentage",
    "Trend_Slope",
    "Trend_Direction",
    "Latest_Value_Status",
    "Historical_Average",
    "Historical_Median",
    "Historical_Minimum",
    "Historical_Maximum",
    "Historical_Standard_Deviation",
    "Historical_Range",
    "Average_Value_Log",
]

existing_leakage_features = [
    column for column in leakage_features
    if column in df.columns
]

print("\nFeatures Excluded to Prevent Data Leakage:")
for column in existing_leakage_features:
    print("-", column)

# ==========================================
# HISTORICAL FEATURES
# ==========================================

historical_year_features = [
    column for column in df.columns
    if "[YR" in column and column != target_column
]

print("\nHistorical Year Features Available:")
print("Number of Historical Features:", len(historical_year_features))

for column in historical_year_features:
    print("-", column)

# ==========================================
# POTENTIAL INPUT FEATURES
# ==========================================

excluded_columns = set(existing_leakage_features)

# The target is also excluded automatically.
excluded_columns.add(target_column)

candidate_features = [
    column for column in df.columns
    if column not in excluded_columns
]

print("\nCandidate Input Features:")
print("Number of Candidate Features:", len(candidate_features))

for column in candidate_features:
    print("-", column)

# ==========================================
# FINAL DECISION
# ==========================================

print("\n" + "=" * 70)
print("TARGET VERIFICATION COMPLETED")
print("=" * 70)

print("Target Variable:", target_column)
print("Problem Type: Regression")
print("Prediction Objective:")
print(
    "Predict the 2025 economic indicator value using historical "
    "and non-leaking features."
)
print("=" * 70)
