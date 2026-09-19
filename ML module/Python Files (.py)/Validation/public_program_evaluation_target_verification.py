# ==========================================
# PUBLIC PROGRAM EVALUATION TARGET VERIFICATION
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
# TARGET COLUMN
# ==========================================

target_column = "Total No. of Workers"

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)

print("=" * 70)
print("PUBLIC PROGRAM EVALUATION TARGET VERIFICATION")
print("=" * 70)

# ==========================================
# VERIFY TARGET
# ==========================================

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

# ==========================================
# TARGET AVAILABILITY
# ==========================================

missing_target_count = df[target_column].isnull().sum()
total_records = len(df)
available_target_count = (
    total_records - missing_target_count
)

print("\nTarget Availability:")
print("Total Records:", total_records)
print(
    "Available Target Values:",
    available_target_count
)
print(
    "Missing Target Values:",
    missing_target_count
)
print(
    "Missing Target Percentage:",
    f"{missing_target_count / total_records * 100:.2f}%"
)

# ==========================================
# TARGET PREVIEW
# ==========================================

print("\nFirst Ten Target Values:")
print(
    df[target_column]
    .head(10)
    .to_string(index=False)
)

# ==========================================
# DIRECT AND DERIVED LEAKAGE FEATURES
# ==========================================

worker_leakage_features = [
    "Total No. of Workers",
    "Workers_Log",
    "Workers_Missing_Flag",
    "State_Total_Workers",
    "State_Average_Workers",
    "Worker_Share_of_State",
    "Workers_Per_Expenditure_Lakh",
    "Worker_Wage_Ratio",
    "Worker_Category",
    "Workers_Outlier_Flag",
    "Workers_vs_State_Median",
]

existing_worker_leakage_features = [
    column
    for column in worker_leakage_features
    if column in df.columns
]

print("\nWorker-Related Leakage Features to Exclude:")
for column in existing_worker_leakage_features:
    print("-", column)

# ==========================================
# RELATED FEATURES REQUIRING REVIEW
# ==========================================

related_features = [
    "Total No. of Active Workers",
    "Total No. of Active Job Cards",
    "Total No. of JobCards issued",
    "Total Households Worked",
    "Total Individuals Worked",
    "Persondays of Central Liability so far",
    "Women Persondays",
]

existing_related_features = [
    column
    for column in related_features
    if column in df.columns
]

print("\nRelated Employment Features Requiring Review:")
for column in existing_related_features:
    print("-", column)

# ==========================================
# CORRELATION WITH TARGET
# ==========================================

numeric_df = df.select_dtypes(
    include=["number"]
)

correlations = (
    numeric_df.corr(numeric_only=True)[target_column]
    .drop(target_column)
    .abs()
    .sort_values(ascending=False)
)

print("\nStrongest Numeric Relationships with Target:")
print(correlations.head(15).to_string())

# ==========================================
# CANDIDATE INPUT FEATURES
# ==========================================

excluded_columns = set(
    existing_worker_leakage_features
)

candidate_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print("\nCandidate Input Features:")
print(
    "Number of Candidate Features:",
    len(candidate_features)
)

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
    "Predict total worker participation using "
    "non-leaking program, district, state, "
    "expenditure, wage, and employment features."
)
print("=" * 70)
