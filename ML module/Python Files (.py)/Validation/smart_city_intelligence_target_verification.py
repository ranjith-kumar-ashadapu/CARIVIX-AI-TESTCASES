# ==========================================
# SMART CITY INTELLIGENCE TARGET VERIFICATION
# ==========================================

import os
import pandas as pd

# ==========================================
# TRAINING FILE PATH
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Data_Splits"
    r"\Smart_City_Intelligence_Train.csv"
)

# ==========================================
# TARGET COLUMN
# ==========================================

target_column = "Traffic_Condition"

# ==========================================
# CHECK FILE
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(train_file)

print("=" * 80)
print("SMART CITY INTELLIGENCE TARGET VERIFICATION")
print("=" * 80)

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

# ==========================================
# TARGET CLASS SUMMARY
# ==========================================

print("\nTarget Class Distribution:")

class_distribution = (
    df[target_column]
    .value_counts(dropna=False)
)

print(class_distribution.to_string())

print("\nTarget Class Percentages:")

class_percentages = (
    df[target_column]
    .value_counts(
        dropna=False,
        normalize=True
    )
    .mul(100)
    .round(2)
)

print(class_percentages.to_string())

# ==========================================
# TARGET AVAILABILITY
# ==========================================

missing_target_count = (
    df[target_column]
    .isnull()
    .sum()
)

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
# TARGET-DERIVED FEATURES
# ==========================================

target_derived_features = [
    # Target
    "Traffic_Condition",

# Direct encoded version of the target
    "Traffic_Condition_ID",

# Features grouped or calculated using traffic condition
    "Average_Vehicles_by_Traffic_Condition",
    "Vehicles_vs_Traffic_Condition_Average",

# Congestion and traffic-condition flags
    "Congestion_Risk_Flag",
    "High_Traffic_Flag",
    "Low_Speed_Flag",

# Categorised versions of related measurements
    "Speed_Category",
    "Speed_Category_ID",
    "Occupancy_Category",
    "Occupancy_Category_ID",
    "Vehicle_Count_Category",
    "Vehicle_Count_Category_ID",
]

existing_target_derived_features = [
    column
    for column in target_derived_features
    if column in df.columns
]

print(
    "\nFeatures Excluded Because They May Reveal "
    "or Derive the Target:"
)

for column in existing_target_derived_features:
    print("-", column)

# ==========================================
# OTHER FEATURES REQUIRING REVIEW
# ==========================================

review_features = [
    "Traffic_Speed_kmh",
    "Traffic_Speed_Log",
    "Road_Occupancy_%",
    "Road_Occupancy_Log",
    "Vehicle_Count",
    "Vehicle_Count_Log",
    "Vehicle_Count_per_Occupancy",
    "Occupancy_per_Vehicle",
    "Traffic_Light_State",
    "Traffic_Light_State_ID",
    "Weather_Condition",
    "Weather_Condition_ID",
    "Accident_Report",
    "Accident_Report_ID",
    "Ride_Sharing_Demand",
    "Parking_Availability",
    "Emission_Levels_g_km",
    "Energy_Consumption_L_h",
    "Hour",
    "Day_of_Week",
    "Time_Period",
    "Is_Peak_Hour",
    "Is_Morning_Peak",
    "Is_Evening_Peak",
    "Is_Off_Peak_Hour",
    "Latitude",
    "Longitude",
]

existing_review_features = [
    column
    for column in review_features
    if column in df.columns
]

print(
    "\nFeatures Requiring Availability-Time Review:"
)

for column in existing_review_features:
    print("-", column)

# ==========================================
# POSSIBLE DATETIME FEATURES
# ==========================================

print("\nDatetime Conversion Preview:")

for column in ["Timestamp", "Date"]:
    if column in df.columns:
        converted_values = pd.to_datetime(
            df[column],
            errors="coerce"
        )

print(
            column,
            "valid datetime values:",
            converted_values.notna().sum(),
            "of",
            len(df)
        )

# ==========================================
# CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(
        dropna=False
    ) <= 1
]

print("\nConstant Columns to Remove:")

if len(constant_columns) == 0:
    print("No constant columns found.")
else:
    for column in constant_columns:
        print("-", column)

# ==========================================
# CANDIDATE INPUT FEATURES
# ==========================================

excluded_columns = set(
    existing_target_derived_features
)

excluded_columns.update(
    constant_columns
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

print("\n" + "=" * 80)
print("TARGET VERIFICATION COMPLETED")
print("=" * 80)

print("Target Variable:", target_column)
print("Problem Type: Multiclass Classification")
print("Target Classes:", sorted(
    df[target_column]
    .dropna()
    .unique()
    .tolist()
))

print("Prediction Objective:")
print(
    "Classify traffic conditions as High, Medium, "
    "or Low using traffic, environmental, infrastructure, "
    "location, and time-based features."
)

print("=" * 80)
