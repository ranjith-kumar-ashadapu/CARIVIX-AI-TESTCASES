# ============================================================
# SMART CITY INTELLIGENCE
# TRAFFIC ANALYSIS: EDA AND MACHINE LEARNING
# Dataset: Traffic Analysis_Final.csv
# Target: Traffic_Condition
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

project_root = Path(__file__).resolve().parents[2]
dataset_path = project_root / "data" / "Traffic Analysis_Final.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")
print("Original Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nDataset Information")
df.info()

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()

print("\nShape after removing duplicates:")
print(df.shape)

# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Timestamp",
    "Traffic_Condition"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ============================================================
# 6. CONVERT TIMESTAMP
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# ============================================================
# 7. CREATE TIME FEATURES
# ============================================================

df["Date"] = df["Timestamp"].dt.date
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour
df["Day_Name"] = df["Timestamp"].dt.day_name()

# ============================================================
# 8. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns_expected = [
    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_%"
]

for column in numeric_columns_expected:

    if column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ============================================================
# 9. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().copy()

print("\nShape after cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ============================================================
# 10. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive Statistics")
print(df.describe(include="all"))

# ============================================================
# 11. TRAFFIC CONDITION DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Traffic_Condition",
    order=df["Traffic_Condition"].value_counts().index
)

plt.title("Traffic Condition Distribution")
plt.xlabel("Traffic Condition")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 12. WEATHER CONDITION DISTRIBUTION
# ============================================================

if "Weather_Condition" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="Weather_Condition",
        order=df["Weather_Condition"].value_counts().index
    )

    plt.title("Weather Condition Distribution")
    plt.xlabel("Weather Condition")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================================
# 13. TRAFFIC LIGHT STATE DISTRIBUTION
# ============================================================

if "Traffic_Light_State" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Traffic_Light_State",
        order=df["Traffic_Light_State"].value_counts().index
    )

    plt.title("Traffic Light State Distribution")
    plt.xlabel("Traffic Light State")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================================
# 14. ACCIDENT REPORT DISTRIBUTION
# ============================================================

if "Accident_Report" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Accident_Report",
        order=df["Accident_Report"].value_counts().index
    )

    plt.title("Accident Report Distribution")
    plt.xlabel("Accident Report")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================================
# 15. NUMERIC COLUMN DISTRIBUTIONS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) > 0:

    df[numeric_columns].hist(
        figsize=(18, 14),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle("Distribution of Numeric Variables")
    plt.tight_layout()
    plt.show()

# ============================================================
# 16. VEHICLE COUNT DISTRIBUTION
# ============================================================

if "Vehicle_Count" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Vehicle_Count",
        bins=30,
        kde=True,
        color="steelblue"
    )

    plt.title("Vehicle Count Distribution")
    plt.xlabel("Vehicle Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ============================================================
# 17. TRAFFIC SPEED DISTRIBUTION
# ============================================================

if "Traffic_Speed_kmh" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Traffic_Speed_kmh",
        bins=30,
        kde=True,
        color="darkorange"
    )

    plt.title("Traffic Speed Distribution")
    plt.xlabel("Traffic Speed (km/h)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ============================================================
# 18. VEHICLE COUNT OUTLIERS
# ============================================================

if "Vehicle_Count" in df.columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df["Vehicle_Count"],
        color="lightgreen"
    )

    plt.title("Vehicle Count Outliers")
    plt.xlabel("Vehicle Count")
    plt.tight_layout()
    plt.show()

# ============================================================
# 19. CORRELATION HEATMAP
# ============================================================

numeric_columns_corr = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns_corr) > 1:

    correlation_matrix = df[
        numeric_columns_corr
    ].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ============================================================
# 20. VEHICLE COUNT VS TRAFFIC SPEED
# ============================================================

if (
    "Vehicle_Count" in df.columns
    and "Traffic_Speed_kmh" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Vehicle_Count",
        y="Traffic_Speed_kmh",
        hue="Traffic_Condition",
        alpha=0.7
    )

    plt.title("Vehicle Count vs Traffic Speed")
    plt.xlabel("Vehicle Count")
    plt.ylabel("Traffic Speed (km/h)")
    plt.tight_layout()
    plt.show()

# ============================================================
# 21. ROAD OCCUPANCY VS VEHICLE COUNT
# ============================================================

if (
    "Road_Occupancy_%" in df.columns
    and "Vehicle_Count" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Road_Occupancy_%",
        y="Vehicle_Count",
        hue="Traffic_Condition",
        alpha=0.7
    )

    plt.title("Road Occupancy vs Vehicle Count")
    plt.xlabel("Road Occupancy (%)")
    plt.ylabel("Vehicle Count")
    plt.tight_layout()
    plt.show()

# ============================================================
# 22. AVERAGE VEHICLE COUNT BY HOUR
# ============================================================

if (
    "Hour" in df.columns
    and "Vehicle_Count" in df.columns
):

    hourly_vehicle_count = (
        df.groupby("Hour")["Vehicle_Count"]
        .mean()
    )

    plt.figure(figsize=(10, 5))

    hourly_vehicle_count.plot(
        marker="o",
        color="purple"
    )

    plt.title("Average Vehicle Count by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Vehicle Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# 23. TRAFFIC SPEED BY WEATHER CONDITION
# ============================================================

if (
    "Weather_Condition" in df.columns
    and "Traffic_Speed_kmh" in df.columns
):

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Weather_Condition",
        y="Traffic_Speed_kmh"
    )

    plt.title("Traffic Speed by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Traffic Speed (km/h)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================================
# 24. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

df_ml = df.copy()

# Target variable
y = df_ml["Traffic_Condition"].copy()

# Remove unnecessary columns
columns_to_drop = [
    "Timestamp",
    "Date",
    "Traffic_Condition"
]

columns_to_drop = [
    col for col in columns_to_drop
    if col in df_ml.columns
]

X = df_ml.drop(columns=columns_to_drop)

# Identify categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

# One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Convert all columns to numeric
for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# Replace infinite values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing values
X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)

# Verify all columns are numeric
non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nRemaining Non-Numeric Columns:")
print(non_numeric_columns)

if len(non_numeric_columns) > 0:
    raise ValueError(
        "Non-numeric columns still exist: "
        + str(non_numeric_columns)
    )

print("\nMachine Learning dataset is ready.")

# ============================================================
# 25. TRAIN, VALIDATION, AND TEST SPLIT
# ============================================================

class_counts = y.value_counts()

can_stratify = (
    len(class_counts) > 1
    and class_counts.min() >= 2
)

if can_stratify:

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    temp_class_counts = y_temp.value_counts()

    can_stratify_temp = (
        len(temp_class_counts) > 1
        and temp_class_counts.min() >= 2
    )

    if can_stratify_temp:

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

    else:

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42
        )

else:

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

print("\nDataset Split")
print("-------------------------")
print("Training Set  :", X_train.shape, y_train.shape)
print("Validation Set:", X_val.shape, y_val.shape)
print("Testing Set   :", X_test.shape, y_test.shape)

# ============================================================
# 26. TRAIN RANDOM FOREST CLASSIFIER
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest model trained successfully.")

# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features")
print(importance.head(20))

top_features = (
    importance.head(20)
    .sort_values(by="Importance")
)

plt.figure(figsize=(11,8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="teal"
)

plt.title("Top 20 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ============================================================
# 28. VALIDATION PERFORMANCE
# ============================================================

validation_predictions = model.predict(X_val)

validation_accuracy = accuracy_score(
    y_val,
    validation_predictions
)

validation_precision = precision_score(
    y_val,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_recall = recall_score(
    y_val,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_f1 = f1_score(
    y_val,
    validation_predictions,
    average="weighted",
    zero_division=0
)

print("\nValidation Performance")
print("-------------------------")
print(f"Accuracy : {validation_accuracy:.4f}")
print(f"Precision: {validation_precision:.4f}")
print(f"Recall   : {validation_recall:.4f}")
print(f"F1 Score : {validation_f1:.4f}")

# ============================================================
# 29. TEST SET EVALUATION
# ============================================================

test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

print("\nTest Performance")
print("-------------------------")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")

# ============================================================
# 30. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("-------------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# ============================================================
# 31. CONFUSION MATRIX
# ============================================================

class_labels = model.classes_

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Traffic Condition")
plt.ylabel("Actual Traffic Condition")
plt.tight_layout()
plt.show()

# ============================================================
# 32. SAVE RESULTS
# ============================================================

# Save predictions
predictions = pd.DataFrame({
    "Actual_Traffic_Condition": y_test.values,
    "Predicted_Traffic_Condition": test_predictions
})

predictions.to_csv(
    "traffic_predictions.csv",
    index=False
)

# Save feature importance
importance.to_csv(
    "traffic_feature_importance.csv",
    index=False
)

# Save cleaned dataset
df.to_csv(
    "traffic_cleaned_dataset.csv",
    index=False
)

print("\nFiles saved successfully:")
print("1. traffic_predictions.csv")
print("2. traffic_feature_importance.csv")
print("3. traffic_cleaned_dataset.csv")

# ============================================================
# 33. COMPLETION MESSAGE
# ============================================================

print("\n==========================================")
print("SMART CITY TRAFFIC ANALYSIS COMPLETED")
print("EDA Completed")
print("Feature Engineering Completed")
print("Random Forest Model Trained")
print("Validation and Testing Completed")
print("Results Saved Successfully")
print("==========================================")
