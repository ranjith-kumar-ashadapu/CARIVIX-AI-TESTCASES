# ============================================================
# GOVERNMENT PROGRAM EVALUATION
# EXPLORATORY DATA ANALYSIS AND MACHINE LEARNING
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 2. LOAD DATASET
# ============================================================

project_root = Path(__file__).resolve().parents[2]
dataset_path = project_root / "data" / "Market_Analysis_1981_2025_Final_Cleaned.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nDataset information:")
df.info()

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDuplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()

print("\nShape after removing duplicates:", df.shape)

# ============================================================
# 5. IDENTIFY IMPORTANT COLUMNS
# ============================================================

target_column = "Total Exp(Rs. in Lakhs.)"

worker_column = "Total No. of Workers"

wage_column = "Average Wage rate per day per person(Rs.)"

state_column = "state_name"

district_column = "district_name"

# Check required columns
required_columns = [
    target_column,
    worker_column,
    wage_column,
    state_column,
    district_column
]

missing_required_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required_columns:
    raise ValueError(
        "The following required columns are missing: "
        + str(missing_required_columns)
    )

# ============================================================
# 6. CONVERT NUMERIC COLUMNS
# ============================================================

# Convert all columns except known categorical columns
categorical_columns = [
    state_column,
    district_column
]

numeric_columns_to_convert = [
    column for column in df.columns
    if column not in categorical_columns
]

for column in numeric_columns_to_convert:

# Convert possible commas, currency symbols, and spaces
    if df[column].dtype == "object":

        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("Rs.", "", regex=False)
            .str.strip()
        )

df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Replace infinite values with missing values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows with missing values
df = df.dropna().copy()

print("\nShape after cleaning:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive statistics:")
print(df.describe(include="all"))

# ============================================================
# 8. UNIQUE VALUES
# ============================================================

print("\nNumber of states:", df[state_column].nunique())
print("States:")
print(df[state_column].unique())

print("\nNumber of districts:", df[district_column].nunique())

# ============================================================
# 9. STATE DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 6))

df[state_column].value_counts().plot(
    kind="bar",
    color="steelblue"
)

plt.title("Number of Districts by State")
plt.xlabel("State")
plt.ylabel("District Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# 10. TOP 10 DISTRICTS BY TOTAL NUMBER OF WORKERS
# ============================================================

top_workers = df.sort_values(
    by=worker_column,
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_workers[district_column].astype(str),
    top_workers[worker_column],
    color="seagreen"
)

plt.title("Top 10 Districts by Total Number of Workers")
plt.xlabel("District")
plt.ylabel("Total Number of Workers")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# 11. HISTOGRAMS OF NUMERIC COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

df[numeric_columns].hist(
    figsize=(20, 18),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Distribution of Numeric Variables")
plt.tight_layout()
plt.show()

# ============================================================
# 12. DISTRIBUTION OF AVERAGE WAGE RATE
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x=wage_column,
    bins=30,
    kde=True,
    color="orange"
)

plt.title("Distribution of Average Wage Rate")
plt.xlabel("Average Wage Rate per Day per Person (Rs.)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# 13. OUTLIER DETECTION FOR TOTAL EXPENDITURE
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df[target_column],
    color="lightgreen"
)

plt.title("Outlier Detection - Total Expenditure")
plt.xlabel("Total Expenditure (Rs. in Lakhs.)")
plt.tight_layout()
plt.show()

# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

numeric_correlation = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_correlation.corr()

plt.figure(
    figsize=(
        max(10, len(correlation_matrix.columns)),
        max(8, len(correlation_matrix.columns))
    )
)

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
# 15. TOP 10 DISTRICTS BY TOTAL EXPENDITURE
# ============================================================

top_expenditure = df.sort_values(
    by=target_column,
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_expenditure[district_column].astype(str),
    top_expenditure[target_column],
    color="darkorange"
)

plt.title("Top 10 Districts by Total Expenditure")
plt.xlabel("District")
plt.ylabel("Total Expenditure (Rs. in Lakhs.)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# 16. WORKERS VS TOTAL EXPENDITURE
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x=worker_column,
    y=target_column,
    color="royalblue"
)

plt.title("Total Workers vs Total Expenditure")
plt.xlabel("Total Number of Workers")
plt.ylabel("Total Expenditure (Rs. in Lakhs.)")
plt.tight_layout()
plt.show()

# ============================================================
# 17. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

df_ml = df.copy()

# Separate features and target
X = df_ml.drop(
    columns=[target_column]
)

y = df_ml[target_column]

# Automatically identify all categorical columns
categorical_columns_for_encoding = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns before encoding:")
print(categorical_columns_for_encoding)

# One-hot encode categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns_for_encoding,
    drop_first=True,
    dtype=int
)

# Convert all feature columns to numeric
for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# Make sure the target is numeric
y = pd.to_numeric(
    y,
    errors="coerce"
)

# Remove invalid rows from X and y
valid_rows = (
    X.notna().all(axis=1)
    & y.notna()
)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Replace infinite values if any
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_rows = (
    X.notna().all(axis=1)
    & y.notna()
)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Final validation
non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nNon-numeric columns remaining in X:")
print(non_numeric_columns)

if len(non_numeric_columns) > 0:
    raise ValueError(
        "Non-numeric columns remain in X: "
        + str(non_numeric_columns)
    )

print("\nAll feature columns are numeric.")

# ============================================================
# 18. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS
# ============================================================

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

print("\nDataset split:")
print("Training set  :", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Testing set   :", X_test.shape, y_test.shape)

# ============================================================
# 19. TRAIN RANDOM FOREST REGRESSOR
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest model trained successfully.")

# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 important features:")
print(importance.head(20))

# Plot the top 20 features
top_features = importance.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="teal"
)

plt.title("Top 20 Important Features - Government Program")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ============================================================
# 21. VALIDATION SET EVALUATION
# ============================================================

validation_predictions = model.predict(X_val)

validation_mae = mean_absolute_error(
    y_val,
    validation_predictions
)

validation_mse = mean_squared_error(
    y_val,
    validation_predictions
)

validation_rmse = np.sqrt(
    validation_mse
)

validation_r2 = r2_score(
    y_val,
    validation_predictions
)

print("\nValidation Performance")
print("----------------------")
print(f"MAE     : {validation_mae:.4f}")
print(f"MSE     : {validation_mse:.4f}")
print(f"RMSE    : {validation_rmse:.4f}")
print(f"R² Score: {validation_r2:.4f}")

# ============================================================
# 22. TEST SET EVALUATION
# ============================================================

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    test_mse
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("\nGovernment Dataset Performance")
print("------------------------------")
print(f"MAE     : {test_mae:.4f}")
print(f"MSE     : {test_mse:.4f}")
print(f"RMSE    : {test_rmse:.4f}")
print(f"R² Score: {test_r2:.4f}")

# ============================================================
# 23. ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.7,
    color="royalblue"
)

minimum_value = min(
    y_test.min(),
    test_predictions.min()
)

maximum_value = max(
    y_test.max(),
    test_predictions.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title("Actual vs Predicted Total Expenditure")
plt.xlabel("Actual Total Expenditure")
plt.ylabel("Predicted Total Expenditure")
plt.tight_layout()
plt.show()

# ============================================================
# 24. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - test_predictions

plt.figure(figsize=(9, 5))

sns.histplot(
    residuals,
    bins=30,
    kde=True,
    color="coral"
)

plt.title("Distribution of Prediction Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# 25. COMPLETION MESSAGE
# ============================================================

print("\nComplete Government Program analysis finished successfully.")
