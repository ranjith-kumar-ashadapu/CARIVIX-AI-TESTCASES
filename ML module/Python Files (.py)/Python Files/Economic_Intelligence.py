# ============================================================
# ECONOMIC DATASET ANALYSIS AND MACHINE LEARNING
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
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
dataset_path = project_root / "data" / "Economic_Trend_Analysis_Cleaned (1).csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
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
# 5. CONVERT NUMERIC COLUMNS
# ============================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Value"] = pd.to_numeric(
    df["Value"],
    errors="coerce"
)

# ============================================================
# 6. REMOVE MISSING AND INVALID VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

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

if "country" in df.columns:
    print("\nNumber of countries:", df["country"].nunique())
    print("Countries:")
    print(df["country"].unique())

if "indicator" in df.columns:
    print("\nNumber of indicators:", df["indicator"].nunique())
    print("Indicators:")
    print(df["indicator"].unique())

if "unit" in df.columns:
    print("\nNumber of units:", df["unit"].nunique())
    print("Units:")
    print(df["unit"].unique())

# ============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ============================================================

# Country distribution
if "country" in df.columns:
    plt.figure(figsize=(12, 5))

    df["country"].value_counts().head(20).plot(
        kind="bar",
        color="steelblue"
    )

    plt.title("Top Countries by Number of Records")
    plt.xlabel("Country")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Indicator distribution
if "indicator" in df.columns:
    plt.figure(figsize=(12, 5))

    df["indicator"].value_counts().head(20).plot(
        kind="bar",
        color="darkorange"
    )

    plt.title("Most Common Economic Indicators")
    plt.xlabel("Indicator")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Scale distribution
if "scale" in df.columns:
    plt.figure(figsize=(8, 5))

    df["scale"].value_counts().plot(
        kind="bar",
        color="seagreen"
    )

    plt.title("Distribution of Measurement Scales")
    plt.xlabel("Scale")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Unit distribution
if "unit" in df.columns:
    plt.figure(figsize=(8, 5))

    df["unit"].value_counts().plot(
        kind="bar",
        color="purple"
    )

    plt.title("Distribution of Units")
    plt.xlabel("Unit")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Histograms for Year and Value
df[["Year", "Value"]].hist(
    figsize=(10, 5),
    bins=30,
    color="skyblue",
    edgecolor="black"
)

plt.suptitle("Distribution of Year and Economic Values")
plt.tight_layout()
plt.show()

# Distribution of Value
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Value",
    bins=40,
    kde=True,
    color="orange"
)

plt.title("Distribution of Economic Indicator Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Outlier detection
plt.figure(figsize=(10, 4))

sns.boxplot(
    x=df["Value"],
    color="lightgreen"
)

plt.title("Outlier Detection for Economic Values")
plt.xlabel("Value")
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_columns = ["Year", "Value"]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(5, 4))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ============================================================
# 10. INDIA-SPECIFIC TREND ANALYSIS
# ============================================================

if "country" in df.columns and "indicator" in df.columns:

    country_name = "India"

india_data = df[
        df["country"].astype(str).str.lower()
        == country_name.lower()
    ].copy()

if not india_data.empty:

    plt.figure(figsize=(12, 6))

selected_indicators = (
            india_data["indicator"]
            .dropna()
            .unique()[:5]
        )

for indicator_name in selected_indicators:

        indicator_data = india_data[
                india_data["indicator"] == indicator_name
            ].sort_values("Year")

        plt.plot(
                indicator_data["Year"],
                indicator_data["Value"],
                marker="o",
                label=str(indicator_name)
            )

        plt.title("Economic Indicators for India")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

else:
        print("\nIndia was not found in the dataset.")

# ============================================================
# 11. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

df_ml = df.copy()

# Make sure the target column is numeric
df_ml["Value"] = pd.to_numeric(
    df_ml["Value"],
    errors="coerce"
)

# Remove invalid rows again
df_ml = df_ml.replace(
    [np.inf, -np.inf],
    np.nan
)

df_ml = df_ml.dropna().copy()

# Automatically identify all categorical columns
categorical_columns = df_ml.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns before encoding:")
print(categorical_columns)

# Encode every categorical column
label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df_ml[column] = encoder.fit_transform(
        df_ml[column].astype(str)
    )

label_encoders[column] = encoder

# Convert remaining columns to numeric where possible
for column in df_ml.columns:

    df_ml[column] = pd.to_numeric(
        df_ml[column],
        errors="coerce"
    )

# Remove rows that became missing after conversion
df_ml = df_ml.replace(
    [np.inf, -np.inf],
    np.nan
)

df_ml = df_ml.dropna().copy()

# ============================================================
# 12. CREATE FEATURES AND TARGET
# ============================================================

X = df_ml.drop(
    columns=["Value"]
)

y = df_ml["Value"]

print("\nData types after encoding:")
print(X.dtypes)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# Verify that no text columns remain
non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print("\nNon-numeric columns in X:")
print(non_numeric_columns)

if len(non_numeric_columns) > 0:
    raise ValueError(
        "Non-numeric columns remain in X: "
        + str(non_numeric_columns)
    )

print("\nAll feature columns are numeric.")

# ============================================================
# 13. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS
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
print("Training set  :", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set      :", X_test.shape)

# ============================================================
# 14. TRAIN RANDOM FOREST REGRESSOR
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
# 15. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 important features:")
print(feature_importance.head(20))

# Plot feature importance
top_features = feature_importance.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

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
# 16. VALIDATION SET EVALUATION
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
# 17. TEST SET EVALUATION
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

print("\nTest Performance")
print("----------------")
print(f"MAE     : {test_mae:.4f}")
print(f"MSE     : {test_mse:.4f}")
print(f"RMSE    : {test_rmse:.4f}")
print(f"R² Score: {test_r2:.4f}")

# ============================================================
# 18. ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.6,
    color="royalblue"
)

# Reference line
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
    linestyle="--"
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.tight_layout()
plt.show()

# ============================================================
# 19. RESIDUAL ANALYSIS
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

print("\nComplete analysis and machine-learning workflow finished successfully.")
