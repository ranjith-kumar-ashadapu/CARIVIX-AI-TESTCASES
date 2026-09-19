# ==========================================
# SMART CITY INTELLIGENCE DATASET INSPECTION
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

print("=" * 80)
print("SMART CITY INTELLIGENCE DATASET INSPECTION")
print("=" * 80)

# ==========================================
# DATASET SHAPE
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Records:", df.shape[0])
print("Number of Columns:", df.shape[1])

# ==========================================
# COLUMN NAMES AND DATA TYPES
# ==========================================

print("\nColumn Names and Data Types:")
print(df.dtypes.to_string())

# ==========================================
# MISSING-VALUE SUMMARY
# ==========================================

print("\nMissing-Value Summary:")

missing_values = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) == 0:
    print("No missing values found.")
else:
    print(missing_values.to_string())

# ==========================================
# UNIQUE-VALUE SUMMARY
# ==========================================

print("\nUnique-Value Summary:")

unique_values = (
    df.nunique(dropna=False)
    .sort_values()
)

print(unique_values.to_string())

# ==========================================
# DUPLICATE-ROW CHECK
# ==========================================

print("\nDuplicate-Row Summary:")

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

# ==========================================
# FIRST FIVE ROWS
# ==========================================

print("\nFirst Five Rows:")
print(df.head().to_string())

# ==========================================
# NUMERICAL COLUMNS
# ==========================================

numerical_columns = (
    df.select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

print("\nNumerical Columns:")
print(numerical_columns)

print(
    "\nNumber of Numerical Columns:",
    len(numerical_columns)
)

# ==========================================
# CATEGORICAL COLUMNS
# ==========================================

categorical_columns = (
    df.select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)

print("\nCategorical Columns:")
print(categorical_columns)

print(
    "\nNumber of Categorical Columns:",
    len(categorical_columns)
)

# ==========================================
# POSSIBLE DATE/TIME COLUMNS
# ==========================================

possible_datetime_columns = []

for column in df.columns:
    column_name = column.lower()

if any(
        keyword in column_name
        for keyword in [
            "date",
            "time",
            "timestamp",
            "datetime",
            "hour",
            "day",
            "month",
            "year"
        ]
    ):
        possible_datetime_columns.append(column)

print("\nPossible Date/Time Columns:")
print(possible_datetime_columns)

# ==========================================
# POSSIBLE IDENTIFIER COLUMNS
# ==========================================

possible_identifier_columns = []

identifier_keywords = [
    "id",
    "index",
    "serial",
    "record",
    "code",
    "number"
]

for column in df.columns:
    column_name = column.lower()

if any(
        keyword in column_name
        for keyword in identifier_keywords
    ):
        possible_identifier_columns.append(column)

print("\nPossible Identifier Columns:")
print(possible_identifier_columns)

# ==========================================
# NUMERICAL DESCRIPTIVE SUMMARY
# ==========================================

if len(numerical_columns) > 0:
    print("\nNumerical Descriptive Summary:")
    print(
        df[numerical_columns]
        .describe()
        .transpose()
        .to_string()
    )

# ==========================================
# CATEGORICAL VALUE PREVIEW
# ==========================================

print("\nCategorical-Value Preview:")

if len(categorical_columns) == 0:
    print("No categorical columns found.")
else:
    for column in categorical_columns:
        print(f"\nColumn: {column}")
        print(
            df[column]
            .value_counts(dropna=False)
            .head(15)
            .to_string()
        )

# ==========================================
# CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

print("\nConstant Columns:")

if len(constant_columns) == 0:
    print("No constant columns found.")
else:
    for column in constant_columns:
        print("-", column)

# ==========================================
# CORRELATION PREVIEW
# ==========================================

if len(numerical_columns) >= 2:
    print("\nNumerical Correlation Matrix Preview:")
    print(
        df[numerical_columns]
        .corr()
        .round(3)
        .to_string()
    )
else:
    print(
        "\nCorrelation analysis requires at least "
        "two numerical columns."
    )

print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)
