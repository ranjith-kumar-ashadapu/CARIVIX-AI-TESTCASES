import os
import pandas as pd

# ==========================================
# FILE PATH
# ==========================================

file_path = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Data_Splits"
    r"\Research_Intelligence_Train.csv"
)

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "File not found:\n" + file_path
    )

df = pd.read_csv(file_path)

print("=" * 80)
print("RESEARCH INTELLIGENCE DATASET INSPECTION")
print("=" * 80)

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names and Data Types:")
print(df.dtypes.to_string())

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing-Value Summary:")

missing_values = df.isnull().sum()
missing_values = missing_values[
    missing_values > 0
]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values.to_string())

# ==========================================
# DUPLICATES
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# COLUMN GROUPS
# ==========================================

numerical_columns = list(
    df.select_dtypes(
        include=["number"]
    ).columns
)

object_columns = list(
    df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
)

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical/Text Columns:")
print(object_columns)

# ==========================================
# POSSIBLE TEXT COLUMNS
# ==========================================

text_keywords = [
    "title",
    "abstract",
    "summary",
    "description",
    "keyword",
    "text",
    "content",
    "paper",
    "research",
    "topic"
]

possible_text_columns = []

for column in df.columns:
    column_name = str(column).lower()

keyword_match = any(
        word in column_name
        for word in text_keywords
    )

if keyword_match:
        possible_text_columns.append(column)

# Include object columns because text columns
# may have names unrelated to text.
for column in object_columns:
    if column not in possible_text_columns:
        possible_text_columns.append(column)

print("\nPossible Text Columns:")

if not possible_text_columns:
    print("No text or object columns detected.")
else:
    for column in possible_text_columns:
        print("-", column)

# ==========================================
# TEXT LENGTHS
# ==========================================

print("\nText-Length Preview:")

if not possible_text_columns:
    print("No text columns available.")
else:
    for column in possible_text_columns:
        column_lengths = (
            df[column]
            .fillna("")
            .astype(str)
            .str.len()
        )

print(
            column,
            "| Minimum:",
            column_lengths.min(),
            "| Median:",
            column_lengths.median(),
            "| Maximum:",
            column_lengths.max()
        )

# ==========================================
# POSSIBLE TARGET COLUMNS
# ==========================================

target_keywords = [
    "target",
    "label",
    "class",
    "category",
    "topic",
    "subject",
    "field",
    "type",
    "area",
    "discipline"
]

print("\nPossible Target Columns:")

target_found = False

for column in df.columns:
    column_name = str(column).lower()

unique_count = df[column].nunique(
        dropna=True
    )

keyword_match = any(
        word in column_name
        for word in target_keywords
    )

if keyword_match:
        target_found = True

print(
            "-",
            column,
            "| Unique Values:",
            unique_count
        )

if not target_found:
    print(
        "No target column detected automatically."
    )

# ==========================================
# LOW-CARDINALITY COLUMNS
# ==========================================

print(
    "\nLow-Cardinality Categorical Columns:"
)

low_cardinality_found = False

for column in object_columns:
    unique_count = df[column].nunique(
        dropna=True
    )

if 2 <= unique_count <= 50:
        low_cardinality_found = True

print(
            "-",
            column,
            "| Unique Values:",
            unique_count
        )

if not low_cardinality_found:
    print(
        "No low-cardinality categorical "
        "columns found."
    )

# ==========================================
# CONSTANT COLUMNS
# ==========================================

print("\nConstant Columns:")

constant_found = False

for column in df.columns:
    unique_count = df[column].nunique(
        dropna=False
    )

if unique_count <= 1:
        constant_found = True
        print("-", column)

if not constant_found:
    print("No constant columns found.")

# ==========================================
# VALUE PREVIEW
# ==========================================

print("\nCategorical Value Preview:")

for column in object_columns:
    print("\nColumn:", column)

print(
        df[column]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)
