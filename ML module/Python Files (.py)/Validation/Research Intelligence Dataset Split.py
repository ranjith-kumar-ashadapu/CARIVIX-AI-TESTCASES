# ==========================================
# RESEARCH INTELLIGENCE DATASET SPLITTING
# ==========================================

import os
import hashlib
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================
# FILE PATHS
# ==========================================

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Feature_Engineering"
    r"\Research_Intelligence_Feature_Engineered.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Data_Splits"
)

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# CHECK INPUT FILE
# ==========================================

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Input file was not found:\n{input_file}"
    )

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

print("=" * 60)
print("RESEARCH INTELLIGENCE DATASET SPLITTING")
print("=" * 60)

print("Original Dataset Shape:", df.shape)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("Shape After Removing Duplicates:", df.shape)

# ==========================================
# SPLIT DATASET
# ==========================================

random_seed = 42

# 70% training and 30% temporary data
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=random_seed,
    shuffle=True
)

# Split temporary data into:
# 15% validation and 15% testing
validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=random_seed,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ==========================================
# VERIFY DATASET SIZES
# ==========================================

total_records = len(df)

print("\nDATASET SPLIT SUMMARY")
print("-" * 60)

print(
    "Training Records:",
    len(train_df),
    f"({len(train_df) / total_records * 100:.2f}%)"
)

print(
    "Validation Records:",
    len(validation_df),
    f"({len(validation_df) / total_records * 100:.2f}%)"
)

print(
    "Testing Records:",
    len(test_df),
    f"({len(test_df) / total_records * 100:.2f}%)"
)

print(
    "Total Records After Splitting:",
    len(train_df) + len(validation_df) + len(test_df)
)

# ==========================================
# VERIFY COLUMN CONSISTENCY
# ==========================================

if not (
    list(train_df.columns)
    == list(validation_df.columns)
    == list(test_df.columns)
):
    raise ValueError(
        "Columns are not consistent across the splits."
    )

print("Column consistency verified.")

# ==========================================
# CREATE ROW HASHES
# ==========================================

def create_row_hash(row):
    row_text = "||".join(row.astype(str).tolist())

    return hashlib.md5(
        row_text.encode("utf-8")
    ).hexdigest()

train_hashes = set(
    train_df.apply(create_row_hash, axis=1)
)

validation_hashes = set(
    validation_df.apply(create_row_hash, axis=1)
)

test_hashes = set(
    test_df.apply(create_row_hash, axis=1)
)

# ==========================================
# VERIFY NO OVERLAPPING RECORDS
# ==========================================

train_validation_overlap = (
    train_hashes.intersection(validation_hashes)
)

train_test_overlap = (
    train_hashes.intersection(test_hashes)
)

validation_test_overlap = (
    validation_hashes.intersection(test_hashes)
)

print("\nOVERLAP VERIFICATION")
print("-" * 60)

print(
    "Training-Validation Overlap:",
    len(train_validation_overlap)
)

print(
    "Training-Testing Overlap:",
    len(train_test_overlap)
)

print(
    "Validation-Testing Overlap:",
    len(validation_test_overlap)
)

if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise ValueError(
        "Duplicate records found across the dataset splits."
    )

print("No duplicate records found across the splits.")

# ==========================================
# SAVE DATASETS
# ==========================================

train_file = os.path.join(
    output_folder,
    "Research_Intelligence_Train.csv"
)

validation_file = os.path.join(
    output_folder,
    "Research_Intelligence_Validation.csv"
)

test_file = os.path.join(
    output_folder,
    "Research_Intelligence_Test.csv"
)

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("RESEARCH INTELLIGENCE SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Training File:")
print(train_file)

print("\nValidation File:")
print(validation_file)

print("\nTesting File:")
print(test_file)

print("=" * 60)
