# ==========================================
# PUBLIC PROGRAM EVALUATION
# FEATURE ENGINEERING
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import numpy as np
import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Public Program Evaluation_final.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Feature_Engineering"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Public_Program_Evaluation_Feature_Engineered.csv"
)

dictionary_file = os.path.join(
    output_folder,
    "Public_Program_Feature_Dictionary.csv"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

original_columns = df.columns.tolist()

print("=" * 60)
print("PUBLIC PROGRAM EVALUATION FEATURE ENGINEERING")
print("=" * 60)

print("Original Dataset Shape:", df.shape)

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

# ==========================================
# COLUMN NAMES
# ==========================================

state_column = "state_name"
district_column = "district_name"

workers_column = "Total No. of Workers"
expenditure_column = "Total Exp(Rs. in Lakhs.)"
wage_column = "Average Wage rate per day per person(Rs.)"

# ==========================================
# NUMERIC CONVERSION
# ==========================================

numeric_columns = [
    workers_column,
    expenditure_column,
    wage_column
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Shape After Duplicate Removal:", df.shape)

# ==========================================
# TEXT CLEANING
# ==========================================

text_columns = df.select_dtypes(
    include="object"
).columns

for column in text_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .replace("nan", np.nan)
    )

# ==========================================
# LOCATION FEATURES
# ==========================================

if state_column in df.columns:

    df["State_ID"] = (
        df[state_column]
        .astype("category")
        .cat.codes
    )

    df["State_Record_Count"] = (
        df.groupby(state_column)[state_column]
        .transform("count")
    )

else:

    df["State_ID"] = np.nan
    df["State_Record_Count"] = np.nan

if district_column in df.columns:

    df["District_ID"] = (
        df[district_column]
        .astype("category")
        .cat.codes
    )

    df["District_Record_Count"] = (
        df.groupby(district_column)[district_column]
        .transform("count")
    )

else:

    df["District_ID"] = np.nan
    df["District_Record_Count"] = np.nan

if (
    state_column in df.columns
    and district_column in df.columns
):

    df["State_District_Count"] = (
        df.groupby(state_column)[district_column]
        .transform("nunique")
    )

else:

    df["State_District_Count"] = np.nan

# ==========================================
# WORKER FEATURES
# ==========================================

if workers_column in df.columns:

    df["Workers_Log"] = np.where(
        df[workers_column] >= 0,
        np.log1p(df[workers_column]),
        np.nan
    )

    df["Workers_Missing_Flag"] = (
        df[workers_column]
        .isnull()
        .astype(int)
    )

    if state_column in df.columns:

        df["State_Total_Workers"] = (
            df.groupby(state_column)[workers_column]
            .transform("sum")
        )

        df["State_Average_Workers"] = (
            df.groupby(state_column)[workers_column]
            .transform("mean")
        )

        df["Worker_Share_of_State"] = np.where(
            df["State_Total_Workers"] != 0,
            (
                df[workers_column]
                /
                df["State_Total_Workers"]
                *
                100
            ),
            np.nan
        )

# ==========================================
# EXPENDITURE FEATURES
# ==========================================

if expenditure_column in df.columns:

    df["Expenditure_Log"] = np.where(
        df[expenditure_column] >= 0,
        np.log1p(df[expenditure_column]),
        np.nan
    )

    df["Expenditure_Missing_Flag"] = (
        df[expenditure_column]
        .isnull()
        .astype(int)
    )

    if state_column in df.columns:

        df["State_Total_Expenditure"] = (
            df.groupby(state_column)[expenditure_column]
            .transform("sum")
        )

        df["State_Average_Expenditure"] = (
            df.groupby(state_column)[expenditure_column]
            .transform("mean")
        )

        df["Expenditure_Share_of_State"] = np.where(
            df["State_Total_Expenditure"] != 0,
            (
                df[expenditure_column]
                /
                df["State_Total_Expenditure"]
                *
                100
            ),
            np.nan
        )

# ==========================================
# WAGE FEATURES
# ==========================================

if wage_column in df.columns:

    df["Wage_Log"] = np.where(
        df[wage_column] >= 0,
        np.log1p(df[wage_column]),
        np.nan
    )

    df["Wage_Missing_Flag"] = (
        df[wage_column]
        .isnull()
        .astype(int)
    )

# ==========================================
# EFFICIENCY FEATURES
# ==========================================

if (
    workers_column in df.columns
    and expenditure_column in df.columns
):

    df["Expenditure_Per_Worker_Lakhs"] = np.where(
        df[workers_column] > 0,
        df[expenditure_column]
        /
        df[workers_column],
        np.nan
    )

    df["Expenditure_Per_Worker_Rs"] = (
        df["Expenditure_Per_Worker_Lakhs"]
        *
        100000
    )

    df["Workers_Per_Expenditure_Lakh"] = np.where(
        df[expenditure_column] > 0,
        df[workers_column]
        /
        df[expenditure_column],
        np.nan
    )

if (
    workers_column in df.columns
    and wage_column in df.columns
):

    df["Worker_Wage_Ratio"] = np.where(
        df[wage_column] > 0,
        df[workers_column]
        /
        df[wage_column],
        np.nan
    )

if (
    expenditure_column in df.columns
    and wage_column in df.columns
):

    df["Expenditure_Wage_Ratio"] = np.where(
        df[wage_column] > 0,
        df[expenditure_column]
        /
        df[wage_column],
        np.nan
    )

# ==========================================
# CATEGORY FEATURES (FIXED)
# ==========================================

if wage_column in df.columns:

    median = df[wage_column].median()

    df["Wage_Category"] = np.select(
        [
            df[wage_column].isnull(),
            df[wage_column] < median,
            df[wage_column] >= median
        ],
        [
            "Missing",
            "Below Median Wage",
            "At or Above Median Wage"
        ],
        default="Missing"
    )

if expenditure_column in df.columns:

    median = df[expenditure_column].median()

    df["Expenditure_Category"] = np.select(
        [
            df[expenditure_column].isnull(),
            df[expenditure_column] < median,
            df[expenditure_column] >= median
        ],
        [
            "Missing",
            "Below Median Expenditure",
            "At or Above Median Expenditure"
        ],
        default="Missing"
    )

if workers_column in df.columns:

    median = df[workers_column].median()

    df["Worker_Category"] = np.select(
        [
            df[workers_column].isnull(),
            df[workers_column] < median,
            df[workers_column] >= median
        ],
        [
            "Missing",
            "Below Median Workers",
            "At or Above Median Workers"
        ],
        default="Missing"
    )

# ==========================================
# OUTLIER FUNCTION
# ==========================================

def create_outlier_flag(dataframe, column, new_column):

    if column not in dataframe.columns:
        return

    values = dataframe[column].dropna()

    if values.empty:

        dataframe[new_column] = 0
        return

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    dataframe[new_column] = np.where(
        (
            dataframe[column] < lower_limit
        )
        |
        (
            dataframe[column] > upper_limit
        ),
        1,
        0
    )

create_outlier_flag(
    df,
    workers_column,
    "Workers_Outlier_Flag"
)

create_outlier_flag(
    df,
    expenditure_column,
    "Expenditure_Outlier_Flag"
)

create_outlier_flag(
    df,
    wage_column,
    "Wage_Outlier_Flag"
)

# ==========================================
# STATE COMPARISON FEATURES
# ==========================================

if state_column in df.columns:

    if workers_column in df.columns:

        df["Workers_vs_State_Median"] = (
            df[workers_column]
            -
            df.groupby(state_column)[workers_column]
            .transform("median")
        )

    if expenditure_column in df.columns:

        df["Expenditure_vs_State_Median"] = (
            df[expenditure_column]
            -
            df.groupby(state_column)[expenditure_column]
            .transform("median")
        )

    if wage_column in df.columns:

        df["Wage_vs_State_Median"] = (
            df[wage_column]
            -
            df.groupby(state_column)[wage_column]
            .transform("median")
        )

# ==========================================
# DATA QUALITY FEATURES
# ==========================================

df["Total_Missing_Values_Per_Row"] = (
    df.isnull().sum(axis=1)
)

df["Total_Available_Values_Per_Row"] = (
    df.notnull().sum(axis=1)
)

df["Missing_Value_Percentage_Per_Row"] = (
    df["Total_Missing_Values_Per_Row"]
    /
    len(df.columns)
    *
    100
)

# ==========================================
# CLEAN INFINITE VALUES
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ==========================================
# FEATURE DICTIONARY
# ==========================================

dictionary_rows = []

for column in df.columns:

    dictionary_rows.append(
        {
            "Feature_Name": column,
            "Feature_Type": str(df[column].dtype),
            "Missing_Values": int(df[column].isnull().sum()),
            "Unique_Values": int(df[column].nunique())
        }
    )

feature_dictionary = pd.DataFrame(dictionary_rows)

# ==========================================
# NEW FEATURES
# ==========================================

new_features = [
    col for col in df.columns
    if col not in original_columns
]

print("\nNew Features Created:")

for feature in new_features:
    print("-", feature)

# ==========================================
# SAVE OUTPUT
# ==========================================

df.to_csv(
    output_file,
    index=False
)

feature_dictionary.to_csv(
    dictionary_file,
    index=False
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("="*60)

print("Original Columns:", len(original_columns))
print("Final Columns:", len(df.columns))
print("New Features Created:", len(new_features))
print("Final Shape:", df.shape)

print("\nSaved Dataset:")
print(output_file)

print("\nSaved Feature Dictionary:")
print(dictionary_file)

print("="*60)
