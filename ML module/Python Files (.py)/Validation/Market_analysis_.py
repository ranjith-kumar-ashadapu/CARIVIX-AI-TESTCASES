# ==========================================
# MARKET ANALYSIS - FEATURE ENGINEERING
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import re
import numpy as np
import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Market_Analysis_1981_2025_Final_Cleaned.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Market_Analysis_Feature_Engineering"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Market_Analysis_Feature_Engineered.csv"
)

dictionary_file = os.path.join(
    output_folder,
    "Market_Analysis_Feature_Dictionary.csv"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

print("=" * 60)
print("MARKET ANALYSIS FEATURE ENGINEERING")
print("=" * 60)

print("Original Dataset Shape:", df.shape)

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.astype(str).str.strip()

# ==========================================
# IDENTIFY YEAR COLUMNS
# ==========================================

year_columns = []

for column in df.columns:
    match = re.search(r"(19|20)\d{2}", column)

    if match:
        year_columns.append(column)

print("\nYear Columns Found:")
print(year_columns)

# Convert year columns to numeric
for column in year_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Sort year columns chronologically
year_columns = sorted(
    year_columns,
    key=lambda column: int(
        re.search(r"(19|20)\d{2}", column).group()
    )
)

# ==========================================
# IDENTIFY IMPORTANT COLUMNS
# ==========================================

country_column = None
indicator_column = None

for column in df.columns:

    column_lower = column.lower()

if column_lower == "country name":
        country_column = column

if column_lower == "series name":
        indicator_column = column

print("\nCountry Column:", country_column)
print("Indicator Column:", indicator_column)

# ==========================================
# MISSING VALUE FEATURES
# ==========================================

if year_columns:

    df["Missing_Year_Count"] = (
        df[year_columns]
        .isnull()
        .sum(axis=1)
    )

    df["Available_Year_Count"] = (
        df[year_columns]
        .notnull()
        .sum(axis=1)
    )

    df["Missing_Year_Percentage"] = (
        df["Missing_Year_Count"]
        / len(year_columns)
        * 100
    )

# ==========================================
# STATISTICAL FEATURES
# ==========================================

if year_columns:

    df["Historical_Average"] = (
        df[year_columns]
        .mean(axis=1)
    )

    df["Historical_Median"] = (
        df[year_columns]
        .median(axis=1)
    )

    df["Historical_Minimum"] = (
        df[year_columns]
        .min(axis=1)
    )

    df["Historical_Maximum"] = (
        df[year_columns]
        .max(axis=1)
    )

    df["Historical_Standard_Deviation"] = (
        df[year_columns]
        .std(axis=1)
    )

    df["Historical_Range"] = (
        df["Historical_Maximum"]
        - df["Historical_Minimum"]
    )

# ==========================================
# FIRST AND LATEST VALUE FEATURES
# ==========================================

if year_columns:

    first_year_column = year_columns[0]
    latest_year_column = year_columns[-1]

    first_year = int(
        re.search(
            r"(19|20)\d{2}",
            first_year_column
        ).group()
    )

    latest_year = int(
        re.search(
            r"(19|20)\d{2}",
            latest_year_column
        ).group()
    )

    df["First_Year_Value"] = df[first_year_column]
    df["Latest_Year_Value"] = df[latest_year_column]

    df["Absolute_Change"] = (
        df["Latest_Year_Value"]
        - df["First_Year_Value"]
    )

    df["Percentage_Change"] = np.where(
        df["First_Year_Value"] != 0,
        (
            df["Absolute_Change"]
            / abs(df["First_Year_Value"])
            * 100
        ),
        np.nan
    )

    number_of_years = latest_year - first_year

if number_of_years > 0:

    df["CAGR_Percentage"] = np.where(
            (
                (df["First_Year_Value"] > 0)
                & (df["Latest_Year_Value"] > 0)
            ),
            (
                (
                    df["Latest_Year_Value"]
                    / df["First_Year_Value"]
                )
                ** (1 / number_of_years)
                - 1
            )
            * 100,
            np.nan
        )

# ==========================================
# TREND FEATURES
# ==========================================

def calculate_trend(row):

    values = row[year_columns].dropna()

    if len(values) < 2:
        return np.nan

    years = []

    for column in values.index:

        year_match = re.search(
            r"(19|20)\d{2}",
            column
        )

        if year_match:
            years.append(
                int(year_match.group())
            )

    if len(years) != len(values):
        return np.nan

    try:

        slope = np.polyfit(
            years,
            values.astype(float),
            1
        )[0]

        return slope

    except (TypeError, ValueError):
        return np.nan
if year_columns:

    df["Trend_Slope"] = df.apply(
        calculate_trend,
        axis=1
    )

    df["Trend_Direction"] = np.select(
        [
            df["Trend_Slope"] > 0,
            df["Trend_Slope"] < 0,
            df["Trend_Slope"] == 0
        ],
        [
            "Increasing",
            "Decreasing",
            "Stable"
        ],
        default="Insufficient Data"
    )

# ==========================================
# LATEST VALUE CATEGORY
# ==========================================

if "Latest_Year_Value" in df.columns:

    df["Latest_Value_Status"] = np.select(
        [
            df["Latest_Year_Value"] > 0,
            df["Latest_Year_Value"] < 0,
            df["Latest_Year_Value"] == 0
        ],
        [
            "Positive",
            "Negative",
            "Zero"
        ],
        default="Missing"
    )

# ==========================================
# COUNTRY FEATURES
# ==========================================

if country_column is not None:

    df["Country_Record_Count"] = (
        df.groupby(country_column)[country_column]
        .transform("count")
    )

    df["Country_ID"] = (
        df[country_column]
        .astype("category")
        .cat.codes
    )

# ==========================================
# ECONOMIC INDICATOR FEATURES
# ==========================================

if indicator_column is not None:

    df["Indicator_Record_Count"] = (
        df.groupby(indicator_column)[indicator_column]
        .transform("count")
    )

    df["Indicator_ID"] = (
        df[indicator_column]
        .astype("category")
        .cat.codes
    )

# ==========================================
# VALUE MAGNITUDE FEATURES
# ==========================================

if "Latest_Year_Value" in df.columns:

    df["Latest_Value_Log"] = np.where(
        df["Latest_Year_Value"] >= 0,
        np.log1p(df["Latest_Year_Value"]),
        np.nan
    )

if "Historical_Average" in df.columns:

    df["Average_Value_Log"] = np.where(
        df["Historical_Average"] >= 0,
        np.log1p(df["Historical_Average"]),
        np.nan
    )

# ==========================================
# YEAR-TO-YEAR CHANGE FEATURES
# ==========================================

if len(year_columns) >= 2:

    previous_column = year_columns[-2]
    latest_column = year_columns[-1]

    df["Latest_Year_Change"] = (
        df[latest_column]
        - df[previous_column]
    )

    df["Latest_Year_Change_Percentage"] = np.where(
        df[previous_column] != 0,
        (
            df["Latest_Year_Change"]
            / abs(df[previous_column])
            * 100
        ),
        np.nan
    )

# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows After Feature Engineering:")
print(duplicate_count)

df = df.drop_duplicates().copy()

# ==========================================
# REPLACE INFINITE VALUES
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ==========================================
# DISPLAY ENGINEERED DATA
# ==========================================

print("\nEngineered Dataset Shape:", df.shape)

print("\nNew Feature Columns:")

original_columns = pd.read_csv(input_file).columns.tolist()
new_features = [
    column
    for column in df.columns
    if column not in original_columns
]

print(new_features)

print("\nFirst 5 Engineered Records:")
print(df.head())

# ==========================================
# CREATE FEATURE DICTIONARY
# ==========================================

feature_descriptions = {
    "Missing_Year_Count":
        "Number of missing values across all year columns",

"Available_Year_Count":
        "Number of available values across all year columns",

"Missing_Year_Percentage":
        "Percentage of missing year values",

"Historical_Average":
        "Average value across all available years",

"Historical_Median":
        "Median value across all available years",

"Historical_Minimum":
        "Minimum value across all available years",

"Historical_Maximum":
        "Maximum value across all available years",

"Historical_Standard_Deviation":
        "Standard deviation across all available years",

"Historical_Range":
        "Difference between historical maximum and minimum",

"First_Year_Value":
        "Value recorded in the earliest available year column",

"Latest_Year_Value":
        "Value recorded in the latest available year column",

"Absolute_Change":
        "Difference between latest year and first year value",

"Percentage_Change":
        "Percentage change between first and latest year value",

"CAGR_Percentage":
        "Compound annual growth rate between first and latest year",

"Trend_Slope":
        "Linear trend slope calculated across the year values",

"Trend_Direction":
        "Indicates whether the trend is increasing, decreasing, or stable",

"Latest_Value_Status":
        "Indicates whether the latest value is positive, negative, zero, or missing",

"Country_Record_Count":
        "Number of records associated with the country",

"Country_ID":
        "Numeric identifier assigned to each country",

"Indicator_Record_Count":
        "Number of records associated with the economic indicator",

"Indicator_ID":
        "Numeric identifier assigned to each economic indicator",

"Latest_Value_Log":
        "Log-transformed latest year value",

"Average_Value_Log":
        "Log-transformed historical average value",

"Latest_Year_Change":
        "Change between the latest year and previous year",

"Latest_Year_Change_Percentage":
        "Percentage change between the latest year and previous year"
}

dictionary_rows = []

for column in df.columns:

    if column in feature_descriptions:

        description = feature_descriptions[column]

else:

    description = "Original dataset column"

dictionary_rows.append(
        {
            "Feature_Name": column,
            "Feature_Type": str(df[column].dtype),
            "Description": description,
            "Missing_Values": int(df[column].isnull().sum()),
            "Unique_Values": int(df[column].nunique())
        }
    )

feature_dictionary = pd.DataFrame(
    dictionary_rows
)

# ==========================================
# SAVE OUTPUT FILES
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

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Original Number of Columns:", len(original_columns))
print("Final Number of Columns:", len(df.columns))
print("New Features Created:", len(new_features))

print("\nFeature-Engineered Dataset Saved At:")
print(output_file)

print("\nFeature Dictionary Saved At:")
print(dictionary_file)

print("=" * 60)
