# ==========================================
# ECONOMIC TREND ANALYSIS
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
    r"\Economic_Trend_Analysis_Cleaned.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Intelligence_Feature_Engineering"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Economic_Trend_Feature_Engineered.csv"
)

dictionary_file = os.path.join(
    output_folder,
    "Economic_Trend_Feature_Dictionary.csv"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

original_columns = df.columns.tolist()

print("=" * 60)
print("ECONOMIC TREND FEATURE ENGINEERING")
print("=" * 60)

print("Original Dataset Shape:", df.shape)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Shape After Removing Duplicates:", df.shape)

# ==========================================
# IDENTIFY YEAR COLUMNS
# ==========================================

year_columns = [
    column
    for column in df.columns
    if column.isdigit()
]

year_columns = sorted(
    year_columns,
    key=lambda column: int(column)
)

print("\nYear Columns:")
print(year_columns)

# Convert year columns to numeric
for column in year_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ==========================================
# IDENTIFY TEXT COLUMNS
# ==========================================

country_column = "COUNTRY"
indicator_column = "INDICATOR"
unit_column = "UNIT"
scale_column = "SCALE"

print("\nAvailable Columns:")
print(df.columns.tolist())

# ==========================================
# YEAR COVERAGE FEATURES
# ==========================================

if len(year_columns) > 0:

    df["Available_Year_Count"] = (
        df[year_columns]
        .notnull()
        .sum(axis=1)
    )

df["Missing_Year_Count"] = (
        df[year_columns]
        .isnull()
        .sum(axis=1)
    )

df["Missing_Year_Percentage"] = (
        df["Missing_Year_Count"]
        / len(year_columns)
        * 100
    )

# ==========================================
# HISTORICAL STATISTICAL FEATURES
# ==========================================

if len(year_columns) > 0:

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
# FIRST AND LATEST YEAR FEATURES
# ==========================================

if len(year_columns) > 0:

    first_year_column = year_columns[0]
    latest_year_column = year_columns[-1]

first_year = int(first_year_column)
latest_year = int(latest_year_column)

df["First_Year_Value"] = (
        df[first_year_column]
    )

df["Latest_Year_Value"] = (
        df[latest_year_column]
    )

df["First_Year"] = first_year
df["Latest_Year"] = latest_year

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

# ==========================================
# LATEST YEAR CHANGE FEATURES
# ==========================================

if len(year_columns) >= 2:

    previous_year_column = year_columns[-2]
    latest_year_column = year_columns[-1]

df["Previous_Year_Value"] = (
        df[previous_year_column]
    )

df["Latest_Year_Change"] = (
        df[latest_year_column]
        - df[previous_year_column]
    )

df["Latest_Year_Change_Percentage"] = np.where(
        df[previous_year_column] != 0,
        (
            df["Latest_Year_Change"]
            / abs(df[previous_year_column])
            * 100
        ),
        np.nan
    )

# ==========================================
# CAGR FEATURE
# ==========================================

if len(year_columns) >= 2:

    number_of_years = (
        int(year_columns[-1])
        - int(year_columns[0])
    )

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
# TREND SLOPE
# ==========================================

def calculate_trend_slope(row):

    values = row[year_columns].dropna()

    if len(values) < 2:
        return np.nan

    available_years = [
        int(column)
        for column in values.index
    ]

    try:

        slope = np.polyfit(
            available_years,
            values.astype(float),
            1
        )[0]

        return slope

    except (TypeError, ValueError):

        return np.nan

if len(year_columns) > 1:

    df["Trend_Slope"] = df.apply(
        calculate_trend_slope,
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
        default="No Trend"
    )

else:

    df["Trend_Slope"] = np.nan

    df["Trend_Direction"] = "No Trend"

# ==========================================
# LATEST VALUE STATUS
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
# LOG FEATURES
# ==========================================

if "Latest_Year_Value" in df.columns:

    df["Latest_Value_Log"] = np.where(
        df["Latest_Year_Value"] >= 0,
        np.log1p(df["Latest_Year_Value"]),
        np.nan
    )

if "Historical_Average" in df.columns:

    df["Historical_Average_Log"] = np.where(
        df["Historical_Average"] >= 0,
        np.log1p(df["Historical_Average"]),
        np.nan
    )

# ==========================================
# COUNTRY FEATURES
# ==========================================

if country_column in df.columns:

    df["Country_ID"] = (
        df[country_column]
        .astype("category")
        .cat.codes
    )

df["Country_Record_Count"] = (
        df.groupby(country_column)[country_column]
        .transform("count")
    )

# ==========================================
# INDICATOR FEATURES
# ==========================================

if indicator_column in df.columns:

    df["Indicator_ID"] = (
        df[indicator_column]
        .astype("category")
        .cat.codes
    )

df["Indicator_Record_Count"] = (
        df.groupby(indicator_column)[indicator_column]
        .transform("count")
    )

# ==========================================
# UNIT FEATURES
# ==========================================

if unit_column in df.columns:

    df["Unit_ID"] = (
        df[unit_column]
        .astype("category")
        .cat.codes
    )

df["Unit_Record_Count"] = (
        df.groupby(unit_column)[unit_column]
        .transform("count")
    )

# ==========================================
# SCALE FEATURES
# ==========================================

if scale_column in df.columns:

    df["Scale_ID"] = (
        df[scale_column]
        .astype("category")
        .cat.codes
    )

# ==========================================
# COUNTRY-INDICATOR GROUP FEATURES
# ==========================================

if (
    country_column in df.columns
    and indicator_column in df.columns
):

    df["Country_Indicator_Record_Count"] = (
        df.groupby(
            [
                country_column,
                indicator_column
            ]
        )[indicator_column]
        .transform("count")
    )

# ==========================================
# OUTLIER FLAG FOR LATEST YEAR
# ==========================================

def create_outlier_flag(
    dataframe,
    column,
    new_column
):

    if column not in dataframe.columns:
        return

    values = dataframe[column].dropna()

    if values.empty:
        dataframe[new_column] = 0
        return

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    dataframe[new_column] = np.where(
        (
            (dataframe[column] < lower_limit)
            | (dataframe[column] > upper_limit)
        ),
        1,
        0
    )

    dataframe.loc[
        dataframe[column].isnull(),
        new_column
    ] = 0

if "Latest_Year_Value" in df.columns:

    create_outlier_flag(
        df,
        "Latest_Year_Value",
        "Latest_Value_Outlier_Flag"
    )

if "Historical_Average" in df.columns:

    create_outlier_flag(
        df,
        "Historical_Average",
        "Historical_Average_Outlier_Flag"
    )

# ==========================================
# DATA QUALITY FEATURES
# ==========================================

df["Total_Missing_Values_Per_Row"] = (
    df.isnull()
    .sum(axis=1)
)

df["Total_Available_Values_Per_Row"] = (
    df.notnull()
    .sum(axis=1)
)

df["Missing_Value_Percentage_Per_Row"] = (
    df["Total_Missing_Values_Per_Row"]
    / len(df.columns)
    * 100
)

# ==========================================
# REPLACE INFINITE VALUES
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ==========================================
# IDENTIFY NEW FEATURES
# ==========================================

new_features = [
    column
    for column in df.columns
    if column not in original_columns
]

print("\n" + "=" * 60)
print("NEW FEATURES CREATED")
print("=" * 60)

for feature in new_features:
    print("-", feature)

# ==========================================
# FEATURE DESCRIPTIONS
# ==========================================

feature_descriptions = {
    "Available_Year_Count":
        "Number of years with available values",

"Missing_Year_Count":
        "Number of missing year values",

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
        "Standard deviation across available yearly values",

"Historical_Range":
        "Difference between historical maximum and minimum",

"First_Year_Value":
        "Value recorded in the earliest year",

"Latest_Year_Value":
        "Value recorded in the latest year",

"First_Year":
        "Earliest year in the dataset",

"Latest_Year":
        "Latest year in the dataset",

"Absolute_Change":
        "Difference between latest and first year values",

"Percentage_Change":
        "Percentage change between latest and first year",

"Previous_Year_Value":
        "Value recorded in the year before the latest year",

"Latest_Year_Change":
        "Change between the latest year and previous year",

"Latest_Year_Change_Percentage":
        "Percentage change between latest and previous year",

"CAGR_Percentage":
        "Compound annual growth rate",

"Trend_Slope":
        "Linear trend slope across all available years",

"Trend_Direction":
        "Increasing, decreasing, stable, or insufficient data",

"Latest_Value_Status":
        "Positive, negative, zero, or missing latest value",

"Latest_Value_Log":
        "Log-transformed latest year value",

"Historical_Average_Log":
        "Log-transformed historical average",

"Country_ID":
        "Numeric identifier assigned to each country",

"Country_Record_Count":
        "Number of records associated with the country",

"Indicator_ID":
        "Numeric identifier assigned to each indicator",

"Indicator_Record_Count":
        "Number of records associated with the indicator",

"Unit_ID":
        "Numeric identifier assigned to each unit",

"Unit_Record_Count":
        "Number of records associated with the unit",

"Scale_ID":
        "Numeric identifier assigned to each scale",

"Country_Indicator_Record_Count":
        "Number of records for a country-indicator combination",

"Latest_Value_Outlier_Flag":
        "IQR-based outlier flag for latest year value",

"Historical_Average_Outlier_Flag":
        "IQR-based outlier flag for historical average",

"Total_Missing_Values_Per_Row":
        "Total missing values in each row",

"Total_Available_Values_Per_Row":
        "Total available values in each row",

"Missing_Value_Percentage_Per_Row":
        "Percentage of missing values in each row"
}

# ==========================================
# CREATE FEATURE DICTIONARY
# ==========================================

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
            "Missing_Values": int(
                df[column].isnull().sum()
            ),
            "Unique_Values": int(
                df[column].nunique()
            )
        }
    )

feature_dictionary = pd.DataFrame(
    dictionary_rows
)

# ==========================================
# SAVE FILES
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
print("Final Dataset Shape:", df.shape)

print("\nFeature-Engineered Dataset Saved At:")
print(output_file)

print("\nFeature Dictionary Saved At:")
print(dictionary_file)

print("=" * 60)

