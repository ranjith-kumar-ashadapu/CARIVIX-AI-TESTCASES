# ==========================================
# SMART CITY INTELLIGENCE
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
    r"\Traffic Analysis_Final.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Feature_Engineering"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Smart_City_Traffic_Feature_Engineered.csv"
)

dictionary_file = os.path.join(
    output_folder,
    "Smart_City_Feature_Dictionary.csv"
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
print("SMART CITY INTELLIGENCE FEATURE ENGINEERING")
print("=" * 60)

print(
    "Original Dataset Shape:",
    df.shape
)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print(
    "\nDuplicate Rows:",
    df.duplicated().sum()
)

df = (
    df
    .drop_duplicates()
    .copy()
)

print(
    "Shape After Removing Duplicates:",
    df.shape
)

# ==========================================
# TIMESTAMP CONVERSION
# ==========================================

timestamp_column = "Timestamp"

if timestamp_column in df.columns:

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    print(
        "\nInvalid Timestamp Rows:",
        df[timestamp_column].isnull().sum()
    )

else:

    print(
        "\nTimestamp column not found"
    )

# ==========================================
# NUMERIC COLUMN CONVERSION
# ==========================================

numeric_columns = [

    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_%"

]

for column in numeric_columns:

    if column in df.columns:

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

# ==========================================
# DATETIME FEATURES
# ==========================================

if timestamp_column in df.columns:

    df["Date"] = (
        df[timestamp_column]
        .dt.date
    )

    df["Year"] = (
        df[timestamp_column]
        .dt.year
    )

    df["Month"] = (
        df[timestamp_column]
        .dt.month
    )

    df["Month_Name"] = (
        df[timestamp_column]
        .dt.month_name()
    )

    df["Day"] = (
        df[timestamp_column]
        .dt.day
    )

    df["Day_of_Week"] = (
        df[timestamp_column]
        .dt.dayofweek
    )

    df["Day_Name"] = (
        df[timestamp_column]
        .dt.day_name()
    )

    df["Week_of_Year"] = (
        df[timestamp_column]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )

    df["Quarter"] = (
        df[timestamp_column]
        .dt.quarter
    )

    df["Hour"] = (
        df[timestamp_column]
        .dt.hour
    )

    df["Minute"] = (
        df[timestamp_column]
        .dt.minute
    )

    df["Is_Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

# ==========================================
# TIME PERIOD FEATURES
# ==========================================

if "Hour" in df.columns:

    df["Time_Period"] = np.select(

        [

            df["Hour"].between(5,11),

            df["Hour"].between(12,16),

            df["Hour"].between(17,20),

            df["Hour"].between(21,23),

            df["Hour"].between(0,4)

        ],

        [

            "Morning",

            "Afternoon",

            "Evening",

            "Night",

            "Late Night"

        ],

        default="Unknown"

    )

    df["Is_Morning_Peak"] = (
        df["Hour"].between(7,10)
    ).astype(int)

    df["Is_Evening_Peak"] = (
        df["Hour"].between(17,20)
    ).astype(int)

    df["Is_Peak_Hour"] = (

        df["Is_Morning_Peak"].eq(1)

        |

        df["Is_Evening_Peak"].eq(1)

    ).astype(int)

    df["Is_Off_Peak_Hour"] = (
        df["Is_Peak_Hour"] == 0
    ).astype(int)

# Replace infinite values early

df = df.replace(
    [np.inf,-np.inf],
    np.nan
)

# ==========================================
# VEHICLE COUNT FEATURES
# ==========================================

vehicle_column = "Vehicle_Count"

if vehicle_column in df.columns:

    df["Vehicle_Count_Log"] = np.where(

        df[vehicle_column] >= 0,

        np.log1p(df[vehicle_column]),

        np.nan

    )

    vehicle_median = (
        df[vehicle_column]
        .median()
    )

    df["Vehicle_Count_Category"] = np.select(

        [

            df[vehicle_column] < vehicle_median,

            df[vehicle_column] >= vehicle_median

        ],

        [

            "Below Median",

            "At or Above Median"

        ],

        default="Missing"

    )

# ==========================================
# TRAFFIC SPEED FEATURES
# ==========================================

speed_column = "Traffic_Speed_kmh"

if speed_column in df.columns:

    df["Traffic_Speed_Log"] = np.where(

        df[speed_column] >= 0,

        np.log1p(df[speed_column]),

        np.nan

    )

    df["Speed_Category"] = np.select(

        [

            df[speed_column] < 20,

            df[speed_column].between(20,40),

            df[speed_column].between(40,60),

            df[speed_column] > 60

        ],

        [

            "Very Slow",

            "Slow",

            "Moderate",

            "Fast"

        ],

        default="Missing"

    )

# ==========================================
# ROAD OCCUPANCY FEATURES
# ==========================================

occupancy_column = "Road_Occupancy_%"

if occupancy_column in df.columns:

    df["Road_Occupancy_Log"] = np.where(

        df[occupancy_column] >= 0,

        np.log1p(df[occupancy_column]),

        np.nan

    )

    df["Occupancy_Category"] = np.select(

        [

            df[occupancy_column] < 25,

            df[occupancy_column].between(25,50),

            df[occupancy_column].between(50,75),

            df[occupancy_column] > 75

        ],

        [

            "Low Occupancy",

            "Moderate Occupancy",

            "High Occupancy",

            "Very High Occupancy"

        ],

        default="Missing"

    )

# ==========================================
# TRAFFIC DENSITY FEATURES
# ==========================================

if (

    vehicle_column in df.columns

    and

    occupancy_column in df.columns

):

    df["Vehicle_Count_per_Occupancy"] = np.where(

        df[occupancy_column] > 0,

        df[vehicle_column] /
        df[occupancy_column],

        np.nan

    )

    df["Occupancy_per_Vehicle"] = np.where(

        df[vehicle_column] > 0,

        df[occupancy_column] /
        df[vehicle_column],

        np.nan

    )

# ==========================================
# TRAFFIC CONGESTION FEATURES
# ==========================================

if (

    vehicle_column in df.columns

    and

    speed_column in df.columns

):

    vehicle_median = (
        df[vehicle_column]
        .median()
    )

    speed_median = (
        df[speed_column]
        .median()
    )

    df["High_Traffic_Flag"] = (

        df[vehicle_column]
        >= vehicle_median

    ).astype(int)

    df["Low_Speed_Flag"] = (

        df[speed_column]
        < speed_median

    ).astype(int)

    df["Congestion_Risk_Flag"] = (

        (

            df["High_Traffic_Flag"] == 1

        )

        &

        (

            df["Low_Speed_Flag"] == 1

        )

    ).astype(int)

# ==========================================
# CATEGORICAL ENCODING
# ==========================================

categorical_columns = [

    "Traffic_Condition",

    "Weather_Condition",

    "Traffic_Light_State",

    "Accident_Report",

    "Time_Period",

    "Speed_Category",

    "Occupancy_Category",

    "Vehicle_Count_Category"

]

for column in categorical_columns:

    if column in df.columns:

        df[column + "_ID"] = (

            df[column]
            .astype("category")
            .cat.codes

        )

# ==========================================
# GROUP BASED FEATURES
# ==========================================

if (

    "Weather_Condition" in df.columns

    and

    speed_column in df.columns

):

    df["Average_Speed_by_Weather"] = (

        df.groupby("Weather_Condition")[speed_column]
        .transform("mean")

    )

    df["Speed_vs_Weather_Average"] = (

        df[speed_column]

        -

        df["Average_Speed_by_Weather"]

    )

if (

    "Traffic_Condition" in df.columns

    and

    vehicle_column in df.columns

):

    df["Average_Vehicles_by_Traffic_Condition"] = (

        df.groupby("Traffic_Condition")[vehicle_column]
        .transform("mean")

    )

    df["Vehicles_vs_Traffic_Condition_Average"] = (

        df[vehicle_column]

        -

        df["Average_Vehicles_by_Traffic_Condition"]

    )

if (

    "Hour" in df.columns

    and

    vehicle_column in df.columns

):

    df["Average_Vehicles_by_Hour"] = (

        df.groupby("Hour")[vehicle_column]
        .transform("mean")

    )

    df["Vehicles_vs_Hour_Average"] = (

        df[vehicle_column]

        -

        df["Average_Vehicles_by_Hour"]

    )

# ==========================================
# OUTLIER FLAG FUNCTION
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

            dataframe[column] < lower_limit

        )

        |

        (

            dataframe[column] > upper_limit

        ),

        1,

        0

    )

    dataframe.loc[

        dataframe[column].isnull(),

        new_column

    ] = 0

create_outlier_flag(

    df,

    vehicle_column,

    "Vehicle_Count_Outlier_Flag"

)

create_outlier_flag(

    df,

    speed_column,

    "Traffic_Speed_Outlier_Flag"

)

create_outlier_flag(

    df,

    occupancy_column,

    "Road_Occupancy_Outlier_Flag"

)
# ==========================================
# MISSING VALUE FEATURES
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

    /

    len(df.columns)

    *

    100

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
# FEATURE DICTIONARY
# ==========================================

feature_descriptions = {

"Date":
"Date extracted from timestamp",

"Year":
"Year extracted from timestamp",

"Month":
"Month number extracted from timestamp",

"Month_Name":
"Month name extracted from timestamp",

"Day":
"Day of month extracted from timestamp",

"Day_of_Week":
"Numerical day of week",

"Day_Name":
"Name of day",

"Week_of_Year":
"Week number of year",

"Quarter":
"Quarter extracted from timestamp",

"Hour":
"Hour extracted from timestamp",

"Minute":
"Minute extracted from timestamp",

"Is_Weekend":
"Weekend indicator flag",

"Time_Period":
"Traffic period category based on hour",

"Is_Morning_Peak":
"Morning peak hour indicator (7 AM - 10 AM)",

"Is_Evening_Peak":
"Evening peak hour indicator (5 PM - 8 PM)",

"Is_Peak_Hour":
"Peak traffic hour indicator",

"Is_Off_Peak_Hour":
"Non-peak hour indicator",

"Vehicle_Count_Log":
"Log transformed vehicle count",

"Vehicle_Count_Category":
"Vehicle count category based on median",

"Traffic_Speed_Log":
"Log transformed traffic speed",

"Speed_Category":
"Traffic speed classification",

"Road_Occupancy_Log":
"Log transformed road occupancy",

"Occupancy_Category":
"Road occupancy classification",

"Vehicle_Count_per_Occupancy":
"Vehicle count divided by occupancy percentage",

"Occupancy_per_Vehicle":
"Occupancy percentage divided by vehicle count",

"High_Traffic_Flag":
"High vehicle count indicator",

"Low_Speed_Flag":
"Low speed indicator",

"Congestion_Risk_Flag":
"High traffic combined with low speed indicator",

"Average_Speed_by_Weather":
"Average traffic speed by weather condition",

"Speed_vs_Weather_Average":
"Difference between speed and weather average speed",

"Average_Vehicles_by_Traffic_Condition":
"Average vehicles by traffic condition",

"Vehicles_vs_Traffic_Condition_Average":
"Difference from traffic condition vehicle average",

"Average_Vehicles_by_Hour":
"Average vehicle count by hour",

"Vehicles_vs_Hour_Average":
"Difference from hourly average vehicle count",

"Vehicle_Count_Outlier_Flag":
"IQR based vehicle count outlier flag",

"Traffic_Speed_Outlier_Flag":
"IQR based traffic speed outlier flag",

"Road_Occupancy_Outlier_Flag":
"IQR based road occupancy outlier flag",

"Total_Missing_Values_Per_Row":
"Total missing values present in each row",

"Total_Available_Values_Per_Row":
"Total available values present in each row",

"Missing_Value_Percentage_Per_Row":
"Percentage of missing values per row"

}

dictionary_rows = []

for column in df.columns:

    if column in feature_descriptions:

        description = feature_descriptions[column]

    else:

        description = "Original dataset column"

    dictionary_rows.append(

        {

            "Feature_Name":
            column,

            "Feature_Type":
            str(df[column].dtype),

            "Description":
            description,

            "Missing_Values":
            int(df[column].isnull().sum()),

            "Unique_Values":
            int(df[column].nunique())

        }

    )

feature_dictionary = pd.DataFrame(
    dictionary_rows
)

# ==========================================
# SAVE FEATURE ENGINEERED DATASET
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

print(

    "Original Number of Columns:",

    len(original_columns)

)

print(

    "Final Number of Columns:",

    len(df.columns)

)

print(

    "New Features Created:",

    len(new_features)

)

print(

    "Final Dataset Shape:",

    df.shape

)

print("\nFeature Engineered Dataset Saved At:")

print(output_file)

print("\nFeature Dictionary Saved At:")

print(dictionary_file)

print("=" * 60)
