# ==========================================
# SMART CITY INTELLIGENCE - COMPLETE EDA
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==========================================
# FILE PATHS
# ==========================================

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task\Traffic Analysis_Final.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_EDA"
)

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# SAVE PLOT FUNCTION
# ==========================================

def save_plot(filename):

    file_path = os.path.join(
        output_folder,
        filename
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved:", file_path)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

print("=" * 60)
print("TRAFFIC ANALYSIS DATASET")
print("=" * 60)

print("Dataset Shape:", df.shape)

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDATASET INFORMATION")

df.info()

print("\nDATA TYPES")

print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMISSING VALUES")

missing = df.isnull().sum()

print(
    missing[missing > 0]
)

print("\nMISSING VALUE PERCENTAGE")

missing_percent = (
    df.isnull().sum() / len(df)
) * 100

print(
    missing_percent[
        missing_percent > 0
    ].sort_values(ascending=False)
)

# ==========================================
# DUPLICATES
# ==========================================

print(
    "\nDUPLICATE ROWS:",
    df.duplicated().sum()
)

df = df.drop_duplicates().copy()

print(
    "Shape After Removing Duplicates:",
    df.shape
)

# ==========================================
# DATETIME CONVERSION
# ==========================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

print(
    "\nInvalid Timestamp Rows:",
    df["Timestamp"].isnull().sum()
)

df["Date"] = df["Timestamp"].dt.date

df["Hour"] = df["Timestamp"].dt.hour

df["Day"] = df["Timestamp"].dt.day_name()

print("\nFIRST 5 ROWS")

print(df.head())

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\nDESCRIPTIVE STATISTICS")

print(
    df.describe(include="all")
)

print("\nCOLUMN NAMES")

print(
    df.columns.tolist()
)

# ==========================================
# NUMERIC FEATURES
# ==========================================

numeric_cols = df.select_dtypes(
    include="number"
).columns

print("\nNUMERIC COLUMNS")

print(
    numeric_cols.tolist()
)

# ==========================================
# 1. TRAFFIC CONDITION
# ==========================================

plt.figure(figsize=(7,5))

sns.countplot(
    x="Traffic_Condition",
    data=df
)

plt.title(
    "Traffic Condition Distribution"
)

plt.xlabel(
    "Traffic Condition"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

save_plot(
    "01_Traffic_Condition_Distribution.png"
)

# ==========================================
# 2. WEATHER CONDITION
# ==========================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Weather_Condition",
    data=df
)

plt.title(
    "Weather Conditions"
)

plt.xlabel(
    "Weather Condition"
)

plt.ylabel(
    "Count"
)

plt.xticks(rotation=45)

plt.tight_layout()

save_plot(
    "02_Weather_Condition_Distribution.png"
)

# ==========================================
# 3. TRAFFIC LIGHT STATE
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="Traffic_Light_State",
    data=df
)

plt.title(
    "Traffic Light States"
)

plt.xlabel(
    "Traffic Light State"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

save_plot(
    "03_Traffic_Light_State_Distribution.png"
)

# ==========================================
# 4. ACCIDENT REPORT
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="Accident_Report",
    data=df
)

plt.title(
    "Accident Reports"
)

plt.xlabel(
    "Accident Report"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

save_plot(
    "04_Accident_Report_Distribution.png"
)

# ==========================================
# 5. NUMERIC FEATURES DISTRIBUTION
# ==========================================

df[numeric_cols].hist(
    figsize=(18,14),
    bins=30,
    edgecolor="black"
)

plt.suptitle(
    "Numeric Features Distribution",
    fontsize=16
)

plt.tight_layout()

save_plot(
    "05_Numeric_Features_Distribution.png"
)

# ==========================================
# 6. VEHICLE COUNT DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Vehicle_Count"],
    bins=30,
    kde=True
)

plt.title(
    "Vehicle Count Distribution"
)

plt.xlabel(
    "Vehicle Count"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

save_plot(
    "06_Vehicle_Count_Distribution.png"
)

# ==========================================
# 7. TRAFFIC SPEED DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Traffic_Speed_kmh"],
    bins=30,
    kde=True
)

plt.title(
    "Traffic Speed Distribution"
)

plt.xlabel(
    "Traffic Speed (km/h)"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

save_plot(
    "07_Traffic_Speed_Distribution.png"
)

# ==========================================
# 8. VEHICLE COUNT BOXPLOT
# ==========================================

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df["Vehicle_Count"]
)

plt.title(
    "Vehicle Count Outliers"
)

plt.xlabel(
    "Vehicle Count"
)

plt.tight_layout()

save_plot(
    "08_Vehicle_Count_Boxplot.png"
)

# ==========================================
# 9. CORRELATION HEATMAP
# ==========================================

corr = df[numeric_cols].corr()

plt.figure(figsize=(12,10))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap"
)

plt.xticks(rotation=45)

plt.tight_layout()

save_plot(
    "09_Correlation_Heatmap.png"
)

# ==========================================
# 10. VEHICLE COUNT VS SPEED
# ==========================================

plt.figure(figsize=(7,5))

sns.scatterplot(
    x="Vehicle_Count",
    y="Traffic_Speed_kmh",
    data=df,
    alpha=0.7
)

plt.title(
    "Vehicle Count vs Traffic Speed"
)

plt.xlabel(
    "Vehicle Count"
)

plt.ylabel(
    "Traffic Speed (km/h)"
)

plt.tight_layout()

save_plot(
    "10_Vehicle_Count_vs_Speed.png"
)

# ==========================================
# 11. ROAD OCCUPANCY VS VEHICLE COUNT
# ==========================================

plt.figure(figsize=(7,5))

sns.scatterplot(
    x="Road_Occupancy_%",
    y="Vehicle_Count",
    data=df,
    alpha=0.7
)

plt.title(
    "Road Occupancy vs Vehicle Count"
)

plt.xlabel(
    "Road Occupancy (%)"
)

plt.ylabel(
    "Vehicle Count"
)

plt.tight_layout()

save_plot(
    "11_Road_Occupancy_vs_Vehicle_Count.png"
)

# ==========================================
# 12. HOURLY TRAFFIC
# ==========================================

hourly_traffic = (
    df.groupby("Hour")
    ["Vehicle_Count"]
    .mean()
)

plt.figure(figsize=(10,5))

hourly_traffic.plot(
    marker="o"
)

plt.title(
    "Average Vehicle Count by Hour"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Average Vehicle Count"
)

plt.xticks(
    range(0,24)
)

plt.grid(True)

plt.tight_layout()

save_plot(
    "12_Average_Vehicle_Count_by_Hour.png"
)

# ==========================================
# 13. TRAFFIC SPEED BY WEATHER
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Weather_Condition",
    y="Traffic_Speed_kmh",
    data=df
)

plt.title(
    "Traffic Speed by Weather Condition"
)

plt.xlabel(
    "Weather Condition"
)

plt.ylabel(
    "Traffic Speed (km/h)"
)

plt.xticks(rotation=45)

plt.tight_layout()

save_plot(
    "13_Traffic_Speed_by_Weather.png"
)

# ==========================================
# SMART CITY ADDITIONAL ANALYSIS
# ==========================================

# Peak Traffic Hours

peak_hours = (
    df.groupby("Hour")
    ["Vehicle_Count"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTOP PEAK TRAFFIC HOURS")

print(
    peak_hours.head()
)

# Traffic Condition vs Speed

traffic_speed = (
    df.groupby("Traffic_Condition")
    ["Traffic_Speed_kmh"]
    .mean()
)

print("\nAVERAGE SPEED BY TRAFFIC CONDITION")

print(
    traffic_speed
)

# Accident Impact

accident_speed = (
    df.groupby("Accident_Report")
    ["Traffic_Speed_kmh"]
    .mean()
)

print("\nACCIDENT IMPACT ON SPEED")

print(
    accident_speed
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)

print(
    "EDA COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print(
    "Final Dataset Shape:",
    df.shape
)

print(
    "Total Records:",
    len(df)
)

print(
    "Total Features:",
    len(df.columns)
)

print(
    "Numeric Features:",
    len(numeric_cols)
)

print(
    "\nAll visualizations saved at:"
)

print(
    os.path.abspath(output_folder)
)

print("=" * 60)
