from pathlib import Path
import json
import re
import sys

import pandas as pd

# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_PATH = Path(
    r"D:\OneDrive\Desktop\Day 3 Task\Baseline_Models"
)

OUTPUT_DIR = BASE_PATH / "Model_Performance_Comparison"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. DATASET AND MODEL INFORMATION
# ============================================================

DATASET_CONFIG = {

    "market_analysis": {
        "dataset": "Market Analysis",
        "task": "Regression",
        "model": "Random Forest Regressor",
    },

    "public_program": {
        "dataset": "Public Program Evaluation",
        "task": "Regression",
        "model": "Random Forest Regressor",
    },

    "economic_trend": {
        "dataset": "Economic Trend",
        "task": "Regression",
        "model": "Random Forest Regressor",
    },

    "smart_city": {
        "dataset": "Smart City Intelligence",
        "task": "Classification",
        "model": "Random Forest Classifier",
    },

    "research_intelligence": {
        "dataset": "Research Intelligence",
        "task": "Classification",
        "model": "TF-IDF + Logistic Regression",
    },
}

METRIC_COLUMNS = [

    "R2",
    "MAE",
    "RMSE",

    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",

]

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_text(value):

    value = str(value).strip().lower()

    value = (
        value
        .replace("%", "percent")
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value
    )

    return value.strip("_")

def normalize_metric_name(value):

    name = normalize_text(value)

    aliases = {

        "r2": "R2",
        "r_2": "R2",
        "r_squared": "R2",
        "r2_score": "R2",

        "mae": "MAE",
        "mean_absolute_error": "MAE",

        "rmse": "RMSE",
        "root_mean_squared_error": "RMSE",

        "accuracy": "Accuracy",
        "accuracy_score": "Accuracy",
        "accuracy_percent": "Accuracy",

        "precision": "Precision",
        "precision_score": "Precision",

        "recall": "Recall",
        "recall_score": "Recall",

        "f1": "F1_Score",
        "f1_score": "F1_Score",
        "f1_macro": "F1_Score",
        "macro_f1": "F1_Score",
        "macro_f1_score": "F1_Score",

    }

    return aliases.get(name)

def convert_to_number(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):

        if pd.isna(value):
            return None

        return float(value)

    text = str(value).strip()

    if text.lower() in {

        "nan",
        "none",
        "null",
        "na",
        "n/a"

    }:

        return None

    text = (
        text
        .replace(",", "")
        .replace("%", "")
    )

    match = re.search(

        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",

        text

    )

    if not match:
        return None

    try:

        return float(match.group())

    except ValueError:

        return None

def identify_dataset(file_path):

    searchable_text = normalize_text(
        str(file_path)
    )

    priority = [

        "smart_city",
        "research_intelligence",
        "public_program",
        "economic_trend",
        "market_analysis",

    ]

    for key in priority:

        if key in searchable_text:

            return DATASET_CONFIG[key]

    for key, value in DATASET_CONFIG.items():

        if key in searchable_text:

            return value

    return None

# ============================================================
# 4. METRIC EXTRACTION FUNCTIONS
# ============================================================

def extract_metrics_from_dataframe(dataframe):
    """
    Extract metrics from CSV/DataFrame formats.

    Supported formats:

    Format 1:
        Metric,Value
        R2,0.85
        MAE,10.2

    Format 2:
        R2,MAE,RMSE
        0.85,10.2,5.4

    """

    metrics = {
        metric: None
        for metric in METRIC_COLUMNS
    }

    if dataframe.empty:

        return metrics

    dataframe = dataframe.copy()

    dataframe.columns = [

        normalize_text(column)

        for column in dataframe.columns

    ]

    # --------------------------------------------------------
    # CASE 1:
    # Metric / Value format
    # --------------------------------------------------------

    metric_column_candidates = [

        "metric",
        "metrics",
        "measure",
        "evaluation_metric",
        "name",

    ]

    value_column_candidates = [

        "value",
        "score",
        "metric_value",
        "result",
        "metric_score",

    ]

    metric_column = next(

        (

            column

            for column in metric_column_candidates

            if column in dataframe.columns

        ),

        None,

    )

    value_column = next(

        (

            column

            for column in value_column_candidates

            if column in dataframe.columns

        ),

        None,

    )

    if metric_column and value_column:

        for _, row in dataframe.iterrows():

            metric_name = normalize_metric_name(

                row[metric_column]

            )

            metric_value = convert_to_number(

                row[value_column]

            )

            if (

                metric_name

                and metric_value is not None

            ):

                metrics[metric_name] = metric_value

    # --------------------------------------------------------
    # CASE 2:
    # Metrics as column names
    # --------------------------------------------------------

    for column in dataframe.columns:

        metric_name = normalize_metric_name(column)

        if metric_name and metrics[metric_name] is None:

            values = (

                dataframe[column]

                .dropna()

                .tolist()

            )

            for value in values:

                metric_value = convert_to_number(value)

                if metric_value is not None:

                    metrics[metric_name] = metric_value

                    break

    return metrics

def extract_metrics_from_text(text):

    """
    Extract metrics from TXT, LOG, JSON text.
    """

    metrics = {

        metric: None

        for metric in METRIC_COLUMNS

    }

    patterns = {

        "R2": [

            r"\br2(?:_score)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

            r"\br_squared\b\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "MAE": [

            r"\bmae\b\s*[:=]\s*([-+]?\d*\.?\d+)",

            r"mean_absolute_error\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "RMSE": [

            r"\brmse\b\s*[:=]\s*([-+]?\d*\.?\d+)",

            r"root_mean_squared_error\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "Accuracy": [

            r"\baccuracy(?:_score)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "Precision": [

            r"\bprecision(?:_score)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "Recall": [

            r"\brecall(?:_score)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

        "F1_Score": [

            r"\bf1(?:_score|_macro)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

            r"\bmacro_f1(?:_score)?\b\s*[:=]\s*([-+]?\d*\.?\d+)",

        ],

    }

    text = text.lower()

    for metric_name, metric_patterns in patterns.items():

        for pattern in metric_patterns:

            match = re.search(

                pattern,

                text

            )

            if match:

                metrics[metric_name] = convert_to_number(

                    match.group(1)

                )

                break

    return metrics

def read_file_metrics(file_path):

    """
    Read supported files and extract metrics.
    """

    suffix = file_path.suffix.lower()

    try:

        # ---------------- CSV ----------------

        if suffix == ".csv":

            dataframe = pd.read_csv(file_path)

            return extract_metrics_from_dataframe(

                dataframe

            )

        # ---------------- JSON ----------------

        elif suffix == ".json":

            with open(

                file_path,

                "r",

                encoding="utf-8"

            ) as file:

                json_data = json.load(file)

            if isinstance(json_data, dict):

                dataframe = pd.json_normalize(

                    json_data

                )

                metrics = extract_metrics_from_dataframe(

                    dataframe

                )

                if any(

                    value is not None

                    for value in metrics.values()

                ):

                    return metrics

            return extract_metrics_from_text(

                json.dumps(json_data)

            )

        # ---------------- TXT / LOG ----------------

        elif suffix in {

            ".txt",

            ".log"

        }:

            text = file_path.read_text(

                encoding="utf-8",

                errors="ignore"

            )

            return extract_metrics_from_text(text)

    except Exception as error:

        print(

            f"Could not read {file_path.name}: {error}"

        )

    return {

        metric: None

        for metric in METRIC_COLUMNS

    }

def file_looks_like_evaluation_file(file_path):

    """
    Check whether file is likely an evaluation result file.
    """

    filename = normalize_text(

        file_path.name

    )

    keywords = [

        "evaluation",

        "metric",

        "metrics",

        "performance",

        "result",

        "report",

        "score",

        "classification",

        "regression",

        "validation",

    ]

    return any(

        keyword in filename

        for keyword in keywords

    )

# ============================================================
# 5. FIND AND EXTRACT ALL EVALUATION RESULTS
# ============================================================

if not BASE_PATH.exists():

    print(
        "ERROR: Folder does not exist:"
    )

    print(BASE_PATH)

    sys.exit(1)

supported_extensions = {

    ".csv",
    ".json",
    ".txt",
    ".log"

}

all_files = [

    file_path

    for file_path in BASE_PATH.rglob("*")

    if (

        file_path.is_file()

        and file_path.suffix.lower()
        in supported_extensions

        and OUTPUT_DIR not in file_path.parents

    )

]

comparison_rows = []

for file_path in all_files:

    dataset_info = identify_dataset(file_path)

    if dataset_info is None:

        continue

    if not file_looks_like_evaluation_file(file_path):

        continue

    metrics = read_file_metrics(file_path)

    has_metrics = any(

        value is not None

        for value in metrics.values()

    )

    if not has_metrics:

        continue

    row = {

        "Dataset": dataset_info["dataset"],

        "Task": dataset_info["task"],

        "Model": dataset_info["model"],

        "Source_File": file_path.name,

        "Source_Path": str(file_path),

    }

    row.update(metrics)

    comparison_rows.append(row)

# ============================================================
# 6. VALIDATE RESULTS
# ============================================================

if not comparison_rows:

    print(
        "No evaluation metrics were found."
    )

    print()

    print(
        "Check:"
    )

    print(
        "1. Files are inside Baseline_Models folder"
    )

    print(
        "2. File names contain evaluation keywords"
    )

    print(
        "3. Files contain supported metrics"
    )

    sys.exit(0)

comparison_df = pd.DataFrame(

    comparison_rows

)

# Remove duplicate entries

comparison_df = comparison_df.drop_duplicates()

# Arrange columns

comparison_df = comparison_df[

    [

        "Dataset",

        "Task",

        "Model",

        "R2",

        "MAE",

        "RMSE",

        "Accuracy",

        "Precision",

        "Recall",

        "F1_Score",

        "Source_File",

        "Source_Path",

    ]

]

print()

print(
    "Extracted Model Performance:"
)

print()

print(
    comparison_df.to_string(index=False)
)

# ============================================================
# 7. SAVE ALL EXTRACTED RESULTS
# ============================================================

all_results_file = (

    OUTPUT_DIR /

    "All_Extracted_Model_Performance.csv"

)

comparison_df.to_csv(

    all_results_file,

    index=False

)

print()

print(
    "Saved:"
)

print(
    all_results_file
)

# ============================================================
# 8. CREATE CONSOLIDATED DATASET SUMMARY
# ============================================================

summary_rows = []

for dataset_name, group in comparison_df.groupby("Dataset"):

    first_row = group.iloc[0]

    summary_row = {

        "Dataset": dataset_name,

        "Task": first_row["Task"],

        "Model": first_row["Model"],

    }

    for metric in METRIC_COLUMNS:

        available_values = (

            group[metric]

            .dropna()

        )

        if not available_values.empty:

            summary_row[metric] = (

                available_values.iloc[0]

            )

        else:

            summary_row[metric] = None

    summary_row["Source_File"] = (

        "; ".join(

            group["Source_File"]

            .astype(str)

            .tolist()

        )

    )

    summary_rows.append(summary_row)

summary_df = pd.DataFrame(

    summary_rows

)

summary_file = (

    OUTPUT_DIR /

    "Consolidated_Model_Performance_Comparison.csv"

)

summary_df.to_csv(

    summary_file,

    index=False

)

print()

print(
    "Saved:"
)

print(
    summary_file
)

# ============================================================
# 9. DISPLAY CONSOLIDATED RESULTS
# ============================================================

print()

print(
    "Consolidated Model Performance:"
)

print()

print(

    summary_df.to_string(

        index=False

    )

)

# ============================================================
# 10. IDENTIFY STRONGER MODELS
# ============================================================

strength_rows = []

for task_name, group in summary_df.groupby("Task"):

    # ---------------- Regression ----------------

    if task_name == "Regression":

        # Higher R2 is better

        if group["R2"].notna().any():

            best_row = group.loc[

                group["R2"].idxmax()

            ]

            strength_rows.append({

                "Task": task_name,

                "Comparison_Metric": "R2",

                "Best_Model": best_row["Model"],

                "Best_Dataset": best_row["Dataset"],

                "Best_Value": best_row["R2"],

                "Interpretation":
                    "Higher R2 indicates stronger regression performance."

            })

        # Lower RMSE is better

        if group["RMSE"].notna().any():

            best_row = group.loc[

                group["RMSE"].idxmin()

            ]

            strength_rows.append({

                "Task": task_name,

                "Comparison_Metric": "RMSE",

                "Best_Model": best_row["Model"],

                "Best_Dataset": best_row["Dataset"],

                "Best_Value": best_row["RMSE"],

                "Interpretation":
                    "Lower RMSE indicates lower prediction error."

            })

    # ---------------- Classification ----------------

    elif task_name == "Classification":

        # Higher F1 is better

        if group["F1_Score"].notna().any():

            best_row = group.loc[

                group["F1_Score"].idxmax()

            ]

            strength_rows.append({

                "Task": task_name,

                "Comparison_Metric": "F1_Score",

                "Best_Model": best_row["Model"],

                "Best_Dataset": best_row["Dataset"],

                "Best_Value": best_row["F1_Score"],

                "Interpretation":
                    "Higher F1 score indicates balanced precision and recall."

            })

        # Higher accuracy is better

        if group["Accuracy"].notna().any():

            best_row = group.loc[

                group["Accuracy"].idxmax()

            ]

            strength_rows.append({

                "Task": task_name,

                "Comparison_Metric": "Accuracy",

                "Best_Model": best_row["Model"],

                "Best_Dataset": best_row["Dataset"],

                "Best_Value": best_row["Accuracy"],

                "Interpretation":
                    "Higher accuracy indicates more correct predictions."

            })

strength_df = pd.DataFrame(

    strength_rows

)

strength_file = (

    OUTPUT_DIR /

    "Model_Strength_Analysis.csv"

)

strength_df.to_csv(

    strength_file,

    index=False

)

print()

print(
    "Saved:"
)

print(
    strength_file
)

# ============================================================
# 11. GENERATE COMPARISON CHARTS
# ============================================================

try:

    import matplotlib.pyplot as plt

    regression_df = summary_df[

        summary_df["Task"] == "Regression"

    ]

    classification_df = summary_df[

        summary_df["Task"] == "Classification"

    ]

    # ---------------- R2 Chart ----------------

    if (

        not regression_df.empty

        and regression_df["R2"].notna().any()

    ):

        chart_df = regression_df.dropna(

            subset=["R2"]

        )

        plt.figure(figsize=(10, 6))

        plt.bar(

            chart_df["Dataset"],

            chart_df["R2"]

        )

        plt.title(

            "Regression Model Comparison - R2"

        )

        plt.xlabel(

            "Dataset"

        )

        plt.ylabel(

            "R2 Score"

        )

        plt.xticks(

            rotation=25,

            ha="right"

        )

        plt.tight_layout()

        plt.savefig(

            OUTPUT_DIR /

            "Regression_R2_Comparison.png",

            dpi=300

        )

        plt.close()

    # ---------------- RMSE Chart ----------------

    if (

        not regression_df.empty

        and regression_df["RMSE"].notna().any()

    ):

        chart_df = regression_df.dropna(

            subset=["RMSE"]

        )

        plt.figure(figsize=(10, 6))

        plt.bar(

            chart_df["Dataset"],

            chart_df["RMSE"]

        )

        plt.title(

            "Regression Model Comparison - RMSE"

        )

        plt.xlabel(

            "Dataset"

        )

        plt.ylabel(

            "RMSE"

        )

        plt.xticks(

            rotation=25,

            ha="right"

        )

        plt.tight_layout()

        plt.savefig(

            OUTPUT_DIR /

            "Regression_RMSE_Comparison.png",

            dpi=300

        )

        plt.close()

    # ---------------- F1 Chart ----------------

    if (

        not classification_df.empty

        and classification_df["F1_Score"].notna().any()

    ):

        chart_df = classification_df.dropna(

            subset=["F1_Score"]

        )

        plt.figure(figsize=(10, 6))

        plt.bar(

            chart_df["Dataset"],

            chart_df["F1_Score"]

        )

        plt.title(

            "Classification Model Comparison - F1 Score"

        )

        plt.xlabel(

            "Dataset"

        )

        plt.ylabel(

            "F1 Score"

        )

        plt.xticks(

            rotation=25,

            ha="right"

        )

        plt.tight_layout()

        plt.savefig(

            OUTPUT_DIR /

            "Classification_F1_Comparison.png",

            dpi=300

        )

        plt.close()

    print()

    print(
        "Charts generated successfully."
    )

except ImportError:

    print()

    print(
        "Matplotlib not installed."
    )

    print(
        "Charts skipped."
    )

# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print()

print(
    "=" * 60
)

print(
    "MODEL PERFORMANCE COMPARISON COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)

print()

print(
    "Output Folder:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Generated Files:"
)

print(
    "- All_Extracted_Model_Performance.csv"
)

print(
    "- Consolidated_Model_Performance_Comparison.csv"
)

print(
    "- Model_Strength_Analysis.csv"
)

print()

print(
    "Completed."
)
