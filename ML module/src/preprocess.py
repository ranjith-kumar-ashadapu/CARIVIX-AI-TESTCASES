"""
Data Preprocessing Module for CARIVIX AI Model Training pipeline.

Provides:
- Missing value handling
- Duplicate removal
- Outlier detection & handling (IQR, Z-score)
- Label Encoding
- One-Hot Encoding
- Target Encoding
- Feature Scaling (Standard, MinMax, Robust)
- Log & Box-Cox transformations
- SMOTE for imbalanced classification
- Complete preprocessing pipeline
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
)

from src.utils import setup_logger

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Missing Value Handling
# =============================================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    """
    logger.info("Handling missing values with strategy: '%s'", strategy)

    if columns is None:
        columns = df.columns.tolist()

    df_clean = df.copy()
    numeric_cols = df_clean[columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean[columns].select_dtypes(include=["object", "category"]).columns.tolist()

    if strategy == "drop":
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=columns)
        dropped = initial_rows - len(df_clean)
        logger.info("Dropped %d rows with missing values", dropped)

    elif strategy == "mean":
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    elif strategy == "median":
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    elif strategy == "mode":
        for col in columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else None
                if mode_val is not None:
                    df_clean[col].fillna(mode_val, inplace=True)

    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'")
        for col in columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(fill_value, inplace=True)

    else:
        raise ValueError(f"Invalid strategy '{strategy}'. Choose from: 'mean', 'median', 'mode', 'drop', 'constant'")

    remaining_nulls = df_clean[columns].isnull().sum().sum()
    logger.info("Missing value handling complete. Remaining nulls: %d", remaining_nulls)
    return df_clean

# =============================================================================
# Duplicate Removal
# =============================================================================

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    logger.info("Removing duplicate rows...")
    initial_rows = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    duplicates_removed = initial_rows - len(df_clean)
    logger.info("Removed %d duplicate rows. Shape: %s -> %s", duplicates_removed, initial_rows, df_clean.shape)
    return df_clean

# =============================================================================
# Outlier Detection & Handling
# =============================================================================

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> Dict[str, np.ndarray]:
    """Detect outliers using the IQR method."""
    logger.info("Detecting outliers using IQR method (multiplier=%.2f).", multiplier)
    outlier_masks = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - multiplier * IQR) | (df[col] > Q3 + multiplier * IQR)
        outlier_masks[col] = mask.values
    return outlier_masks

def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict[str, np.ndarray]:
    """Detect outliers using the Z-score method."""
    logger.info("Detecting outliers using Z-score method (threshold=%.2f).", threshold)
    outlier_masks = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        mask = pd.Series(False, index=df.index)
        mask.loc[df[col].dropna().index] = z_scores > threshold
        outlier_masks[col] = mask.values
    return outlier_masks

def handle_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5, strategy: str = "clip", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Detect and handle outliers in numerical columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Handling outliers using '%s' method (threshold=%.2f, strategy='%s').", method, threshold, strategy)
    df_clean = df.copy()

    outlier_masks = detect_outliers_iqr(df_clean, columns, multiplier=threshold) if method == "iqr" else detect_outliers_zscore(df_clean, columns, threshold=threshold)

    for col, mask in outlier_masks.items():
        n_outliers = mask.sum()
        if n_outliers == 0:
            continue
        if strategy == "clip":
            lower = df_clean[col].quantile(0.25) - threshold * (df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25))
            upper = df_clean[col].quantile(0.75) + threshold * (df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25))
            if method == "zscore":
                lower = df_clean[col].mean() - threshold * df_clean[col].std()
                upper = df_clean[col].mean() + threshold * df_clean[col].std()
            df_clean[col] = df_clean[col].clip(lower, upper)
        elif strategy == "remove":
            df_clean = df_clean[~mask]
        elif strategy == "winsorize":
            lower = df_clean[col].quantile(0.05)
            upper = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(lower, upper)

    logger.info("Outlier handling complete. Shape: %s", df_clean.shape)
    return df_clean

# =============================================================================
# Feature Transformation (Log, Box-Cox)
# =============================================================================

def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply natural log transformation to skewed numerical features."""
    logger.info("Applying log transformation to %d columns.", len(columns))
    df_transformed = df.copy()
    shift_values = {}
    for col in columns:
        if col not in df_transformed.columns:
            continue
        shift = -df_transformed[col].min() + 1 if df_transformed[col].min() <= 0 else 0
        shift_values[col] = shift
        df_transformed[col] = np.log1p(df_transformed[col] + shift)
    logger.info("Log transformation complete.")
    return df_transformed, shift_values

def apply_boxcox_transform(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply Box-Cox transformation to make features more normally distributed."""
    logger.info("Applying Box-Cox transformation to %d columns.", len(columns))
    df_transformed = df.copy()
    transformers = {}
    for col in columns:
        if col not in df_transformed.columns:
            continue
        shift = -df_transformed[col].min() + 1e-6 if df_transformed[col].min() <= 0 else 0
        pt = PowerTransformer(method="box-cox", standardize=False)
        df_transformed[col] = pt.fit_transform((df_transformed[col].values.reshape(-1, 1) + shift)).flatten()
        transformers[col] = {"power_transformer": pt, "shift": shift}
    logger.info("Box-Cox transformation complete.")
    return df_transformed, transformers

# =============================================================================
# Imbalanced Data Handling (SMOTE)
# =============================================================================

def apply_smote(X: pd.DataFrame, y: pd.Series, method: str = "smote", random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE or other resampling techniques for imbalanced datasets."""
    logger.info("Applying '%s' for imbalanced data handling.", method)
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        raise ImportError("imbalanced-learn is required. Install: pip install imbalanced-learn")

    X_numeric = X.select_dtypes(include=[np.number])
    sampler_map = {"smote": SMOTE(random_state=random_state), "adasyn": ADASYN(random_state=random_state), "random_oversample": RandomOverSampler(random_state=random_state), "random_undersample": RandomUnderSampler(random_state=random_state)}
    if method not in sampler_map:
        raise ValueError(f"Invalid imbalanced method '{method}'.")
    X_resampled, y_resampled = sampler_map[method].fit_resample(X_numeric, y)
    logger.info("Resampling complete. %s -> %s", X.shape, X_resampled.shape)
    return X_resampled, y_resampled

# =============================================================================
# Label Encoding
# =============================================================================

def label_encode(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Apply Label Encoding to specified categorical columns."""
    logger.info("Applying Label Encoding to columns: %s", columns)
    df_encoded = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in columns:
        if col not in df_encoded.columns:
            continue
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    logger.info("Label Encoding complete. Encoded %d columns.", len(encoders))
    return df_encoded, encoders

# =============================================================================
# One-Hot Encoding
# =============================================================================

def one_hot_encode(df: pd.DataFrame, columns: List[str], drop_first: bool = False, max_categories: Optional[int] = None) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Apply One-Hot Encoding to specified categorical columns."""
    logger.info("Applying One-Hot Encoding to columns: %s (drop_first=%s)", columns, drop_first)
    df_encoded = df.copy()
    valid_columns = [col for col in columns if col in df_encoded.columns]
    if not valid_columns:
        return df_encoded, OneHotEncoder()
    ohe = OneHotEncoder(drop="first" if drop_first else None, sparse_output=False, handle_unknown="ignore", max_categories=max_categories)
    encoded_array = ohe.fit_transform(df_encoded[valid_columns].astype(str))
    encoded_columns = ohe.get_feature_names_out(valid_columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df_encoded.index)
    df_encoded = df_encoded.drop(columns=valid_columns)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    logger.info("One-Hot Encoding complete. Added %d new columns.", len(encoded_columns))
    return df_encoded, ohe

# =============================================================================
# Target Encoding
# =============================================================================

def target_encode(df: pd.DataFrame, columns: List[str], target: pd.Series, alpha: float = 5.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply Target Encoding to categorical columns using smoothed mean of the target."""
    logger.info("Applying Target Encoding to columns: %s (alpha=%.2f)", columns, alpha)
    from sklearn.model_selection import KFold
    df_encoded = df.copy()
    encoding_maps = {}
    global_mean = target.mean()
    for col in columns:
        if col not in df_encoded.columns:
            continue
        te_column = f"{col}_target_enc"
        df_encoded[te_column] = np.nan
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(df_encoded):
            train_agg = df_encoded.iloc[train_idx].groupby(col)[target.name].agg(["mean", "count"])
            train_agg.columns = ["mean_target", "count"]
            train_agg["smoothed"] = (train_agg["count"] * train_agg["mean_target"] + alpha * global_mean) / (train_agg["count"] + alpha)
            df_encoded.loc[df_encoded.index[val_idx], te_column] = df_encoded.loc[df_encoded.index[val_idx], col].map(train_agg["smoothed"].to_dict())
        df_encoded[te_column].fillna(global_mean, inplace=True)
    logger.info("Target Encoding complete. Encoded %d columns.", len(encoding_maps))
    return df_encoded, encoding_maps

# =============================================================================
# Feature Scaling
# =============================================================================

def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """Scale numerical features using standardization or normalization."""
    logger.info("Scaling features using '%s' scaler for %d columns.", method, len(columns))
    valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_columns:
        return df.copy(), StandardScaler()
    df_scaled = df.copy()
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaling method '{method}'.")
    df_scaled[valid_columns] = scaler.fit_transform(df_scaled[valid_columns])
    return df_scaled, scaler

# =============================================================================
# Complete Preprocessing Pipeline
# =============================================================================

def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict[str, Any], target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Run the full preprocessing pipeline on the dataset.

    Orchestrates: Missing values, duplicates, outliers, log transform,
    encoding, scaling, and SMOTE for imbalanced data.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    transformers = {}
    preprocess_cfg = config.get("preprocessing", {})

    # Step 1: Handle Missing Values
    if preprocess_cfg.get("handle_missing", True):
        df = handle_missing_values(df, strategy=preprocess_cfg.get("missing_strategy", "mean"))
        transformers["missing_strategy"] = preprocess_cfg.get("missing_strategy", "mean")

    # Step 2: Remove Duplicates
    if preprocess_cfg.get("remove_duplicates", True):
        df = remove_duplicates(df)

    # Step 3: Handle Outliers
    if preprocess_cfg.get("handle_outliers", False):
        df = handle_outliers(df, method=preprocess_cfg.get("outlier_method", "iqr"), threshold=preprocess_cfg.get("outlier_threshold", 1.5), strategy=preprocess_cfg.get("outlier_strategy", "clip"))
        transformers["outlier_config"] = {"method": preprocess_cfg.get("outlier_method"), "threshold": preprocess_cfg.get("outlier_threshold"), "strategy": preprocess_cfg.get("outlier_strategy")}

    # Step 4: Separate features and target
    if target_column is not None and target_column in df.columns:
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
    elif target_column is None:
        y = None
        X = df.copy()
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Step 5: Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Column types - Numeric: %d, Categorical: %d", len(numeric_cols), len(categorical_cols))

    # Step 6: Apply Log Transformation to skewed features
    if preprocess_cfg.get("apply_log_transform", False):
        skew_threshold = preprocess_cfg.get("skewness_threshold", 1.0)
        skewed_cols = [c for c in numeric_cols if c in X.columns and abs(X[c].skew()) > skew_threshold]
        if skewed_cols:
            X, shift_values = apply_log_transform(X, columns=skewed_cols)
            transformers["log_shift_values"] = shift_values

    # Step 7: Encode target if categorical
    if y is not None and (y.dtype == "object" or str(y.dtype) == "category"):
        y_encoded, label_enc = label_encode(pd.DataFrame({"target": y}), columns=["target"])
        y = y_encoded["target"]
        transformers["target_encoder"] = label_enc["target"]

    # Step 8: Encode categorical features
    low_cardinality_cols = [c for c in categorical_cols if X[c].nunique() <= 10]
    high_cardinality_cols = [c for c in categorical_cols if X[c].nunique() > 10]

    if low_cardinality_cols:
        X, label_encoders = label_encode(X, columns=low_cardinality_cols)
        transformers["label_encoders"] = label_encoders

    if high_cardinality_cols:
        X, ohe_encoder = one_hot_encode(X, columns=high_cardinality_cols, drop_first=True, max_categories=20)
        transformers["onehot_encoder"] = ohe_encoder

    # Step 9: Scale numerical features
    numeric_cols_to_scale = [c for c in X.select_dtypes(include=[np.number]).columns if c in X.columns and X[c].nunique() > 2]
    if numeric_cols_to_scale:
        X, scaler = scale_features(X, columns=numeric_cols_to_scale, method=preprocess_cfg.get("scaling_method", "standard"))
        transformers["scaler"] = scaler

    # Step 10: Handle Imbalanced Data
    if preprocess_cfg.get("handle_imbalanced", False) and y is not None and len(np.unique(y)) == 2:
        X, y = apply_smote(X, y, method=preprocess_cfg.get("imbalanced_method", "smote"))
        transformers["imbalanced_method"] = preprocess_cfg.get("imbalanced_method", "smote")

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)
    return X, y, transformers

