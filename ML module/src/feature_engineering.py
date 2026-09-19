"""
Feature Engineering Module for CARIVIX AI Model Training pipeline.

Provides:
- Numerical feature creation (polynomial, interaction, ratios)
- Binning features (equal width, quantile-based)
- Categorical feature encoding and aggregation
- Date feature extraction (year, month, day, weekday, etc.)
- PCA for dimensionality reduction
- Feature selection (variance threshold, correlation-based)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from src.utils import setup_logger

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Numerical Feature Engineering
# =============================================================================

def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
) -> pd.DataFrame:
    """
    Create polynomial features from numerical columns.
    """
    logger.info("Creating polynomial features (degree=%d, interaction_only=%s) for %d columns.", degree, interaction_only, len(columns))
    df_poly = df.copy()
    valid_columns = [col for col in columns if col in df_poly.columns]
    if len(valid_columns) < 2:
        logger.warning("Need at least 2 columns for polynomial features.")
        return df_poly

    new_features_count = 0
    new_columns = {}
    for i in range(len(valid_columns)):
        for j in range(i, len(valid_columns)):
            col_i, col_j = valid_columns[i], valid_columns[j]
            if i == j:
                if not interaction_only:
                    for d in range(2, degree + 1):
                        new_col = f"{col_i}^{d}"
                        if new_col not in df_poly.columns:
                            new_columns[new_col] = df_poly[col_i] ** d
                            new_features_count += 1
            else:
                new_col = f"{col_i}_x_{col_j}"
                if new_col not in df_poly.columns:
                    new_columns[new_col] = df_poly[col_i] * df_poly[col_j]
                    new_features_count += 1

    if new_columns:
        df_poly = pd.concat([df_poly, pd.DataFrame(new_columns, index=df_poly.index)], axis=1)

    logger.info("Created %d new polynomial features.", new_features_count)
    return df_poly

def create_ratio_features(df: pd.DataFrame, column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Create ratio features from pairs of numerical columns."""
    logger.info("Creating ratio features for %d column pairs.", len(column_pairs))
    df_ratio = df.copy()
    new_features_count = 0
    for num_col, den_col in column_pairs:
        if num_col not in df_ratio.columns or den_col not in df_ratio.columns:
            continue
        ratio_name = f"{num_col}_div_{den_col}"
        epsilon = 1e-10
        df_ratio[ratio_name] = df_ratio[num_col] / (df_ratio[den_col] + epsilon)
        new_features_count += 1
    logger.info("Created %d ratio features.", new_features_count)
    return df_ratio

# =============================================================================
# Binning Features
# =============================================================================

def create_binning_features(
    df: pd.DataFrame,
    columns: List[str],
    n_bins: int = 5,
    strategy: str = "quantile",
    encode: str = "onehot-dense",
) -> pd.DataFrame:
    """
    Create binning features from numerical columns.

    Args:
        df: Input DataFrame.
        columns: List of numerical column names to bin.
        n_bins: Number of bins.
        strategy: Binning strategy ('uniform' for equal-width, 'quantile' for equal-frequency).
        encode: Encoding method ('onehot-dense' or 'ordinal').

    Returns:
        DataFrame with additional binning features.
    """
    logger.info("Creating binning features for %d columns (n_bins=%d, strategy=%s).", len(columns), n_bins, strategy)
    df_bin = df.copy()
    new_features_count = 0

    for col in columns:
        if col not in df_bin.columns:
            continue
        try:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            binned = discretizer.fit_transform(df_bin[[col]])

            if encode == "ordinal":
                new_col = f"{col}_binned"
                df_bin[new_col] = binned.flatten()
                new_features_count += 1
            else:
                for i in range(n_bins):
                    new_col = f"{col}_bin_{i}"
                    df_bin[new_col] = binned[:, i]
                    new_features_count += 1

            logger.debug("Binned '%s' into %d bins.", col, n_bins)
        except Exception as exc:
            logger.warning("Binning failed for '%s': %s", col, exc)

    logger.info("Created %d binning features.", new_features_count)
    return df_bin

# =============================================================================
# Categorical Feature Engineering
# =============================================================================

def create_frequency_encoding(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Create frequency encoding for categorical columns."""
    logger.info("Creating frequency encoding for %d columns.", len(columns))
    df_freq = df.copy()
    new_features_count = 0
    for col in columns:
        if col not in df_freq.columns:
            continue
        freq_col = f"{col}_freq"
        if freq_col not in df_freq.columns:
            freq_map = df_freq[col].value_counts() / len(df_freq)
            df_freq[freq_col] = df_freq[col].map(freq_map)
            new_features_count += 1
    logger.info("Created %d frequency encoding features.", new_features_count)
    return df_freq

def create_target_encoding(df: pd.DataFrame, columns: List[str], target: pd.Series, alpha: float = 5.0) -> pd.DataFrame:
    """Create target encoding for categorical columns (mean of target per category)."""
    logger.info("Creating target encoding for %d columns (alpha=%.2f).", len(columns), alpha)
    df_te = df.copy()
    new_features_count = 0
    global_mean = target.mean()
    for col in columns:
        if col not in df_te.columns:
            continue
        te_col = f"{col}_target_enc"
        if te_col not in df_te.columns:
            agg_df = df_te.groupby(col)[target.name].agg(["mean", "count"]).reset_index()
            agg_df["smoothed"] = (agg_df["count"] * agg_df["mean"] + alpha * global_mean) / (agg_df["count"] + alpha)
            df_te[te_col] = df_te[col].map(agg_df.set_index(col)["smoothed"].to_dict())
            new_features_count += 1
    logger.info("Created %d target encoding features.", new_features_count)
    return df_te

# =============================================================================
# Text Feature Engineering
# =============================================================================

def create_text_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Create simple text-derived features for categorical/string columns."""
    logger.info("Creating text-derived features for %d columns.", len(columns))
    df_text = df.copy()
    created_features = []

    for col in columns:
        if col not in df_text.columns:
            continue

        text_series = df_text[col].astype(str)
        length_col = f"{col}_length"
        word_count_col = f"{col}_word_count"
        unique_word_count_col = f"{col}_unique_word_count"

        if length_col not in df_text.columns:
            df_text[length_col] = text_series.str.len()
            created_features.append(length_col)
        if word_count_col not in df_text.columns:
            df_text[word_count_col] = text_series.str.split().str.len()
            created_features.append(word_count_col)
        if unique_word_count_col not in df_text.columns:
            df_text[unique_word_count_col] = text_series.str.split().apply(lambda values: len(set(values)))
            created_features.append(unique_word_count_col)

    logger.info("Created %d text features.", len(created_features))
    return df_text

# =============================================================================
# Date Feature Extraction
# =============================================================================

def extract_date_features(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Extract various features from datetime columns."""
    logger.info("Extracting date features from %d columns.", len(date_columns))
    df_date = df.copy()
    new_features_count = 0
    for col in date_columns:
        if col not in df_date.columns:
            continue
        try:
            df_date[col] = pd.to_datetime(df_date[col], errors="coerce", infer_datetime_format=True)
        except (ValueError, TypeError) as exc:
            logger.warning("Could not convert '%s' to datetime: %s.", col, exc)
            continue
        if df_date[col].isna().all():
            logger.warning("Column '%s' could not be converted to datetime values.", col)
            continue
        prefix = col
        date_features = {
            f"{prefix}_year": df_date[col].dt.year,
            f"{prefix}_month": df_date[col].dt.month,
            f"{prefix}_day": df_date[col].dt.day,
            f"{prefix}_day_of_week": df_date[col].dt.dayofweek,
            f"{prefix}_quarter": df_date[col].dt.quarter,
            f"{prefix}_is_weekend": (df_date[col].dt.dayofweek >= 5).astype(int),
            f"{prefix}_day_of_year": df_date[col].dt.dayofyear,
            f"{prefix}_week_of_year": df_date[col].dt.isocalendar().week.astype(int),
            f"{prefix}_hour": df_date[col].dt.hour,
            f"{prefix}_minute": df_date[col].dt.minute,
            f"{prefix}_second": df_date[col].dt.second,
        }
        for feat_name, feat_values in date_features.items():
            if feat_name not in df_date.columns:
                df_date[feat_name] = feat_values
                new_features_count += 1
    logger.info("Extracted %d date features.", new_features_count)
    return df_date

def infer_datetime_columns(df: pd.DataFrame, sample_size: int = 100, threshold: float = 0.9) -> List[str]:
    """Infer object columns that represent datetime values."""
    date_columns: List[str] = []
    for col in df.select_dtypes(include=["object"]).columns.tolist():
        non_null_values = df[col].dropna().astype(str)
        if len(non_null_values) < 3:
            continue
        sample = non_null_values.sample(n=min(sample_size, len(non_null_values)), random_state=42)
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().sum() / len(sample) >= threshold:
            date_columns.append(col)
    return date_columns

# =============================================================================
# Clustering Features
# =============================================================================

def create_clustering_features(
    df: pd.DataFrame,
    columns: List[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create clustering-based features from numeric columns."""
    logger.info("Creating clustering features for %d columns.", len(columns))
    df_cluster = df.copy()
    valid_columns = [col for col in columns if col in df_cluster.columns and pd.api.types.is_numeric_dtype(df_cluster[col])]

    if len(valid_columns) < 2 or len(df_cluster) < 5:
        logger.warning("Insufficient data for clustering features.")
        return df_cluster

    scaled = StandardScaler().fit_transform(df_cluster[valid_columns].fillna(0))
    cluster_model = KMeans(n_clusters=min(n_clusters, len(df_cluster)), random_state=random_state, n_init=10)
    clusters = cluster_model.fit_predict(scaled)
    df_cluster["cluster_label"] = clusters
    logger.info("Created clustering features using %d clusters.", n_clusters)
    return df_cluster

# =============================================================================
# Dimensionality Reduction (PCA)
# =============================================================================

def apply_pca(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_ratio: float = 0.95,
    prefix: str = "pca",
) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA for dimensionality reduction on numerical features.

    Args:
        df: Input DataFrame.
        n_components: Number of PCA components. Overrides variance_ratio if set.
        variance_ratio: Explained variance ratio to retain.
        prefix: Prefix for PCA component column names.

    Returns:
        Tuple of (DataFrame with PCA components, fitted PCA object).
    """
    logger.info("Applying PCA (n_components=%s, variance_ratio=%.2f).", n_components, variance_ratio)

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.warning("Need at least 2 numeric features for PCA.")
        return df, PCA()

    # Fill remaining NaN values
    numeric_df = numeric_df.fillna(0)

    # Determine number of components
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(numeric_df)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = int(np.argmax(cumsum >= variance_ratio) + 1)
        n_components = max(1, min(n_components, numeric_df.shape[1]))
        logger.info("Auto-selected %d PCA components to explain %.0f%% variance.", n_components, variance_ratio * 100)

    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(numeric_df)

    pca_columns = [f"{prefix}_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_components, columns=pca_columns, index=df.index)

    logger.info("PCA complete. Shape: %s -> %s (explained variance: %.4f)", numeric_df.shape, pca_df.shape, pca.explained_variance_ratio_.sum())

    return pca_df, pca

# =============================================================================
# Feature Selection
# =============================================================================

def select_features_variance_threshold(
    df: pd.DataFrame, threshold: float = 0.01
) -> Tuple[pd.DataFrame, VarianceThreshold, List[str]]:
    """Select features based on variance threshold (removes low-variance features)."""
    logger.info("Selecting features using variance threshold (threshold=%.4f).", threshold)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return df, VarianceThreshold(), []

    constant_cols = [col for col in numeric_df.columns if numeric_df[col].std() == 0]
    if constant_cols:
        logger.info("Dropping %d constant columns.", len(constant_cols))
        numeric_df = numeric_df.drop(columns=constant_cols)

    if numeric_df.shape[1] == 0:
        return df, VarianceThreshold(), []

    selector = VarianceThreshold(threshold=threshold)
    selected_array = selector.fit_transform(numeric_df)
    selected_mask = selector.get_support()
    selected_features = numeric_df.columns[selected_mask].tolist()
    removed_features = numeric_df.columns[~selected_mask].tolist()

    logger.info("Feature selection: %d selected, %d removed.", len(selected_features), len(removed_features))

    df_selected = pd.DataFrame(selected_array, columns=selected_features, index=df.index)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric_cols:
        df_selected[col] = df[col].values

    return df_selected, selector, selected_features

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated features to reduce multicollinearity."""
    logger.info("Removing highly correlated features (threshold=%.2f).", threshold)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return df, []

    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    logger.info("Dropping %d highly correlated features.", len(to_drop))
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

def run_feature_engineering_pipeline(X: pd.DataFrame, y: Optional[pd.Series], config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full feature engineering pipeline.

    Supports:
    1. Date feature extraction
    2. Polynomial/interaction features
    3. Ratio features
    4. Frequency encoding
    5. Binning features
    6. PCA dimensionality reduction
    7. Feature selection (variance threshold + correlation removal)
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    fe_config = config.get("feature_engineering", {})
    metadata = {"created_features": [], "removed_features": []}

    # Step 1: Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
    inferred_datetime_cols = infer_datetime_columns(X)
    datetime_cols = list(dict.fromkeys(datetime_cols + inferred_datetime_cols))
    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
    text_cols = [col for col in categorical_cols if X[col].nunique() > 1]

    logger.info("Feature types - Numeric: %d, Categorical: %d, Datetime: %d", len(numeric_cols), len(categorical_cols), len(datetime_cols))

    # Step 2: Date Feature Extraction
    if fe_config.get("extract_date_features", True) and datetime_cols:
        original_cols = set(X.columns)
        X = extract_date_features(X, date_columns=datetime_cols)
        metadata["created_features"].extend(list(set(X.columns) - original_cols))

    # Step 3: Create text-derived features
    if fe_config.get("create_text_features", True) and text_cols:
        original_cols = set(X.columns)
        X = create_text_features(X, columns=text_cols[:5])
        metadata["created_features"].extend(list(set(X.columns) - original_cols))

    # Step 4: Create polynomial features
    if len(numeric_cols) >= 2:
        limited_numeric_cols = numeric_cols[:20]
        X = create_polynomial_features(X, columns=limited_numeric_cols, degree=2, interaction_only=True)
        new_poly = [col for col in X.columns if "_x_" in col]
        metadata["created_features"].extend(new_poly)

    # Step 5: Create frequency encoding
    if categorical_cols:
        X = create_frequency_encoding(X, columns=categorical_cols)
        new_freq = [col for col in X.columns if col.endswith("_freq")]
        metadata["created_features"].extend(new_freq)

    # Step 6: Create clustering features
    if fe_config.get("create_clustering_features", False) and len(numeric_cols) >= 2:
        X = create_clustering_features(X, columns=numeric_cols[:10])
        if "cluster_label" in X.columns:
            metadata["created_features"].append("cluster_label")

    # Step 7: Create binning features
    if fe_config.get("create_binning_features", False) and numeric_cols:
        n_bins = fe_config.get("n_bins", 5)
        strategy = fe_config.get("binning_strategy", "quantile")
        X = create_binning_features(X, columns=numeric_cols[:5], n_bins=n_bins, strategy=strategy)  # Limit to first 5 numeric cols

    # Step 6: Apply PCA
    if fe_config.get("apply_pca", False):
        pca_n_components = fe_config.get("pca_n_components")
        pca_variance_ratio = fe_config.get("pca_variance_ratio", 0.95)
        pca_df, pca_obj = apply_pca(X, n_components=pca_n_components, variance_ratio=pca_variance_ratio)
        # Keep original features and add PCA components
        X = pd.concat([X, pca_df], axis=1)
        metadata["pca"] = pca_obj
        metadata["created_features"].extend(pca_df.columns.tolist())

    # Step 7: Feature Selection
    selection_method = fe_config.get("feature_selection_method", "variance")
    if selection_method == "variance":
        variance_threshold = fe_config.get("variance_threshold", 0.01)
        X, selector, selected_features = select_features_variance_threshold(X, threshold=variance_threshold)
        metadata["selector"] = selector
        metadata["selected_features"] = selected_features
    elif selection_method == "correlation":
        corr_threshold = fe_config.get("correlation_threshold", 0.95)
        X, dropped_features = remove_highly_correlated_features(X, threshold=corr_threshold)
        metadata["removed_features"].extend(dropped_features)

    # Handle infinite or NaN values that might have been created
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("Feature engineering complete. Final shape: %s", X.shape)
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
    logger.info("=" * 60)
    return X, metadata

