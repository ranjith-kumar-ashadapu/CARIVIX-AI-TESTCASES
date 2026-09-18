"""
CARIVIX AI — Optimized Preprocessing Functions
================================================
Companion to data_transformation.py. Focus: performance.

Optimization techniques applied:
  1. Vectorized pandas/numpy ops instead of row-wise .apply()/loops
  2. Categorical dtype downcasting to cut memory footprint
  3. Chunked reading for large CSVs that don't fit in memory
  4. Optional multiprocessing for CPU-bound custom transforms
  5. Before/after benchmarking helper to prove the optimization worked

Dependencies: pandas, numpy
"""

from __future__ import annotations
import time
import logging
from functools import wraps
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterator

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("carivix.preprocessing_optimized")


# ----------------------------------------------------------------------
# Benchmark decorator — use this to compare naive vs optimized versions
# ----------------------------------------------------------------------
def benchmark(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[benchmark] {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


# ----------------------------------------------------------------------
# 1. VECTORIZATION — avoid .apply(axis=1) / python loops
# ----------------------------------------------------------------------
@benchmark
def slow_row_apply(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """NAIVE baseline (what NOT to do): row-wise apply. O(n) with heavy python overhead."""
    return df.apply(lambda row: row[col_a] * 1.18 if row[col_b] == "taxable" else row[col_a], axis=1)


@benchmark
def fast_vectorized(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """OPTIMIZED: same result, pure vectorized numpy — typically 20-100x faster."""
    return np.where(df[col_b].values == "taxable", df[col_a].values * 1.18, df[col_a].values)


# ----------------------------------------------------------------------
# 2. MEMORY OPTIMIZATION — downcast numeric types, use category dtype
# ----------------------------------------------------------------------
def optimize_dtypes(df: pd.DataFrame, category_threshold: float = 0.5) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest safe dtype and convert
    low-cardinality object columns to 'category' to reduce memory.
    category_threshold: convert to category if n_unique / n_rows < threshold.
    """
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        n_unique = df[col].nunique(dropna=True)
        n_rows = len(df)
        if n_rows > 0 and (n_unique / n_rows) < category_threshold:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    logger.info(f"Memory: {start_mem:.2f} MB -> {end_mem:.2f} MB  ({reduction:.1f}% reduction)")
    return df


# ----------------------------------------------------------------------
# 3. CHUNKED I/O — process large CSVs without loading fully into memory
# ----------------------------------------------------------------------
def process_large_csv_in_chunks(
    filepath: str,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    chunksize: int = 100_000,
    output_path: str | None = None,
) -> pd.DataFrame | None:
    """
    Stream a large CSV in chunks, apply `transform_fn` to each chunk,
    and either write incrementally to output_path or return the concatenated result.
    """
    processed_chunks: list[pd.DataFrame] = []
    header_written = False

    reader: Iterator[pd.DataFrame] = pd.read_csv(filepath, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        chunk = transform_fn(chunk)
        if output_path:
            chunk.to_csv(output_path, mode="a", index=False, header=not header_written)
            header_written = True
        else:
            processed_chunks.append(chunk)
        logger.info(f"Processed chunk {i + 1} ({len(chunk)} rows)")

    if output_path:
        logger.info(f"Streamed all chunks -> {output_path}")
        return None
    return pd.concat(processed_chunks, ignore_index=True)


# ----------------------------------------------------------------------
# 4. PARALLELIZATION — for CPU-bound custom row transforms that can't vectorize
# ----------------------------------------------------------------------
def parallel_apply(df: pd.DataFrame, func: Callable[[pd.DataFrame], pd.Series],
                    n_jobs: int | None = None) -> pd.Series:
    """
    Split df into n_jobs partitions, run `func` on each partition in a
    separate process, then reassemble. Use only for genuinely CPU-bound
    transforms — vectorized numpy/pandas is almost always faster for
    simple arithmetic (see fast_vectorized above).
    """
    n_jobs = n_jobs or max(cpu_count() - 1, 1)
    partitions = np.array_split(df, n_jobs)
    with Pool(n_jobs) as pool:
        results = pool.map(func, partitions)
    return pd.concat(results)


if __name__ == "__main__":
    # Demonstrate the vectorization speedup on a synthetic dataset
    n = 200_000
    demo = pd.DataFrame({
        "amount": np.random.uniform(10, 1000, n),
        "tax_status": np.random.choice(["taxable", "exempt"], n),
    })

    _ = slow_row_apply(demo, "amount", "tax_status")
    _ = fast_vectorized(demo, "amount", "tax_status")

    optimize_dtypes(demo)
