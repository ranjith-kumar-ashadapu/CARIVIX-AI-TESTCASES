"""
Shared utilities for the CARIVIX AI RAG evaluation module.

Provides:
    - Timer context manager for performance profiling
    - Elapsed-time formatting helpers
    - A dependency-free console table renderer
    - A lightweight lexical relevance scorer (no ground truth required)
"""

import re
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Timing Utilities
# =============================================================================

@contextmanager
def Timer(name: str = "Operation"):
    """
    Context manager that measures elapsed wall-clock time.

    Args:
        name: Human-readable label for the timed operation.

    Yields:
        A reference to the elapsed-time dict so callers can read the
        duration live or after the block completes.

    Usage::

        with Timer("Embedding") as t:
            do_work()
        print(t["elapsed"])
    """
    start = time.perf_counter()
    timer = {"elapsed": 0.0, "name": name}
    try:
        yield timer
    finally:
        timer["elapsed"] = time.perf_counter() - start

def format_elapsed(seconds: float) -> str:
    """
    Format a duration in seconds into a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string like "1.234s", "2.3m", or "1.5h".
    """
    if seconds < 60:
        return f"{seconds:.3f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}m"
    return f"{seconds / 3600:.2f}h"

# =============================================================================
# Console Table Rendering (dependency-free)
# =============================================================================

def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    title: str = "",
    max_width: int = 120,
) -> str:
    """
    Render a well-formatted ASCII table suitable for the console.

    Args:
        headers: Column header names.
        rows: Iterable of row sequences (each row length == len(headers)).
        title: Optional title printed above the table.
        max_width: Maximum output width in characters.

    Returns:
        A multi-line string representation of the table.
    """
    rows = list(rows)
    n_cols = len(headers)

    # Normalize every cell to a display string.
    def _cell(value: Any) -> str:
        if value is None:
            return "N/A"
        return str(value)

    str_rows = [[_cell(c) for c in row] for row in rows]

    # Compute column widths (clamp to max_width budget).
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        if len(row) != n_cols:
            logger.warning(
                "Table row has %d cells, expected %d. Skipping row.",
                len(row),
                n_cols,
            )
            continue
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Respect max_width by clamping the widest columns.
    total = sum(col_widths) + 3 * n_cols + 1
    if total > max_width:
        overflow = total - max_width
        # Shrink the widest columns first.
        while overflow > 0:
            widest_idx = max(range(n_cols), key=lambda i: col_widths[i])
            if col_widths[widest_idx] <= 10:
                break
            col_widths[widest_idx] -= 1
            overflow -= 1

    def _row_line(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(col_widths[i]))
        return " | ".join(parts).rstrip()

    def _separator() -> str:
        return "-+-".join("-" * w for w in col_widths)

    lines: List[str] = []
    if title:
        width = sum(col_widths) + 3 * n_cols + 1
        lines.append("=" * width)
        lines.append(f"  {title}")
        lines.append("=" * width)

    lines.append(_row_line(headers))
    lines.append(_separator())
    for row in str_rows:
        if len(row) == n_cols:
            lines.append(_row_line(row))
    return "\n".join(lines)

# =============================================================================
# Relevance Scoring (lexical proxy, no ground truth required)
# =============================================================================

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "what", "which", "how", "why", "when", "where", "do", "does",
    "did", "not", "at", "by", "from", "as", "your", "you", "we", "they", "he",
    "she", "i", "me", "my", "can", "could", "will", "would", "should", "have",
    "has", "had", "about", "please", "tell", "explain", "describe", "give",
}

def tokenize(text: str) -> set:
    """Lowercase, strip punctuation, and split into word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def lexical_overlap(query: str, document_text: str) -> float:
    """
    Compute a lightweight lexical relevance score between a query and a
    document chunk.

    Uses the Jaccard-like overlap of meaningful (non-stopword) tokens.
    This is a proxy used only when no ground-truth labels are available.

    Args:
        query: The user query string.
        document_text: The retrieved chunk text.

    Returns:
        A float in [0.0, 1.0] representing token overlap.
    """
    q_tokens = tokenize(query) - _STOPWORDS
    doc_tokens = tokenize(document_text) - _STOPWORDS

    if not q_tokens:
        return 0.0

    intersection = q_tokens & doc_tokens
    if not intersection:
        return 0.0

    # Weighted overlap: fraction of query tokens found in the document.
    return len(intersection) / len(q_tokens)

def is_duplicate_chunk(self_chunk_id: Any, other_chunk_id: Any) -> bool:
    """
    Check whether two retrieved chunk identifiers refer to the same chunk.

    Handles heterogeneous metadata where chunk_id may be an int, a tuple,
    or a (source, chunk_id) combination.

    Args:
        self_chunk_id: Identifier of the first chunk.
        other_chunk_id: Identifier of the second chunk.

    Returns:
        True if they represent the same chunk.
    """
    return self_chunk_id == other_chunk_id

def summarize_metrics(values: Sequence[float]) -> Dict[str, float]:
    """
    Compute basic summary statistics (mean, min, max, std) for a numeric list.

    Args:
        values: Sequence of numeric values.

    Returns:
        Dictionary with 'mean', 'min', 'max', 'std'.
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": round(mean, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(variance ** 0.5, 6),
    }

