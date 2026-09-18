"""
viz_helpers.py
Reusable visualization helper functions for Python/R-style data workflows.

Built on matplotlib + seaborn. Each function follows a consistent pattern:
    plot_<type>(data, x, y, ..., title=None, save_path=None, ax=None) -> Axes

Design goals:
- Consistent styling across all plots (set once via `set_style`)
- Every function returns the Axes object so plots can be composed/customized further
- Every function can optionally save to disk (save_path) or render inline
- Works with pandas DataFrames, dicts, or plain lists/arrays
"""

from __future__ import annotations
import os
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

DataLike = Union[pd.DataFrame, dict, None]


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
def set_style(
    theme: str = "whitegrid",
    palette: str = "deep",
    font_scale: float = 1.0,
    figsize: tuple = (8, 5),
) -> None:
    """Apply a consistent theme to all subsequent plots. Call once at the top of a script/notebook."""
    sns.set_theme(style=theme, palette=palette, font_scale=font_scale)
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 13


def _finalize(ax: plt.Axes, title: Optional[str], xlabel: Optional[str],
              ylabel: Optional[str], save_path: Optional[str], rotate_x: int = 0) -> plt.Axes:
    """Shared finishing touches: labels, title, save-to-disk, layout."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right")
    ax.figure.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Core chart types
# ---------------------------------------------------------------------------
def plot_line(data: DataLike, x: str, y: Union[str, Sequence[str]], title: str = None,
              xlabel: str = None, ylabel: str = None, save_path: str = None,
              ax: plt.Axes = None, markers: bool = True) -> plt.Axes:
    """Line chart. `y` can be a single column or a list of columns for multi-series."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    y_cols = [y] if isinstance(y, str) else list(y)
    for col in y_cols:
        sns.lineplot(data=df, x=x, y=col, marker="o" if markers else None, ax=ax, label=col)
    if len(y_cols) > 1:
        ax.legend(title=None)
    return _finalize(ax, title, xlabel or x, ylabel or (y if isinstance(y, str) else "Value"), save_path)


def plot_bar(data: DataLike, x: str, y: str, title: str = None, xlabel: str = None,
             ylabel: str = None, save_path: str = None, ax: plt.Axes = None,
             horizontal: bool = False) -> plt.Axes:
    """Bar chart, vertical by default (set horizontal=True to flip)."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    if horizontal:
        sns.barplot(data=df, x=y, y=x, ax=ax)
    else:
        sns.barplot(data=df, x=x, y=y, ax=ax)
    return _finalize(ax, title, xlabel or x, ylabel or y, save_path, rotate_x=0 if horizontal else 30)


def plot_scatter(data: DataLike, x: str, y: str, hue: str = None, size: str = None,
                  title: str = None, xlabel: str = None, ylabel: str = None,
                  save_path: str = None, ax: plt.Axes = None) -> plt.Axes:
    """Scatter plot with optional hue (color grouping) and size encoding."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, ax=ax, alpha=0.8)
    return _finalize(ax, title, xlabel or x, ylabel or y, save_path)


def plot_histogram(data: DataLike, column: str, bins: int = 20, kde: bool = True,
                    title: str = None, xlabel: str = None, ylabel: str = "Count",
                    save_path: str = None, ax: plt.Axes = None) -> plt.Axes:
    """Histogram with optional KDE overlay for distribution shape."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    sns.histplot(data=df, x=column, bins=bins, kde=kde, ax=ax)
    return _finalize(ax, title, xlabel or column, ylabel, save_path)


def plot_box(data: DataLike, x: str, y: str, title: str = None, xlabel: str = None,
             ylabel: str = None, save_path: str = None, ax: plt.Axes = None) -> plt.Axes:
    """Box plot for comparing distributions across categories."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    return _finalize(ax, title, xlabel or x, ylabel or y, save_path, rotate_x=30)


def plot_heatmap(data: DataLike, title: str = None, save_path: str = None,
                  ax: plt.Axes = None, annot: bool = True, cmap: str = "viridis") -> plt.Axes:
    """Heatmap, typically used for correlation matrices (pass df.corr()) or pivot tables."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    sns.heatmap(df, annot=annot, cmap=cmap, fmt=".2f", ax=ax, linewidths=0.5)
    return _finalize(ax, title, None, None, save_path)


def plot_pie(data: DataLike, labels: str, values: str, title: str = None,
             save_path: str = None, ax: plt.Axes = None) -> plt.Axes:
    """Pie chart from a label column and a value column."""
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    ax = ax or plt.gca()
    ax.pie(df[values], labels=df[labels], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    return _finalize(ax, title, None, None, save_path)


def plot_grid(plot_fns: Sequence[callable], ncols: int = 2, figsize: tuple = (12, 8),
              save_path: str = None) -> plt.Figure:
    """
    Compose multiple plots into a grid.
    `plot_fns` is a list of zero-arg callables, each of which draws onto its own ax
    e.g. plot_grid([lambda ax=None: plot_line(df, 'date', 'sales', ax=ax), ...])
    """
    n = len(plot_fns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()
    for fn, ax in zip(plot_fns, axes_flat):
        fn(ax=ax)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show():
    """Convenience re-export so callers don't need a separate matplotlib import just to render."""
    plt.show()
