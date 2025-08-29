#!/usr/bin/env python3
"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib

This script fulfills the assignment requirements:

Task 1: Load & Explore
- Load a CSV dataset (if provided) or fall back to iris from scikit-learn.
- If scikit-learn is unavailable, generate a synthetic iris-like dataset.
- Display head(), dtypes, missing value counts.
- Clean data by filling missing numeric values with medians and categorical with modes.

Task 2: Basic Analysis
- Compute descriptive statistics via describe().
- Group by a categorical column (prefer "species") and compute mean of a numeric column.

Task 3: Visualization (Matplotlib, no seaborn)
- Line chart: trend of a numeric column over a pseudo time index (or real datetime if present).
- Bar chart: mean of a numeric column by category.
- Histogram: distribution of a numeric column.
- Scatter plot: relationship between two numeric columns.

Error Handling:
- try/except for file reading and optional scikit-learn import.

Usage:
    python3 pandas_matplotlib_assignment.py               # use iris (preferred) or synthetic fallback
    python3 pandas_matplotlib_assignment.py --csv data.csv  # use your own CSV

Outputs:
- Saves plots to PNG files in the current folder.
"""

from __future__ import annotations
import argparse
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def try_load_csv(path: Path) -> pd.DataFrame:
    """Attempt to read a CSV file with UTF-8; raise on failure."""
    df = pd.read_csv(path)
    return df


def load_iris_df() -> pd.DataFrame:
    """Load iris from scikit-learn; return as pandas DataFrame with 'species'."""
    try:
        from sklearn.datasets import load_iris
    except Exception:
        raise ImportError("scikit-learn not available")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Ensure friendly column names (sklearn already provides)
    if 'target' in df.columns:
        target_names = iris.target_names
        df['species'] = df['target'].map(dict(enumerate(target_names)))
        df.drop(columns=['target'], inplace=True)
    elif 'species' not in df.columns:
        df['species'] = pd.Categorical(np.random.choice(['setosa','versicolor','virginica'], size=len(df)))
    return df


def make_synthetic_iris_like(n: int = 150, random_state: int = 42) -> pd.DataFrame:
    """Generate a simple synthetic iris-like dataset if sklearn is unavailable."""
    rng = np.random.default_rng(random_state)
    species = np.array(['setosa', 'versicolor', 'virginica'])
    sp = rng.choice(species, size=n, replace=True)
    # rough ranges
    sepal_length = rng.normal(5.8, 0.8, size=n)
    sepal_width  = rng.normal(3.0, 0.4, size=n)
    petal_length = rng.normal(3.7, 1.5, size=n)
    petal_width  = rng.normal(1.1, 0.5, size=n)
    df = pd.DataFrame({
        'sepal length (cm)': sepal_length,
        'sepal width (cm)':  sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)':  petal_width,
        'species': sp
    })
    return df


def select_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Pick reasonable defaults:
    - categorical: prefer 'species' or any object/category column
    - num_x, num_y: pick two numeric columns for scatter; also use one for line/hist/bar
    """
    cat = None
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            cat = col
            break
    if cat is None:
        # if no category, create one by binning first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric columns found to create a category.")
        first = num_cols[0]
        cat = f"{first}_bin"
        df[cat] = pd.qcut(df[first], q=3, labels=["low", "mid", "high"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Need at least two numeric columns for analysis and scatter plot.")
    x = num_cols[0]
    y = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    return cat, x, y


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric with median and categorical with mode."""
    df = df.copy()
    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
    # Categorical / object
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill)
    # Drop rows that are entirely NA (unlikely)
    df.dropna(how="all", inplace=True)
    return df


def explore(df: pd.DataFrame) -> None:
    """Print basic exploration info."""
    print("\n=== HEAD ===")
    print(df.head(), "\n")
    print("=== DTYPES ===")
    print(df.dtypes, "\n")
    print("=== Missing values per column ===")
    print(df.isna().sum(), "\n")


def analyze(df: pd.DataFrame, cat: str, num_a: str, num_b: str) -> None:
    """Print basic statistics and groupby results."""
    print("=== DESCRIBE (numeric) ===")
    print(df.describe(numeric_only=True), "\n")

    # Grouping by categorical
    print(f"=== GROUPBY mean of {num_a} by {cat} ===")
    print(df.groupby(cat)[num_a].mean().sort_values(ascending=False), "\n")

    # Simple observation example
    grp = df.groupby(cat)[num_a].mean().sort_values(ascending=False)
    top_cat = grp.index[0]
    print(f"Observation: On average, '{top_cat}' has the highest mean {num_a:.} among groups.\n")


def ensure_time_index(df: pd.DataFrame) -> pd.Series:
    """
    Ensure a DatetimeIndex for line plot:
    - If the df already has a datetime-like column, use the first one.
    - Else create a synthetic daily date range starting at 2020-01-01.
    """
    # try to find a datetime column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return pd.to_datetime(df[col])
    # synthetic index
    return pd.date_range(start="2020-01-01", periods=len(df), freq="D")


def visualize(df: pd.DataFrame, cat: str, num_a: str, num_b: str, basename: str = "plot") -> None:
    """Create the four required plots and save them as PNG files."""
    dates = ensure_time_index(df)
    # Line chart: trend over pseudo time for num_a (rolling mean as example)
    plt.figure()
    series = pd.Series(df[num_a].values, index=dates).rolling(window=7, min_periods=1).mean()
    series.plot()
    plt.title(f"Trend of {num_a} over time (7-day rolling mean)")
    plt.xlabel("Date")
    plt.ylabel(num_a)
    plt.tight_layout()
    plt.savefig(f"{basename}_line.png")

    # Bar chart: mean of num_a by category
    plt.figure()
    df.groupby(cat)[num_a].mean().sort_values().plot(kind="bar")
    plt.title(f"Average {num_a} by {cat}")
    plt.xlabel(cat)
    plt.ylabel(f"Mean {num_a}")
    plt.tight_layout()
    plt.savefig(f"{basename}_bar.png")

    # Histogram: distribution of num_a
    plt.figure()
    plt.hist(df[num_a].values, bins=20)
    plt.title(f"Distribution of {num_a}")
    plt.xlabel(num_a)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{basename}_hist.png")

    # Scatter: num_a vs num_b
    plt.figure()
    plt.scatter(df[num_a].values, df[num_b].values)
    plt.title(f"{num_a} vs {num_b}")
    plt.xlabel(num_a)
    plt.ylabel(num_b)
    plt.tight_layout()
    plt.savefig(f"{basename}_scatter.png")

    # Show all plots when running interactively
    try:
        plt.show()
    except Exception:
        pass


def load_dataset(csv_path: Optional[Path]) -> Tuple[pd.DataFrame, str]:
    """
    Load dataset from CSV if provided, otherwise iris (or synthetic fallback).
    Returns df and a name tag.
    """
    if csv_path:
        print(f"Loading CSV: {csv_path}")
        df = try:
            df = try_load_csv(csv_path)
            return df, csv_path.name
        except FileNotFoundError:
            print("Error: file not found. Falling back to iris dataset.")
        except pd.errors.EmptyDataError:
            print("Error: empty CSV file. Falling back to iris dataset.")
        except Exception as e:
            print(f"Error reading CSV ({e}). Falling back to iris dataset.")

    # Try iris from sklearn, then synthetic
    try:
        df = load_iris_df()
        return df, "iris"
    except ImportError:
        print("scikit-learn not available; generating synthetic iris-like dataset.")
        df = make_synthetic_iris_like()
        return df, "synthetic_iris_like"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pandas + Matplotlib Assignment")
    parser.add_argument("--csv", type=str, help="Path to a CSV dataset", default=None)
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else None

    # Load
    df, tag = load_dataset(csv_path)

    # Explore
    explore(df)

    # Clean
    df_clean = clean_dataset(df)

    # Choose columns
    cat, num_a, num_b = select_columns(df_clean)
    print(f"Using categorical='{cat}', numeric A='{num_a}', numeric B='{num_b}'\n")

    # Analyze
    analyze(df_clean, cat, num_a, num_b)

    # Visualize
    visualize(df_clean, cat, num_a, num_b, basename=f"{tag}")

    print("Saved figures:", ", ".join([f"{tag}_line.png", f"{tag}_bar.png", f"{tag}_hist.png", f"{tag}_scatter.png"]))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
