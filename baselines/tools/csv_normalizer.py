"""
CSV format pre-cleaning utility
===============================

First preprocessing step after data is handed to DemandClean: normalize numeric formatting.

Background:
    Numeric columns in the raw dirty CSV may use inconsistent formats, for example:
    - Integer values stored as floats: "1.0", "4.0", "10.0"
    - Mixed "1" and "1.0" within the same column

    RAHA performs string-level comparison on the raw CSV, so these formatting differences
    would be misclassified as errors.

Approach:
    Self-contained format normalization (does not peek at the clean data):
    - For each column, check whether every valid value is an integer value.
    - If so, rewrite "1.0" -> "1" to a uniform integer format.
    - This is a standard preprocessing step; it does not "peek at the answer".

    During later scoring (evaluation/paper_score.py) the unified format is considered part of the cleaning result.

Usage:
    from demandclean.tools.csv_normalizer import normalize_dirty_format

    # Normalize a dirty DataFrame.
    dirty_normalized = normalize_dirty_format(dirty_df)

    # Or operate directly on a file.
    norm_path = normalize_dirty_to_file(dirty_path)
"""

import os
import tempfile
from typing import Optional, Set

import pandas as pd

# Common missing-value / placeholder markers; skipped when judging a column.
_NA_LIKE_VALUES: Set[str] = {
    '', 'nan', 'NaN', 'NAN', 'null', 'NULL', 'none', 'None', 'NONE',
    'empty', 'Empty', 'EMPTY', 'na', 'NA', 'N/A', 'n/a', '?', '-', '.',
}


def normalize_dirty_format(
    dirty_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply self-contained numeric-format normalization to a dirty DataFrame.

    Rules (applied per column):
      1. Skip NaN / empty / common missing markers (e.g. "?", "empty", "nan").
      2. For the remaining values, check whether they are all integers.
      3. If so, rewrite "1.0" -> "1" to a uniform integer format.
      4. Leave non-numeric columns and columns with real decimals untouched.

    Args:
        dirty_df: dirty DataFrame
        verbose: whether to print statistics

    Returns:
        a deep copy of the dirty DataFrame with normalized formatting
    """
    result = dirty_df.copy()
    total_normalized = 0

    for col in result.columns:
        col_loc = result.columns.get_loc(col)
        col_normalized = 0

        # Collect the column's non-null values to decide whether it is an all-integer column.
        non_null_mask = result[col].notna()
        if non_null_mask.sum() == 0:
            continue

        values = result.loc[non_null_mask, col]
        all_integer = True
        has_numeric = False

        for val in values:
            s = str(val).strip()
            # Skip missing markers.
            if s in _NA_LIKE_VALUES:
                continue
            try:
                f = float(s)
                if f != f:  # NaN
                    continue
                has_numeric = True
                if f != int(f):
                    # Real decimal present (e.g. 3.14); leave the column alone.
                    all_integer = False
                    break
            except (ValueError, TypeError):
                # Non-numeric (e.g. the string "Beer"); skip this value and continue.
                # Do not abort the column-wide judgement, allow some non-numeric dirt to coexist.
                continue

        if not all_integer or not has_numeric:
            continue

        # The numeric part of this column is all integer-valued; normalize format: "1.0" -> "1".
        for i in result.index[non_null_mask]:
            s = str(result.at[i, col]).strip()
            if s in _NA_LIKE_VALUES:
                continue
            try:
                f = float(s)
                if f != f:
                    continue
                if f != int(f):
                    continue
                int_s = str(int(f))
                if s != int_s:
                    result.iat[result.index.get_loc(i), col_loc] = int_s
                    col_normalized += 1
            except (ValueError, TypeError):
                pass

        total_normalized += col_normalized

    if verbose and total_normalized > 0:
        print(f"  [Format pre-clean] normalized {total_normalized} integer-as-float values (e.g. \"1.0\" -> \"1\")")

    return result


def normalize_cell_format(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Cell-by-cell numeric format normalization.

    Unlike normalize_dirty_format (column-level decision), this one decides per cell:
    - "1.0" -> "1" (integer values stored as floats are normalized to integer format)
    - "3.14" unchanged (real decimals untouched)
    - Non-numeric strings unchanged

    Use case: when a dirty column mixes real integers with injected float dirt,
    the column-level method would skip the entire column because it sees a decimal,
    whereas the per-cell method normalizes "1.0" on good rows while keeping
    an outlier "142.3" untouched on bad rows.

    Args:
        df: input DataFrame (dtype=str)
        verbose: whether to print statistics

    Returns:
        a deep copy of the DataFrame with normalized formatting
    """
    result = df.copy()
    total_normalized = 0

    for col_idx in range(result.shape[1]):
        for row_idx in range(result.shape[0]):
            val = result.iat[row_idx, col_idx]
            if not isinstance(val, str) or val in _NA_LIKE_VALUES:
                continue
            try:
                f = float(val)
                if f != f:  # NaN
                    continue
                # Normalize only when the value is integer-valued (e.g. 1.0, -343.0) and the string contains '.'.
                if f == int(f) and '.' in val:
                    int_str = str(int(f))
                    if val != int_str:
                        result.iat[row_idx, col_idx] = int_str
                        total_normalized += 1
            except (ValueError, TypeError, OverflowError):
                continue

    if verbose and total_normalized > 0:
        print(f"  [Per-cell format normalization] normalized {total_normalized} values (e.g. \"1.0\" -> \"1\")")

    return result


def normalize_dirty_to_file(
    dirty_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Load a dirty CSV, apply format normalization, and write the result to disk.

    Args:
        dirty_path: path to the dirty CSV
        output_path: output path; if None, writes to a temp file
        verbose: whether to print statistics

    Returns:
        path to the normalized CSV
    """
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)

    normalized = normalize_dirty_format(dirty_df, verbose=verbose)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.csv', prefix='normalized_dirty_')
        os.close(fd)

    normalized.to_csv(output_path, index=False)

    if verbose:
        print(f"  [Format pre-clean] written: {output_path}")

    return output_path
