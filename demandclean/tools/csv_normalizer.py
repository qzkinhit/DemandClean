"""
CSV format pre-cleaning tool
============================

First preprocessing step after data is handed to DemandClean: normalize
numeric formatting.

Background:
    Raw dirty CSVs often have inconsistent numeric formatting, e.g.:
    - Integer values stored as floats: "1.0", "4.0", "10.0"
    - A single column mixing "1" and "1.0"

    RAHA compares strings at the cell level on the raw CSV, so these cosmetic
    differences get misclassified as errors.

Approach:
    Self-contained format pre-cleaning (no reference to the clean data):
    - For each column, check whether it is integer-valued (all valid entries
      are integers).
    - If so, normalize "1.0" -> "1".
    - This is standard preprocessing and does not peek at the ground truth.

    Downstream evaluation (evaluation/paper_score.py) counts format unification as part of the
    cleaning quality.

Usage:
    from demandclean.tools.csv_normalizer import normalize_dirty_format

    # Pass in only the dirty data; returns a format-normalized DataFrame
    dirty_normalized = normalize_dirty_format(dirty_df)

    # Or operate on files directly
    norm_path = normalize_dirty_to_file(dirty_path)
"""

import os
import tempfile
from typing import Optional, Set

import pandas as pd

# Common missing-value / placeholder tokens; skipped without affecting column-level inference.
_NA_LIKE_VALUES: Set[str] = {
    '', 'nan', 'NaN', 'NAN', 'null', 'NULL', 'none', 'None', 'NONE',
    'empty', 'Empty', 'EMPTY', 'na', 'NA', 'N/A', 'n/a', '?', '-', '.',
}


def normalize_dirty_format(
    dirty_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Self-contained numeric format pre-cleaning for a dirty DataFrame.

    Rules (per column):
      1. Skip NaN / blanks / common missing tokens (e.g. "?", "empty", "nan").
      2. Check whether every valid numeric value in the column is an integer.
      3. If so, normalize "1.0" -> "1".
      4. Non-numeric columns and columns containing true decimals are left alone.

    Args:
        dirty_df: dirty DataFrame
        verbose: whether to print summary stats

    Returns:
        The normalized dirty DataFrame (deep copy; the original is unchanged).
    """
    result = dirty_df.copy()
    total_normalized = 0

    for col in result.columns:
        col_loc = result.columns.get_loc(col)
        col_normalized = 0

        # Collect non-null values to decide whether this is an "all-integer" column
        non_null_mask = result[col].notna()
        if non_null_mask.sum() == 0:
            continue

        values = result.loc[non_null_mask, col]
        all_integer = True
        has_numeric = False

        for val in values:
            s = str(val).strip()
            # Skip missing tokens
            if s in _NA_LIKE_VALUES:
                continue
            try:
                f = float(s)
                if f != f:  # NaN
                    continue
                has_numeric = True
                if f != int(f):
                    # A true decimal (e.g. 3.14) is present; skip the whole column
                    all_integer = False
                    break
            except (ValueError, TypeError):
                # Non-numeric (e.g. the string "Beer"): skip this value and continue.
                # Don't abort column-level inference so a few stray dirty strings
                # don't block normalization of the rest.
                continue

        if not all_integer or not has_numeric:
            continue

        # Column is integer-valued overall; normalize "1.0" -> "1"
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
        print(f"  [format pre-clean] normalized {total_normalized} integer-as-float values (e.g. \"1.0\" -> \"1\")")

    return result


def normalize_cell_format(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Normalize numeric formatting cell by cell.

    Unlike normalize_dirty_format (which decides per column), this method
    operates per cell:
    - "1.0" -> "1" (integer values stored as floats become ints)
    - "3.14" stays unchanged (true decimals)
    - non-numeric strings stay unchanged

    Use case: when a dirty CSV mixes integers with injected float-valued dirty
    entries in the same column, the per-column method skips the whole column
    upon seeing a decimal; this per-cell method normalizes "1.0" -> "1" in
    clean rows while leaving anomalous "142.3" untouched.

    Args:
        df: input DataFrame (dtype=str)
        verbose: whether to print summary stats

    Returns:
        The normalized DataFrame (deep copy).
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
                # Only normalize when the value is an integer (e.g. 1.0, -343.0)
                # and the string contains a "."
                if f == int(f) and '.' in val:
                    int_str = str(int(f))
                    if val != int_str:
                        result.iat[row_idx, col_idx] = int_str
                        total_normalized += 1
            except (ValueError, TypeError, OverflowError):
                continue

    if verbose and total_normalized > 0:
        print(f"  [per-cell format normalization] normalized {total_normalized} values (e.g. \"1.0\" -> \"1\")")

    return result


def normalize_dirty_to_file(
    dirty_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Read a dirty CSV, run format pre-cleaning, and write the result to a file.

    Args:
        dirty_path: path to the dirty CSV
        output_path: output path; uses a temporary file when None
        verbose: whether to print summary stats

    Returns:
        Path to the pre-cleaned CSV.
    """
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)

    normalized = normalize_dirty_format(dirty_df, verbose=verbose)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.csv', prefix='normalized_dirty_')
        os.close(fd)

    normalized.to_csv(output_path, index=False)

    if verbose:
        print(f"  [format pre-clean] wrote: {output_path}")

    return output_path
