"""
Runner of Raha (error detection, SIGMOD 2019) and Baran (error correction, SIGMOD 2020).

Both learn from labeled tuples. The ground truth used is LABELING_BUDGET tuples.

Wide tables: when a table has more than MAX_COLUMNS_PER_SPLIT columns, the columns are split
into groups, every group is cleaned separately and the results are merged.

Usage:
    python run_raha_baran_base.py --dirty_path <dirty table> --clean_path <clean table>

Example:
    python run_raha_baran_base.py \\
        --dirty_path data/hospitals/dirty_index.csv \\
        --clean_path data/hospitals/clean_index.csv \\
        --task_name hospitals_raha_baran \\
        --labeling_budget 20 \\
        --output_path results/baselines/raha_baran/
"""

import os
import sys
import argparse
import time
import logging
import tempfile
import shutil
import multiprocessing
import gc
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

# Put the baselines root first on the path (ahead of the tools package of methods/Baran_Raha)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'Methods/Baran_Raha/'))

# Format normalization: the input preprocessing of DemandClean, so that format differences are not taken as syntactic errors
from tools.csv_normalizer import normalize_dirty_format

# Column-split threshold of wide tables
# vicinity_models of Baran/Raha is O(n^2), wider tables are split
MAX_COLUMNS_PER_SPLIT = 30

# Default cap of parallel processes (multiprocessing.Pool of Raha uses every CPU core by default, wide tables run out of memory)
DEFAULT_MAX_WORKERS = 12


def limit_pool_workers(max_workers: int):
    """
    Cap the default number of processes of multiprocessing.Pool to bound the memory use.
    The official code stays unchanged, the return value of os.cpu_count() is replaced.
    Pool() calls os.cpu_count() when processes=None.
    """
    _original_cpu_count = os.cpu_count

    def _limited_cpu_count():
        real_count = _original_cpu_count()
        return min(max_workers, real_count) if real_count else max_workers

    os.cpu_count = _limited_cpu_count


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """Logger that writes to the console and to a file."""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def split_columns(columns: List[str], index_col: str, max_cols: int) -> List[List[str]]:
    """
    Split the columns into groups of at most max_cols columns.

    Args:
        columns: all column names
        index_col: index column (part of every group)
        max_cols: maximum columns per group (without the index column)

    Returns:
        column groups, each with the index column and a part of the feature columns
    """
    # Without the index column
    feature_cols = [c for c in columns if c != index_col]

    # Split by max_cols
    groups = []
    for i in range(0, len(feature_cols), max_cols):
        group_cols = [index_col] + feature_cols[i:i + max_cols]
        groups.append(group_cols)

    return groups


def clean_single_split(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    columns: List[str],
    split_idx: int,
    task_name: str,
    temp_dir: str,
    cleaner,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean one column group.

    Args:
        dirty_df: full dirty table
        clean_df: full clean table
        columns: columns of the group (with the index column)
        split_idx: index of the group
        task_name: task name
        temp_dir: temporary directory
        cleaner: BaranRahaWrapper instance
        logger: logger

    Returns:
        (cleaned group, run information)
    """
    index_col = columns[0]  # The first column is the index column

    # Extract the group
    dirty_subset = dirty_df[columns].copy()
    clean_subset = clean_df[columns].copy()

    # Temporary files
    split_name = f"{task_name}_split{split_idx}"
    dirty_path = os.path.join(temp_dir, f"{split_name}_dirty.csv")
    clean_path = os.path.join(temp_dir, f"{split_name}_clean.csv")
    output_path = os.path.join(temp_dir, f"{split_name}_cleaned.csv")

    dirty_subset.to_csv(dirty_path, index=False)
    clean_subset.to_csv(clean_path, index=False)

    logger.info(f"  split {split_idx}: {len(columns)-1} columns ({columns[1]}...{columns[-1]})")

    # Run Baran/Raha (stderr suppressed, dBoost raises BrokenPipeError otherwise)
    try:
        import io
        import contextlib

        # Suppress stderr (the "Discarding" messages of dBoost trigger BrokenPipeError)
        with contextlib.redirect_stderr(io.StringIO()):
            repaired_df, info = cleaner.clean(
                dirty_path=dirty_path,
                clean_path=clean_path,
                output_path=output_path,
                task_name=split_name,
                index_attribute=index_col
            )
        return repaired_df, info
    except Exception as e:
        logger.warning(f"  split {split_idx} failed: {e}, dirty data kept")
        return dirty_subset, {'error': str(e)}


def merge_splits(
    base_df: pd.DataFrame,
    split_results: List[pd.DataFrame],
    column_groups: List[List[str]],
    index_col: str
) -> pd.DataFrame:
    """
    Merge the cleaned column groups.

    Args:
        base_df: original table (column order)
        split_results: cleaned groups
        column_groups: columns of every group
        index_col: index column

    Returns:
        merged DataFrame
    """
    # Start from the first group
    result = split_results[0].set_index(index_col)

    # Merge the other groups
    for i, (split_df, cols) in enumerate(zip(split_results[1:], column_groups[1:]), 1):
        split_df = split_df.set_index(index_col)
        # Non-index columns only
        feature_cols = [c for c in cols if c != index_col]
        for col in feature_cols:
            if col in split_df.columns:
                result[col] = split_df[col]

    # Reset the index and restore the column order
    result = result.reset_index()
    original_cols = [c for c in base_df.columns if c in result.columns]
    result = result[original_cols]

    return result


def run_with_column_split(
    dirty_path: str,
    clean_path: str,
    output_file: str,
    task_name: str,
    index_attribute: str,
    cleaner,
    logger: logging.Logger,
    max_columns: int = MAX_COLUMNS_PER_SPLIT
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean a wide table group by group.

    Args:
        dirty_path: path of the dirty table
        clean_path: path of the clean table
        output_file: output file
        task_name: task name
        index_attribute: index column
        cleaner: BaranRahaWrapper instance
        logger: logger
        max_columns: maximum columns per group

    Returns:
        (merged result, run information)
    """
    # Read the data
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    n_cols = len(dirty_df.columns)
    logger.info(f"wide table: {n_cols} columns, cleaned in column groups")

    # Split the columns
    column_groups = split_columns(
        list(dirty_df.columns),
        index_attribute,
        max_columns
    )
    n_splits = len(column_groups)
    logger.info(f"{n_splits} groups of at most {max_columns} columns")

    # Temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"baran_split_{task_name}_")
    logger.info(f"temporary directory: {temp_dir}")

    try:
        split_results = []
        total_detected = 0
        total_corrected = 0

        for i, cols in enumerate(column_groups):
            repaired_df, info = clean_single_split(
                dirty_df, clean_df, cols, i, task_name,
                temp_dir, cleaner, logger
            )
            split_results.append(repaired_df)
            total_detected += info.get('detected_cells', 0)
            total_corrected += info.get('corrected_cells', 0)

            # Explicit garbage collection frees the large objects of Baran/Raha (vicinity_models)
            gc.collect()

        # Merge the results
        logger.info("merging the groups...")
        merged_df = merge_splits(dirty_df, split_results, column_groups, index_attribute)

        # Save the result
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        logger.info(f"merged result saved: {output_file}")

        # Run information
        combined_info = {
            'ground_truth_cost': cleaner.get_ground_truth_cost() * n_splits,
            'method': 'Baran_Raha_ColumnSplit',
            'type': 'data-oriented',
            'auto_level': 3,
            'n_splits': n_splits,
            'total_detected_cells': total_detected,
            'total_corrected_cells': total_corrected,
            'columns_per_split': max_columns
        }

        return merged_df, combined_info

    finally:
        # Remove the temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"temporary directory removed: {temp_dir}")
        except Exception as e:
            logger.warning(f"could not remove the temporary directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Baran/Raha data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_raha_baran_base.py --dirty_path data/hospitals/dirty_index.csv --clean_path data/hospitals/clean_index.csv

Arguments:
  --labeling_budget: number of labeled tuples (20)
  --max_columns: maximum columns per column group (40)

Notes:
  - the ground truth used is labeling_budget tuples
  - with clean_path the sampled tuples are labeled from the clean table
  - tables with more than 40 columns are cleaned in column groups
        """
    )

    # Data paths
    parser.add_argument('--dirty_path', type=str, default='../data/breast_cancer/dirty_index.csv',
                        help='path of the dirty table')
    parser.add_argument('--clean_path', type=str, default='../data/breast_cancer/clean_index.csv',
                        help='path of the clean table (labels of the sampled tuples)')

    # Task
    parser.add_argument('--task_name', type=str, default='breast_cancer_raha_baran',
                        help='task name')
    parser.add_argument('--output_path', type=str, default='../../results/raha_baran/',
                        help='output directory')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='index column')

    # Raha and Baran
    parser.add_argument('--labeling_budget', type=int, default=20,
                        help='number of labeled tuples (20)')
    parser.add_argument('--classification_model', type=str, default='ABC',
                        choices=['ABC', 'DTC', 'GBC', 'GNB', 'KNC', 'SGDC', 'SVC'],
                        help='classifier (ABC)')
    parser.add_argument('--min_correction_probability', type=float, default=0.0,
                        help='minimum probability of a correction candidate (0.0)')
    parser.add_argument('--min_correction_occurrence', type=int, default=2,
                        help='minimum occurrence of a correction (2)')

    # Wide tables
    parser.add_argument('--max_columns', type=int, default=MAX_COLUMNS_PER_SPLIT,
                        help=f'maximum columns per column group ({MAX_COLUMNS_PER_SPLIT})')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'cap of Raha processes, bounds the memory use ({DEFAULT_MAX_WORKERS})')

    # Label
    parser.add_argument('--label_column', type=str, default='style',
                        help='label column')

    parser.add_argument('--verbose', action='store_true',
                        help='print progress information')
    parser.add_argument('--use_split', action='store_true',
                        help='60/20/20 split of DemandClean (seed 42)')

    args = parser.parse_args()

    # Cap the Raha processes before Baran/Raha is imported
    limit_pool_workers(args.max_workers)

    # Column-split threshold from the command line
    max_columns = args.max_columns

    # Output directory
    result_path = os.path.join(args.output_path, args.task_name)
    os.makedirs(result_path, exist_ok=True)

    # Logging
    logger = setup_logging(result_path, args.task_name)

    # Start time
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"started: {start_datetime}")
    logger.info("=" * 60)
    logger.info("Raha and Baran")
    logger.info("=" * 60)
    logger.info(f"dirty table: {args.dirty_path}")
    logger.info(f"clean table: {args.clean_path or 'not given'}")
    logger.info(f"task: {args.task_name}")
    logger.info(f"labeled tuples: {args.labeling_budget}")
    logger.info(f"column-split threshold: {max_columns}")
    logger.info(f"process cap: {args.max_workers}")
    logger.info("-" * 60)

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': args.labeling_budget}

    # =========================================================================
    # Format normalization of the numeric values of the dirty CSV, as in the DemandClean input
    # Integer-valued floats such as "1.0" become "1"
    # So that RAHA does not take format differences (dirty "1.0" against clean "1") as syntactic errors
    # =========================================================================
    normalized_dirty_path = None
    actual_dirty_path = args.dirty_path  # Dirty path passed to Baran

    try:
        dirty_raw = pd.read_csv(args.dirty_path, dtype=str, keep_default_na=False)
        dirty_normalized = normalize_dirty_format(dirty_raw, verbose=True)

        # Any change
        n_changed = (dirty_raw != dirty_normalized).sum().sum()
        if n_changed > 0:
            # Write the normalized temporary file
            normalized_dirty_path = os.path.join(result_path, f"{args.task_name}_dirty_normalized.csv")
            dirty_normalized.to_csv(normalized_dirty_path, index=False)
            actual_dirty_path = normalized_dirty_path
            logger.info(f"[format normalization] {n_changed} cells normalized, dirty input: {normalized_dirty_path}")
        else:
            logger.info("[format normalization] nothing to normalize")
    except Exception as e:
        logger.warning(f"[format normalization] failed, original dirty input kept: {e}")

    try:
        from Methods.Baran_Raha.baran_raha_wrapper import BaranRahaWrapper

        cleaner = BaranRahaWrapper(
            labeling_budget=args.labeling_budget,
            classification_model=args.classification_model,
            min_correction_candidate_probability=args.min_correction_probability,
            min_correction_occurrence=args.min_correction_occurrence,
            verbose=args.verbose
        )

        # Column count decides about the column split
        dirty_df = pd.read_csv(actual_dirty_path)
        n_cols = len(dirty_df.columns)

        if n_cols > max_columns + 1:  # +1 for index column
            # Wide table: column groups
            logger.info(f"{n_cols} columns, above the threshold ({max_columns}), column groups enabled")
            repaired_df, clean_info = run_with_column_split(
                dirty_path=actual_dirty_path,
                clean_path=args.clean_path,
                output_file=output_file,
                task_name=args.task_name,
                index_attribute=args.index_attribute,
                cleaner=cleaner,
                logger=logger,
                max_columns=max_columns
            )
        else:
            # Regular run
            logger.info(f"{n_cols} columns, within the threshold")
            repaired_df, clean_info = cleaner.clean(
                dirty_path=actual_dirty_path,
                clean_path=args.clean_path,
                output_path=output_file,
                task_name=args.task_name,
                index_attribute=args.index_attribute
            )
        success = True

    except ImportError as e:
        logger.error(f"import failed: {e}")
        logger.error("make sure the Baran/Raha modules are installed")
        logger.error("required: raha, mwparserfromhell, py7zr")
    except Exception as e:
        logger.error(f"error: {e}")
        import traceback
        traceback.print_exc()

    # Time
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"time: {elapsed_time:.2f} s")
    logger.info(f"status: {'success' if success else 'failure'}")

    if 'n_splits' in clean_info:
        logger.info(f"column groups: {clean_info['n_splits']}")
        logger.info(f"detected cells: {clean_info.get('total_detected_cells', 'N/A')}")
        logger.info(f"corrected cells: {clean_info.get('total_corrected_cells', 'N/A')}")

    logger.info(f"labeled tuples: {clean_info.get('labeling_budget', args.labeling_budget)}")
    logger.info(f"ground truth used: {clean_info.get('ground_truth_cost', args.labeling_budget)} labeled tuples")



    logger.info(f"\nresults saved to: {result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
