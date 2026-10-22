"""
Runner of HoloClean (VLDB 2017), data repair with a probabilistic graphical model.

HoloClean combines constraints, statistics and knowledge-base signals, runs fully automatically
and needs a PostgreSQL database.

Wide and long tables:
- more than MAX_COLUMNS columns: the columns are split into groups
- more than MAX_ROWS rows: the rows are split into chunks
- every part is cleaned separately and the results are merged

Usage:
    python run_holoclean_base.py --dirty_path <dirty table> --rule_path <constraint file>

Example:
    python run_holoclean_base.py \\
        --dirty_path data/hospitals/dirty_index.csv \\
        --rule_path data/hospitals/rules_baselines.txt \\
        --clean_path data/hospitals/clean_index.csv \\
        --task_name hospitals_holoclean \\
        --output_path results/baselines/holoclean/
"""

import os
import sys
import argparse
import time
import logging
import tempfile
import shutil
import gc
import pandas as pd
from typing import List, Tuple, Dict, Optional

# Paths of the baselines package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../Methods/HoloClean/')

# Split thresholds
MAX_COLUMNS_PER_SPLIT = 40  # Column-split threshold
MAX_ROWS_PER_SPLIT = 5000   # Row-split threshold

# Missing value inside HoloClean
HOLOCLEAN_NULL_REPR = '_nan_'
# Standard missing value
STANDARD_NULL_REPR = 'empty'


def normalize_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the _nan_ missing values of HoloClean by the standard empty."""
    return df.replace(HOLOCLEAN_NULL_REPR, STANDARD_NULL_REPR)


try:
    from tools.get_T_table import transform_csv_file
    from tools.insert_null import inject_missing_values
    import Methods.HoloClean as holoclean
    from Methods.HoloClean.detect import NullDetector, ViolationDetector
    from Methods.HoloClean.repair.featurize import (
        InitAttrFeaturizer, OccurAttrFeaturizer,
        FreqFeaturizer, ConstraintFeaturizer
    )
    HAS_HOLOCLEAN = True
except ImportError as e:
    HAS_HOLOCLEAN = False
    IMPORT_ERROR = str(e)


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


def parse_holoclean_dc_rules(rules_path):
    """
    Parse the rules of the HOLOCLEAN_DC section of a rule file.

    Args:
        rules_path: rule file

    Returns:
        list: denial constraints
    """
    dc_rules = []
    in_holoclean_section = False
    has_sections = False

    with open(rules_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Section markers present
    for line in lines:
        if line.strip().startswith('['):
            has_sections = True
            break

    if has_sections:
        # With sections, only the HOLOCLEAN_DC part
        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('[HOLOCLEAN_DC]'):
                in_holoclean_section = True
                continue
            elif line_stripped.startswith('[') and in_holoclean_section:
                # Next section reached, stop
                in_holoclean_section = False
                continue

            if in_holoclean_section and line_stripped and not line_stripped.startswith('#'):
                dc_rules.append(line_stripped)
    else:
        # Without sections, every non-empty non-comment line is a constraint
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                dc_rules.append(line_stripped)

    return dc_rules


def create_temp_dc_file(dc_rules, output_dir):
    """
    Write a temporary constraint file for HoloClean.

    Args:
        dc_rules: denial constraints
        output_dir: output directory

    Returns:
        str: path of the temporary file
    """
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    temp_path = os.path.join(output_dir, 'temp_dc_rules.txt')

    with open(temp_path, 'w', encoding='utf-8') as f:
        for rule in dc_rules:
            f.write(rule + '\n')

    return temp_path


def filter_rules_for_columns(dc_rules: List[str], columns: List[str]) -> List[str]:
    """
    Keep the denial constraints that involve only the columns of the current group.

    Args:
        dc_rules: all denial constraints
        columns: columns of the current group

    Returns:
        constraints of the group
    """
    filtered = []
    col_set = set(columns)
    for rule in dc_rules:
        # Constraint format: t1&t2&EQ(t1.col1,t2.col1)&...
        # Columns of the constraint
        rule_cols = set()
        for part in rule.split('&'):
            part = part.strip()
            if '(' in part and '.' in part:
                # colname of t1.colname or t2.colname
                import re
                col_refs = re.findall(r't\d+\.(\w+)', part)
                rule_cols.update(col_refs)
        # Keep the constraints whose columns all belong to the group
        if rule_cols and rule_cols.issubset(col_set):
            filtered.append(rule)
    return filtered


def _cleanup_holoclean_db(db_user: str, db_name: str, logger: logging.Logger,
                          db_host: str = '127.0.0.1', db_port: int = 5432,
                          db_pwd: str = 'abcd1234'):
    """
    Drop the auxiliary tables that HoloClean leaves in the database.
    The auxiliary tables (cell_distr, dk_cells and others) have fixed names without a task prefix,
    so consecutive runs on parts of a table collide.
    """
    # Fixed names of the auxiliary tables of HoloClean
    aux_tables = [
        'c_cells', 'dk_cells', 'cell_domain', 'pos_values',
        'cell_distr', 'inf_values_idx', 'inf_values_dom',
        'grounding', 'Feature', 'pos_values_idx'
    ]

    try:
        import psycopg2
        conn = psycopg2.connect(
            user=db_user, password=db_pwd,
            dbname=db_name, host=db_host, port=db_port
        )
        conn.autocommit = True
        cur = conn.cursor()
        for table in aux_tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"    database clean-up warning (the run continues): {e}")


def run_holoclean_on_subset(
    dirty_path: str,
    clean_path: str,
    output_path: str,
    task_name: str,
    index_attribute: str,
    dc_rules: List[str],
    args,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Run HoloClean on one part of a table.

    Args:
        dirty_path: dirty part
        clean_path: clean part
        output_path: output file
        task_name: unique task name (database table names)
        index_attribute: index column
        dc_rules: constraints of the part
        args: command-line arguments
        logger: logger

    Returns:
        repaired DataFrame, None on failure
    """
    from tools.get_T_table import transform_csv_file
    from tools.insert_null import inject_missing_values

    temp_dir = tempfile.mkdtemp(prefix=f"holo_{task_name}_")

    try:
        # Drop leftover auxiliary tables (fixed names collide across runs)
        _cleanup_holoclean_db(args.db_user, args.db_name, logger)

        # Normalize the missing values
        ori_dirty = os.path.join(temp_dir, "dirty_ori.csv")
        inject_missing_values(
            csv_file=dirty_path, output_file=ori_dirty,
            missing_value_in_ori_data='empty',
            missing_value_representation='', attributes_error_ratio=None
        )
        ori_clean = os.path.join(temp_dir, "clean_ori.csv")
        inject_missing_values(
            csv_file=clean_path, output_file=ori_clean,
            missing_value_in_ori_data='empty',
            missing_value_representation='', attributes_error_ratio=None
        )

        # Remove the index column
        dirty_data = pd.read_csv(ori_dirty)
        if index_attribute in dirty_data.columns:
            dirty_no_idx = dirty_data.drop(columns=[index_attribute])
        else:
            dirty_no_idx = dirty_data

        dirty_holo_path = os.path.join(temp_dir, "dirty_holo.csv")
        dirty_no_idx.to_csv(dirty_holo_path, index=False, encoding='utf-8')

        # Clean data in the HoloClean format
        clean_holo_path = os.path.join(temp_dir, "clean_holo.csv")
        transform_csv_file(ori_clean, clean_holo_path)

        clean_data = pd.read_csv(clean_path)
        clean_data_attributes = clean_data.columns.tolist()

        # Initialize HoloClean
        hc = holoclean.HoloClean(
            db_user=args.db_user, db_name=args.db_name,
            domain_thresh_1=0, domain_thresh_2=0,
            weak_label_thresh=args.weak_label_thresh,
            max_domain=10000, cor_strength=0.6, nb_cor_strength=0.8,
            epochs=args.epochs, weight_decay=0.01,
            learning_rate=args.learning_rate, threads=args.threads,
            batch_size=1, verbose=False,
            timeout=3 * 60000, feature_norm=False,
            weight_norm=False, print_fw=True
        ).session

        # Load the data
        safe_name = task_name.replace('-', '_').replace('.', '_')
        if safe_name[0].isdigit():
            safe_name = 't_' + safe_name
        hc.load_data(safe_name, dirty_holo_path)

        # Load the rules
        dc_file_path = None
        if dc_rules:
            dc_file_path = os.path.join(temp_dir, "dc_rules.txt")
            with open(dc_file_path, 'w') as f:
                for rule in dc_rules:
                    f.write(rule + '\n')
            hc.load_dcs(dc_file_path)
            hc.ds.set_constraints(hc.get_dcs())

        # Detect errors
        detectors = [NullDetector()]
        if dc_file_path:
            detectors.append(ViolationDetector())

        try:
            hc.detect_errors(detectors)
        except (ValueError, Exception) as e:
            if "Wrong number of items" in str(e) or "empty" in str(e).lower():
                logger.info(f"    no errors in this part, data returned unchanged")
                return dirty_data
            raise

        # Domain, featurization, repair
        hc.setup_domain()
        featurizers = [InitAttrFeaturizer(), OccurAttrFeaturizer(), FreqFeaturizer()]
        if dc_file_path:
            featurizers.append(ConstraintFeaturizer())
        hc.repair_errors(featurizers)

        # Repair result
        try:
            hc.evaluate(
                fpath=clean_holo_path, tid_col='tid',
                attr_col='attribute', val_col='correct_val',
                output_csv_path=output_path,
                attrubte_list=clean_data_attributes,
                clean_data=clean_data, index_attribute=index_attribute
            )
            return pd.read_csv(output_path)
        except Exception:
            # Fallback: query the database
            query = "SELECT * FROM {} ORDER BY _tid_".format(
                hc.ds.raw_data.name + '_repaired')
            data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
            repaired_df = pd.DataFrame(data, columns=columns)
            if '_tid_' in repaired_df.columns:
                repaired_df.drop(columns=['_tid_'], inplace=True)
            if clean_data is not None and index_attribute in clean_data.columns:
                if len(clean_data) == len(repaired_df):
                    repaired_df[index_attribute] = clean_data[index_attribute].values
            repaired_df.to_csv(output_path, index=False, encoding='utf-8')
            return repaired_df

    except Exception as e:
        err_msg = str(e).lower()
        # Without cells to repair HoloClean creates neither the _vid_ column nor the cell_distr table
        # The dirty data is returned then (nothing to repair)
        if '_vid_' in err_msg or 'cell_distr' in err_msg or 'inferring on 0' in err_msg:
            logger.info(f"    nothing to repair in this part, data returned unchanged")
            return pd.read_csv(dirty_path)  # Return the dirty data
        logger.warning(f"    cleaning of the part failed: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_with_column_split(
    dirty_path: str, clean_path: str, output_file: str,
    task_name: str, index_attribute: str,
    dc_rules: List[str], args, logger: logging.Logger,
    max_columns: int
) -> Tuple[pd.DataFrame, Dict]:
    """Clean a wide table in column groups."""
    dirty_df = pd.read_csv(dirty_path)
    n_cols = len([c for c in dirty_df.columns if c != index_attribute])
    logger.info(f"column groups: {n_cols} feature columns, at most {max_columns} per group")

    # Split the columns
    feature_cols = [c for c in dirty_df.columns if c != index_attribute]
    column_groups = []
    for i in range(0, len(feature_cols), max_columns):
        group = [index_attribute] + feature_cols[i:i + max_columns]
        column_groups.append(group)

    n_splits = len(column_groups)
    logger.info(f"{n_splits} groups")

    temp_dir = tempfile.mkdtemp(prefix=f"holo_colsplit_{task_name}_")
    split_results = []

    try:
        for i, cols in enumerate(column_groups):
            logger.info(f"  column group {i}/{n_splits}: {len(cols)-1} columns ({cols[1]}...{cols[-1]})")

            # Save the group
            split_dirty = os.path.join(temp_dir, f"split{i}_dirty.csv")
            split_clean = os.path.join(temp_dir, f"split{i}_clean.csv")
            split_output = os.path.join(temp_dir, f"split{i}_cleaned.csv")

            dirty_df[cols].to_csv(split_dirty, index=False)
            pd.read_csv(clean_path)[cols].to_csv(split_clean, index=False)

            # Constraints of the group
            sub_rules = filter_rules_for_columns(dc_rules, cols)

            split_name = f"{task_name}_cs{i}"
            result = run_holoclean_on_subset(
                split_dirty, split_clean, split_output, split_name,
                index_attribute, sub_rules, args, logger
            )

            if result is not None:
                split_results.append((i, result))
                logger.info(f"  column group {i} done")
            else:
                # Dirty data on failure
                split_results.append((i, dirty_df[cols].copy()))
                logger.warning(f"  column group {i} failed, dirty data kept")

            gc.collect()

        # Merge the column groups by joining on the index column
        logger.info("merging the column groups...")
        merged = split_results[0][1].set_index(index_attribute)
        for idx, split_df in split_results[1:]:
            split_df = split_df.set_index(index_attribute)
            for col in split_df.columns:
                if col != index_attribute:
                    merged[col] = split_df[col]
        merged = merged.reset_index()
        # Restore the column order
        original_cols = [c for c in dirty_df.columns if c in merged.columns]
        merged = merged[original_cols]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # _nan_ becomes empty
    merged = normalize_null_values(merged)
    merged.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"merged result saved: {output_file}")

    return merged, {
        'ground_truth_cost': 0,
        'split_mode': 'column',
        'n_splits': n_splits
    }


def run_with_row_split(
    dirty_path: str, clean_path: str, output_file: str,
    task_name: str, index_attribute: str,
    dc_rules: List[str], args, logger: logging.Logger,
    max_rows: int
) -> Tuple[pd.DataFrame, Dict]:
    """Clean a long table in row chunks."""
    dirty_df = pd.read_csv(dirty_path)
    n_rows = len(dirty_df)
    logger.info(f"row chunks: {n_rows} rows, at most {max_rows} per chunk")

    clean_df = pd.read_csv(clean_path)

    # Split the rows
    row_groups = []
    for start in range(0, n_rows, max_rows):
        end = min(start + max_rows, n_rows)
        row_groups.append((start, end))

    n_splits = len(row_groups)
    logger.info(f"{n_splits} groups")

    temp_dir = tempfile.mkdtemp(prefix=f"holo_rowsplit_{task_name}_")
    split_results = []

    try:
        for i, (start, end) in enumerate(row_groups):
            logger.info(f"  row chunk {i}/{n_splits}: rows {start} to {end} ({end-start} rows)")

            # Save the group
            split_dirty = os.path.join(temp_dir, f"split{i}_dirty.csv")
            split_clean = os.path.join(temp_dir, f"split{i}_clean.csv")
            split_output = os.path.join(temp_dir, f"split{i}_cleaned.csv")

            dirty_df.iloc[start:end].to_csv(split_dirty, index=False)
            clean_df.iloc[start:end].to_csv(split_clean, index=False)

            split_name = f"{task_name}_rs{i}"
            result = run_holoclean_on_subset(
                split_dirty, split_clean, split_output, split_name,
                index_attribute, dc_rules, args, logger
            )

            if result is not None:
                split_results.append(result)
                logger.info(f"  row chunk {i} done")
            else:
                split_results.append(dirty_df.iloc[start:end].copy())
                logger.warning(f"  row chunk {i} failed, dirty data kept")

            gc.collect()

        # Merge the row chunks by concatenation
        logger.info("merging the row chunks...")
        merged = pd.concat(split_results, ignore_index=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # _nan_ becomes empty
    merged = normalize_null_values(merged)
    merged.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"merged result saved: {output_file}")

    return merged, {
        'ground_truth_cost': 0,
        'split_mode': 'row',
        'n_splits': n_splits
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run HoloClean data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_holoclean_base.py --dirty_path data/hospitals/dirty_index.csv --rule_path data/hospitals/rules_baselines.txt

Notes:
  - needs a PostgreSQL database
  - the constraint file holds denial constraints
  - fully automatic
        """
    )

    # Data paths
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='path of the dirty table')
    parser.add_argument('--rule_path', type=str, default=None,
                        help='constraint file (denial constraints)')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='path of the clean table (optional)')

    # Task
    parser.add_argument('--task_name', type=str, default='beers_holoclean',
                        help='task name')
    parser.add_argument('--output_path', type=str, default='../../results/holoclean/',
                        help='output directory')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='index column')

    # Database
    parser.add_argument('--db_user', type=str, default='holocleanuser',
                        help='database user')
    parser.add_argument('--db_name', type=str, default='holo',
                        help='database name')

    # HoloClean
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs (10)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate (0.001)')
    parser.add_argument('--threads', type=int, default=1,
                        help='threads (1)')
    parser.add_argument('--weak_label_thresh', type=float, default=0.99,
                        help='weak-label threshold (0.99)')

    # Splits
    parser.add_argument('--max_columns', type=int, default=MAX_COLUMNS_PER_SPLIT,
                        help=f'column-split threshold ({MAX_COLUMNS_PER_SPLIT})')
    parser.add_argument('--max_rows', type=int, default=MAX_ROWS_PER_SPLIT,
                        help=f'row-split threshold ({MAX_ROWS_PER_SPLIT})')

    # Label
    parser.add_argument('--label_column', type=str, default='style',
                        help='label column')

    parser.add_argument('--verbose', action='store_true',
                        help='print progress information')
    parser.add_argument('--use_split', action='store_true',
                        help='60/20/20 split of DemandClean (seed 42)')

    args = parser.parse_args()

    # Output directory
    result_path = os.path.join(args.output_path, args.task_name)
    os.makedirs(result_path, exist_ok=True)

    # Logging
    logger = setup_logging(result_path, args.task_name)

    # Check the dependencies
    if not HAS_HOLOCLEAN:
        logger.error(f"HoloClean import failed: {IMPORT_ERROR}")
        logger.error("make sure HoloClean is installed and PostgreSQL is configured")
        return

    # Start time
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"started: {start_datetime}")
    logger.info("=" * 60)
    logger.info("HoloClean")
    logger.info("=" * 60)
    logger.info(f"dirty table: {args.dirty_path}")
    logger.info(f"constraint file: {args.rule_path or 'not given'}")
    logger.info(f"task: {args.task_name}")
    logger.info(f"database: {args.db_name}")
    logger.info(f"column-split threshold: {args.max_columns}")
    logger.info(f"row-split threshold: {args.max_rows}")
    logger.info("-" * 60)

    # Parse the constraints first (the split modes need them too)
    dc_rules = []
    if args.rule_path and os.path.exists(args.rule_path):
        dc_rules = parse_holoclean_dc_rules(args.rule_path)
        logger.info(f"{len(dc_rules)} denial constraints loaded")

    # Table size decides about splitting
    dirty_df = pd.read_csv(args.dirty_path)
    n_rows, n_cols = dirty_df.shape
    feature_cols = n_cols - 1  # Without the index column
    logger.info(f"table: {n_rows} rows x {n_cols} columns ({n_rows * feature_cols} cells)")

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': 0}

    need_col_split = feature_cols > args.max_columns
    need_row_split = n_rows > args.max_rows

    if need_col_split or need_row_split:
        # Split modes
        try:
            if need_col_split:
                logger.info(f"{feature_cols} columns, above the threshold ({args.max_columns}), column groups enabled")
                merged_df, clean_info = run_with_column_split(
                    args.dirty_path, args.clean_path, output_file,
                    args.task_name, args.index_attribute,
                    dc_rules, args, logger, args.max_columns
                )
            else:
                logger.info(f"{n_rows} rows, above the threshold ({args.max_rows}), row chunks enabled")
                merged_df, clean_info = run_with_row_split(
                    args.dirty_path, args.clean_path, output_file,
                    args.task_name, args.index_attribute,
                    dc_rules, args, logger, args.max_rows
                )
            success = True
        except Exception as e:
            logger.error(f"error in the split mode: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Regular mode
        logger.info(f"table within the thresholds")

        try:
            # Step 1: missing values become 'empty'
            logger.info("Step 1: normalizing the missing values...")
            ori_empty_dirty_path = os.path.join(result_path, "dirty_ori_empty.csv")
            inject_missing_values(
                csv_file=args.dirty_path,
                output_file=ori_empty_dirty_path,
                missing_value_in_ori_data='empty',
                missing_value_representation='',
                attributes_error_ratio=None
            )

            if args.clean_path and os.path.exists(args.clean_path):
                ori_empty_clean_path = os.path.join(result_path, "clean_ori_empty.csv")
                inject_missing_values(
                    csv_file=args.clean_path,
                    output_file=ori_empty_clean_path,
                    missing_value_in_ori_data='empty',
                    missing_value_representation='',
                    attributes_error_ratio=None
                )
            else:
                ori_empty_clean_path = None

            # Step 2: read the dirty table and drop the index column
            logger.info("Step 2: dropping the index column...")
            dirty_data = pd.read_csv(ori_empty_dirty_path)
            if args.index_attribute in dirty_data.columns:
                dirty_data_no_index = dirty_data.drop(columns=[args.index_attribute])
            else:
                dirty_data_no_index = dirty_data

            dirty_holoclean_path = os.path.join(result_path, "dirty_holoclean.csv")
            dirty_data_no_index.to_csv(dirty_holoclean_path, index=False, encoding='utf-8')

            # Step 3: clean data in the HoloClean format (tid, attribute, correct_val)
            clean_holoclean_path = None
            if ori_empty_clean_path:
                logger.info("Step 3: converting the clean data to the HoloClean format...")
                clean_holoclean_path = os.path.join(result_path, "clean_holoclean.csv")
                transform_csv_file(ori_empty_clean_path, clean_holoclean_path)

            clean_data = pd.read_csv(args.clean_path) if args.clean_path else None
            clean_data_attributes = clean_data.columns.tolist() if clean_data is not None else []

            # Step 4: initialize HoloClean
            logger.info("Step 4: initializing HoloClean...")
            hc = holoclean.HoloClean(
                db_user=args.db_user,
                db_name=args.db_name,
                domain_thresh_1=0,
                domain_thresh_2=0,
                weak_label_thresh=args.weak_label_thresh,
                max_domain=10000,
                cor_strength=0.6,
                nb_cor_strength=0.8,
                epochs=args.epochs,
                weight_decay=0.01,
                learning_rate=args.learning_rate,
                threads=args.threads,
                batch_size=1,
                verbose=args.verbose,
                timeout=3 * 60000,
                feature_norm=False,
                weight_norm=False,
                print_fw=True
            ).session

            # Step 5: load the data
            logger.info("Step 5: loading the data into HoloClean...")
            safe_task_name = args.task_name.replace('-', '_').replace('.', '_')
            if safe_task_name[0].isdigit():
                safe_task_name = 't_' + safe_task_name
            hc.load_data(safe_task_name, dirty_holoclean_path)

            # Step 6: load the constraints
            dc_file_path = None
            if args.rule_path and os.path.exists(args.rule_path):
                logger.info("Step 6: loading the constraints...")
                local_dc_rules = parse_holoclean_dc_rules(args.rule_path)
                if local_dc_rules:
                    dc_file_path = create_temp_dc_file(local_dc_rules, result_path)
                    logger.info(f"{len(local_dc_rules)} denial constraints parsed")
                    hc.load_dcs(dc_file_path)
                    hc.ds.set_constraints(hc.get_dcs())
                else:
                    logger.warning("warning: no HOLOCLEAN_DC rules in the rule file")

            # Step 7: detect errors
            logger.info("Step 7: detecting errors...")
            detectors = [NullDetector()]
            if dc_file_path:
                detectors.append(ViolationDetector())

            no_errors_detected = False
            try:
                hc.detect_errors(detectors)
            except (ValueError, Exception) as detect_err:
                if "Wrong number of items" in str(detect_err) or "empty" in str(detect_err).lower():
                    logger.warning(f"no errors detected (the data may be clean): {detect_err}")
                    no_errors_detected = True
                else:
                    raise

            if no_errors_detected:
                logger.info("nothing to repair, writing the data unchanged...")
                output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
                dirty_data.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"saved: {output_file}")
                success = True
                clean_info = {
                    'ground_truth_cost': 0,
                    'detectors': len(detectors),
                    'featurizers': 0,
                    'no_errors_detected': True
                }
            else:
                # Steps 8 to 10: domain, featurization, repair
                numerical_error = False
                try:
                    logger.info("Step 8: setting up the domain...")
                    hc.setup_domain()

                    logger.info("Step 9: featurizing...")
                    featurizers = [
                        InitAttrFeaturizer(),
                        OccurAttrFeaturizer(),
                        FreqFeaturizer()
                    ]
                    if dc_file_path:
                        featurizers.append(ConstraintFeaturizer())

                    logger.info("Step 10: repairing...")
                    hc.repair_errors(featurizers)

                except ZeroDivisionError as zde:
                    logger.warning(f"numerical error (limit of wide data): {zde}")
                    logger.info("HoloClean cannot process this table, writing the data unchanged...")
                    numerical_error = True

                # Step 11: save the repair
                logger.info("Step 11: saving the repaired data...")
                output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")

                if numerical_error:
                    dirty_data.to_csv(output_file, index=False, encoding='utf-8')
                    logger.info(f"saved (data unchanged): {output_file}")
                    success = True
                    clean_info = {
                        'ground_truth_cost': 0,
                        'detectors': len(detectors),
                        'featurizers': 0,
                        'numerical_error': True
                    }
                elif clean_holoclean_path and clean_data is not None:
                    try:
                        hc.evaluate(
                            fpath=clean_holoclean_path,
                            tid_col='tid',
                            attr_col='attribute',
                            val_col='correct_val',
                            output_csv_path=output_file,
                            attrubte_list=clean_data_attributes,
                            clean_data=clean_data,
                            index_attribute=args.index_attribute
                        )
                        logger.info(f"repaired data saved: {output_file}")
                    except (AttributeError, Exception) as eval_err:
                        logger.warning(f"evaluate failed: {eval_err}")
                        logger.info("exporting the repaired data another way...")
                        try:
                            query = "SELECT * FROM {} ORDER BY _tid_".format(
                                hc.ds.raw_data.name + '_repaired')
                            data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
                            repaired_df = pd.DataFrame(data, columns=columns)
                            if '_tid_' in repaired_df.columns:
                                repaired_df.drop(columns=['_tid_'], inplace=True)
                            if clean_data is not None and args.index_attribute in clean_data.columns:
                                if len(clean_data) == len(repaired_df):
                                    repaired_df[args.index_attribute] = clean_data[args.index_attribute].values
                            if clean_data_attributes:
                                attr_list = [c for c in clean_data_attributes if c != args.index_attribute]
                                missing_cols = [c for c in attr_list if c not in repaired_df.columns]
                                for col in missing_cols:
                                    repaired_df[col] = ''
                                ordered_cols = [args.index_attribute] + attr_list
                                repaired_df = repaired_df[ordered_cols]
                            # _nan_ becomes empty
                            repaired_df = normalize_null_values(repaired_df)
                            repaired_df.to_csv(output_file, index=False, encoding='utf-8')
                            logger.info(f"repaired data saved (fallback export): {output_file}")
                        except Exception as fallback_err:
                            logger.error(f"fallback export failed too: {fallback_err}")
                            import traceback
                            traceback.print_exc()
                            output_file = None
                else:
                    try:
                        query = "SELECT * FROM {} ORDER BY _tid_".format(
                            hc.ds.raw_data.name + '_repaired')
                        data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
                        repaired_df = pd.DataFrame(data, columns=columns)
                        if '_tid_' in repaired_df.columns:
                            repaired_df.drop(columns=['_tid_'], inplace=True)
                        if args.index_attribute in dirty_data.columns:
                            repaired_df[args.index_attribute] = dirty_data[args.index_attribute].values
                        # _nan_ becomes empty
                        repaired_df = normalize_null_values(repaired_df)
                        repaired_df.to_csv(output_file, index=False)
                        logger.info(f"repaired data saved: {output_file}")
                    except Exception as save_err:
                        logger.warning(f"warning: could not save the repaired data: {save_err}")
                        output_file = None

                if not numerical_error:
                    success = True
                    clean_info = {
                        'ground_truth_cost': 0,
                        'detectors': len(detectors),
                        'featurizers': len(featurizers)
                    }

        except Exception as e:
            logger.error(f"error: {e}")
            import traceback
            traceback.print_exc()
            success = False
            clean_info = {'ground_truth_cost': 0}
            output_file = None

    # Time
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"time: {elapsed_time:.2f} s")
    logger.info(f"status: {'success' if success else 'failure'}")
    logger.info(f"ground truth used: {clean_info.get('ground_truth_cost', 0)} (fully automatic)")



    logger.info(f"\nresults saved to: {result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
