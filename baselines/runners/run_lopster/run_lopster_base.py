"""
Runner of Lopster (VLDB 2024), data cleaning with latent operators.

- a VAE learns a representation of the data used to detect and repair errors
- the VAE is trained on clean.csv
- the official TensorFlow implementation is called
- the input is preprocessed (numbers with units, percent signs, missing-value tokens)

Usage:
    python run_lopster_base.py --dataset adult --data_path <data directory>

Example:
    python run_lopster_base.py \
        --dataset adult \
        --data_path data \
        --task_name adult_lopster \
        --output_path results/baselines/lopster/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd
import numpy as np
import shutil
import re

# Put the baselines root on the path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Preprocessing that turns the input into the numeric format of the Lopster VAE
# ============================================================================

def clean_numeric_value(val):
    """
    Clean a numeric field with units or percent signs.

    Formats:
    - "12.0 oz", "12.0 oz.", "12 ounce" -> 12.0
    - "0.05", "0.09%", "5%" -> 0.05, 0.09, 5.0
    - "empty", "null", "none", "na", "" -> NaN
    - plain numbers stay unchanged
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().strip()

    # Missing-value tokens
    if val_str in ['empty', 'null', 'none', 'na', 'n/a', '', ' ', 'nan', '-']:
        return np.nan

    # Remove the percent sign
    val_str = val_str.replace('%', '')

    # Numeric part (decimals and negative numbers)
    match = re.search(r'(-?\d+\.?\d*)', val_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan

    return np.nan


def prepare_data_for_lopster(input_path, output_path, logger=None):
    """
    Prepare a table for Lopster in a purely numeric format.

    1. remove the index column (not used by Lopster)
    2. missing-value tokens become NaN
    3. convertible columns become numeric
    4. numbers with units ("12.0 oz" -> 12.0)

    Args:
        input_path: input file
        output_path: output file
        logger: logger (optional)

    Returns:
        processed DataFrame
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    df = pd.read_csv(input_path)
    original_shape = df.shape

    # 1. Remove the index column
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        log(f"  - index column removed")

    # 2. Normalize the missing-value tokens
    missing_markers = ['empty', 'Empty', 'EMPTY', 'null', 'NULL', 'Null',
                       'none', 'None', 'NONE', 'na', 'NA', 'N/A', 'n/a',
                       '', ' ', 'nan', 'NaN', 'NAN', '-']
    df = df.replace(missing_markers, np.nan)

    # 3. Convert the columns to numbers where possible
    converted_cols = []
    for col in df.columns:
        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Non-numeric content (NaN excluded)
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        # Direct conversion
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            converted_cols.append(col)
            continue
        except:
            pass

        # Strict test for a numeric column with units
        # Formats: "12.0 oz", "0.05%", "12 ounces"
        # The value is mainly a number, optionally followed by a unit
        sample_values = non_null.head(20).astype(str).tolist()

        # Strict match: number (optional decimals), optional space, optional unit or percent sign
        # Units are short (1 to 7 characters), so "10 Barrel Brewing Company" does not match
        unit_pattern = r'^\s*-?\d+\.?\d*\s*(%|oz\.?|ounces?|ml|l|kg|g|lb\.?|lbs?|in|ft|m|cm|mm)?\s*$'

        # Do most sampled values have the format of a number with a unit
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0

        # Convert only when at least 80% of the values match
        if match_ratio >= 0.8:
            df[col] = df[col].apply(clean_numeric_value)
            converted_cols.append(f"{col}(with units)")

    if converted_cols:
        log(f"  - numeric columns: {converted_cols}")

    # 4. Save
    df.to_csv(output_path, index=False)
    log(f"  - saved to: {output_path}")
    log(f"  - shape: {original_shape} -> {df.shape}")

    return df


def prepare_lopster_dataset(data_path, dataset, result_path, logger=None):
    """
    Prepare a dataset for Lopster (the data directory is not modified).

    Files written to result_path:
    - lopster_data/clean.csv: training input of Lopster (no index, encoded)
    - lopster_data/dirty01.csv: cleaning input of Lopster (no index, encoded)
    - clean_encoded.csv: encoded clean table with index
    - dirty_encoded.csv: encoded dirty table with index

    Args:
        data_path: root directory of the datasets (read only)
        dataset: dataset name
        result_path: output directory
        logger: logger

    Returns:
        dict with
            - dirty_src: path of the dirty table
            - clean_src: path of the clean table
            - lopster_data_dir: data directory of Lopster
            - dirty_encoded_path: encoded dirty table with index
            - clean_encoded_path: encoded clean table with index
            - index_column: values of the index column (if any)
            - n_rows: number of rows
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    dataset_dir = os.path.join(data_path, dataset)

    # Source files (read only)
    clean_src = None
    dirty_src = None

    # Prefer the _index.csv files
    if os.path.exists(os.path.join(dataset_dir, 'clean_index.csv')):
        clean_src = os.path.join(dataset_dir, 'clean_index.csv')
    elif os.path.exists(os.path.join(dataset_dir, 'clean.csv')):
        clean_src = os.path.join(dataset_dir, 'clean.csv')

    if os.path.exists(os.path.join(dataset_dir, 'dirty_index.csv')):
        dirty_src = os.path.join(dataset_dir, 'dirty_index.csv')
    elif os.path.exists(os.path.join(dataset_dir, 'dirty.csv')):
        dirty_src = os.path.join(dataset_dir, 'dirty.csv')

    if not clean_src:
        raise FileNotFoundError(f"clean table not found: {dataset_dir}/clean_index.csv or clean.csv")
    if not dirty_src:
        raise FileNotFoundError(f"dirty table not found: {dataset_dir}/dirty_index.csv or dirty.csv")

    # Data directory of Lopster (under the results)
    lopster_data_dir = os.path.join(result_path, 'lopster_data', dataset)
    os.makedirs(lopster_data_dir, exist_ok=True)

    # Read the original data and keep its metadata
    original_dirty = pd.read_csv(dirty_src)
    original_clean = pd.read_csv(clean_src)

    index_column = original_dirty['index'].values if 'index' in original_dirty.columns else None

    log(f"\npreparing the Lopster input (written to the results directory, the data is not modified)...")
    log(f"  source files: {os.path.basename(clean_src)}, {os.path.basename(dirty_src)}")
    log(f"  Lopster data directory: {lopster_data_dir}")

    # ========== 1. Lopster input (no index, encoded) ==========
    lopster_clean = os.path.join(lopster_data_dir, 'clean.csv')
    lopster_dirty = os.path.join(lopster_data_dir, 'dirty01.csv')

    log(f"\nLopster training data (no index):")
    prepare_data_for_lopster(clean_src, lopster_clean, logger)
    log(f"\nLopster cleaning data (no index):")
    prepare_data_for_lopster(dirty_src, lopster_dirty, logger)

    # ========== 2. encoded tables with index ==========
    clean_encoded_path = os.path.join(result_path, 'clean_encoded.csv')
    dirty_encoded_path = os.path.join(result_path, 'dirty_encoded.csv')

    log(f"\nencoded tables with index:")

    # Strict unit pattern (as in prepare_data_for_lopster)
    unit_pattern = r'^\s*-?\d+\.?\d*\s*(%|oz\.?|ounces?|ml|l|kg|g|lb\.?|lbs?|in|ft|m|cm|mm)?\s*$'

    # Clean table
    clean_df = original_clean.copy()
    missing_markers = ['empty', 'Empty', 'EMPTY', 'null', 'NULL', 'Null',
                       'none', 'None', 'NONE', 'na', 'NA', 'N/A', 'n/a',
                       '', ' ', 'nan', 'NaN', 'NAN', '-']
    clean_df = clean_df.replace(missing_markers, np.nan)

    # Convert the columns to numbers where possible
    for col in clean_df.columns:
        if col == 'index':
            continue
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            continue
        non_null = clean_df[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='raise')
            continue
        except:
            pass
        # Strict unit pattern
        sample_values = non_null.head(20).astype(str).tolist()
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0
        if match_ratio >= 0.8:
            clean_df[col] = clean_df[col].apply(clean_numeric_value)

    clean_df.to_csv(clean_encoded_path, index=False)
    log(f"  - clean_encoded.csv: {clean_encoded_path}")

    # Dirty table
    dirty_df = original_dirty.copy()
    dirty_df = dirty_df.replace(missing_markers, np.nan)

    for col in dirty_df.columns:
        if col == 'index':
            continue
        if pd.api.types.is_numeric_dtype(dirty_df[col]):
            continue
        non_null = dirty_df[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            dirty_df[col] = pd.to_numeric(dirty_df[col], errors='raise')
            continue
        except:
            pass
        # Strict unit pattern
        sample_values = non_null.head(20).astype(str).tolist()
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0
        if match_ratio >= 0.8:
            dirty_df[col] = dirty_df[col].apply(clean_numeric_value)

    dirty_df.to_csv(dirty_encoded_path, index=False)
    log(f"  - dirty_encoded.csv: {dirty_encoded_path}")

    log(f"\nLopster input ready.")

    return {
        'dirty_src': dirty_src,
        'clean_src': clean_src,
        'lopster_data_dir': lopster_data_dir,
        'dirty_encoded_path': dirty_encoded_path,
        'clean_encoded_path': clean_encoded_path,
        'index_column': index_column,
        'n_rows': len(original_dirty),
        'original_columns': list(original_dirty.columns)
    }


def postprocess_lopster_output(lopster_output_path, original_info, output_path, logger=None):
    """
    Restore the format of the original data in the Lopster output.

    1. restore the index column
    2. restore the ID columns (beer_name and others, which Lopster must not change)
    3. keep the row count of the original data
    4. keep the column order of the original data

    Args:
        lopster_output_path: output file of Lopster
        original_info: information returned by prepare_lopster_dataset
        output_path: final output path
        logger: logger

    Returns:
        processed DataFrame
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"\npostprocessing the Lopster output...")

    # Read the Lopster output
    lopster_df = pd.read_csv(lopster_output_path)
    log(f"  Lopster output rows: {len(lopster_df)}")
    log(f"  original rows: {original_info['n_rows']}")

    # Read the dirty table (to restore the ID columns)
    original_dirty = pd.read_csv(original_info['dirty_src'])

    # Same row count
    if len(lopster_df) != original_info['n_rows']:
        log(f"  warning: row counts differ, Lopster={len(lopster_df)}, original={original_info['n_rows']}")
        # Missing rows are filled from the dirty table
        if len(lopster_df) < original_info['n_rows']:
            # Drop the index column for the comparison
            if 'index' in original_dirty.columns:
                original_dirty_no_idx = original_dirty.drop(columns=['index'])
            else:
                original_dirty_no_idx = original_dirty
            # Fill the missing rows with the last rows of the dirty table
            missing_rows = original_info['n_rows'] - len(lopster_df)
            log(f"  {missing_rows} missing rows filled")
            extra_rows = original_dirty_no_idx.iloc[len(lopster_df):].copy()
            # Align the columns
            for col in lopster_df.columns:
                if col not in extra_rows.columns:
                    extra_rows[col] = np.nan
            extra_rows = extra_rows[lopster_df.columns]
            lopster_df = pd.concat([lopster_df, extra_rows], ignore_index=True)

    # ========== restore the ID columns (Lopster must not change them) ==========
    # id_cols of the dataset configuration
    try:
        import json
        config_path = os.path.join(PROJECT_ROOT, 'Methods', 'Lopster', 'dataset_configuration.json')
        with open(config_path, 'r') as f:
            dataset_config = json.load(f)

        dataset_name = os.path.basename(os.path.dirname(original_info['dirty_src']))
        if dataset_name in dataset_config:
            id_cols = dataset_config[dataset_name].get('id_cols', [])
            date_cols = dataset_config[dataset_name].get('date_cols', [])
            restore_cols = id_cols + date_cols

            if restore_cols:
                log(f"  ID and date columns restored: {restore_cols}")
                for col in restore_cols:
                    if col in original_dirty.columns and col in lopster_df.columns:
                        # Overwrite the Lopster output with the original values (ID columns stay unchanged)
                        lopster_df[col] = original_dirty[col].values[:len(lopster_df)]
    except Exception as e:
        log(f"  could not read the dataset configuration: {e}")

    # Restore the index column
    if original_info['index_column'] is not None:
        if 'index' in lopster_df.columns:
            lopster_df['index'] = original_info['index_column'][:len(lopster_df)]
        else:
            lopster_df.insert(0, 'index', original_info['index_column'][:len(lopster_df)])
        log(f"  index column restored")

    # Column order of the original data
    original_columns = original_info['original_columns']
    final_columns = []
    for col in original_columns:
        if col in lopster_df.columns:
            final_columns.append(col)
    # Columns added by Lopster
    for col in lopster_df.columns:
        if col not in final_columns:
            final_columns.append(col)

    lopster_df = lopster_df[final_columns]
    log(f"  final column order: {final_columns}")

    # Save
    lopster_df.to_csv(output_path, index=False)
    log(f"  saved to: {output_path}")
    log(f"  final rows: {len(lopster_df)}")

    return lopster_df


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


def main():
    parser = argparse.ArgumentParser(
        description='Run Lopster data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_lopster_base.py --dataset adult --data_path data

Notes:
  - Lopster trains its VAE on clean.csv
  - needs TensorFlow
        """
    )

    # Data paths
    parser.add_argument('--dataset', type=str, default='beers',
                        help='dataset name (for example adult, beers)')
    parser.add_argument('--data_path', type=str, default='../../Data',
                        help='root directory of the datasets')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='path of the clean table (optional, derived from the dataset by default)')

    # Task
    parser.add_argument('--task_name', type=str, default='beers_lopster',
                        help='task name (the dataset name by default)')
    parser.add_argument('--output_path', type=str, default='../../results/lospter',
                        help='output directory')

    # Lopster
    parser.add_argument('--latent_dim', type=int, default=120,
                        help='dimension of the latent space (120)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epochs (100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate (0.001)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (256)')
    parser.add_argument('--K', type=int, default=12,
                        help='number of translation steps K (12)')
    parser.add_argument('--clean_ratio', type=float, default=1.0,
                        help='share of the clean rows used (0.01 to 1.0, default 1.0 = 100%%)')

    # Columns
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='index column')
    parser.add_argument('--label_column', type=str, default='style',
                        help='label column')

    parser.add_argument('--verbose', action='store_true',
                        help='print progress information')
    parser.add_argument('--use_split', action='store_true',
                        help='60/20/20 split of DemandClean (seed 42)')

    args = parser.parse_args()

    # Default task name
    if args.task_name is None:
        args.task_name = f"{args.dataset}_lopster"

    # Derive clean_path
    if args.clean_path is None:
        args.clean_path = os.path.join(args.data_path, args.dataset, 'clean.csv')

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
    logger.info("Lopster")
    logger.info("=" * 60)
    logger.info(f"dataset: {args.dataset}")
    logger.info(f"data path: {args.data_path}")
    logger.info(f"task: {args.task_name}")
    logger.info(f"latent dimension: {args.latent_dim}")
    logger.info(f"epochs: {args.epochs}")
    logger.info("-" * 60)

    # Absolute paths
    original_cwd = os.getcwd()
    abs_data_path = os.path.join(original_cwd, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    abs_output_path = os.path.join(original_cwd, args.output_path) if not os.path.isabs(args.output_path) else args.output_path
    abs_result_path = os.path.join(abs_output_path, args.task_name)

    # Preprocess the input into the Lopster format (written to the results directory)
    preprocess_info = None
    try:
        preprocess_info = prepare_lopster_dataset(abs_data_path, args.dataset, abs_result_path, logger)
    except Exception as e:
        logger.warning(f"preprocessing warning: {e}")
        logger.warning("falling back to the original data files...")
        import traceback
        traceback.print_exc()

    # Run inside the Lopster directory (datasets.py reads its configuration by a relative path)
    lopster_dir = os.path.join(PROJECT_ROOT, 'Methods', 'Lopster')

    try:
        os.chdir(lopster_dir)
        # Put the Lopster directory on the path
        if lopster_dir not in sys.path:
            sys.path.insert(0, lopster_dir)

        # Import and run Lopster
        from lopster_wrapper import LopsterWrapper

        # Cleaner
        cleaner = LopsterWrapper(
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            K=args.K,
            clean_ratio=args.clean_ratio,
            verbose=args.verbose
        )

        # Clean the preprocessed data under the results directory
        output_file = os.path.join(abs_result_path, f"{args.task_name}_cleaned.csv")

        # Lopster data path
        if preprocess_info and 'lopster_data_dir' in preprocess_info:
            # Preprocessed data directory (under the results)
            lopster_data_path = os.path.dirname(preprocess_info['lopster_data_dir'])  # lopster_data/
            logger.info(f"\npreprocessed data: {preprocess_info['lopster_data_dir']}")
        else:
            # Fall back to the original data directory
            lopster_data_path = abs_data_path
            logger.info(f"\noriginal data: {lopster_data_path}")

        repaired_df, clean_info = cleaner.clean(
            dataset=args.dataset,
            path=lopster_data_path,
            output_path=output_file
        )
        success = True

    except Exception as e:
        logger.error(f"error: {e}")
        import traceback
        traceback.print_exc()
        success = False
        clean_info = {'ground_truth_cost': 0}
        repaired_df = None
    finally:
        os.chdir(original_cwd)

    # Time
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"time: {elapsed_time:.2f} s")
    logger.info(f"status: {'success' if success else 'failure'}")
    logger.info(f"ground truth used: {clean_info.get('ground_truth_cost', 0)} clean rows (VAE training)")

    # Postprocessing: add the index column
    if success and preprocess_info and os.path.exists(output_file):
        try:
            postprocess_lopster_output(output_file, preprocess_info, output_file, logger)
        except Exception as e:
            logger.warning(f"postprocessing warning: {e}")
            import traceback
            traceback.print_exc()


    logger.info(f"\nresults saved to: {abs_result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
