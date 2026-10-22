"""
data_splitter.py - DemandClean-aligned data splitting module

Provides the same seed=42, 60/20/20 train/val/test split as DemandClean
(scripts/run_demandclean.py).

All run_xxx_base.py scripts can import from this module to keep the split consistent.

Usage:
    from tools.data_splitter import get_demandclean_split

    split = get_demandclean_split(dirty_path, clean_path)
    dirty_train = split['dirty_train']
    clean_val = split['clean_val']
    clean_test = split['clean_test']
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Optional


SEED = 42
TRAIN_RATIO = 0.6  # 60% train, 20% val, 20% test


def compute_split_indices(n_total: int, seed: int = SEED):
    """
    Compute the same 60/20/20 split indices as DemandClean.

    Matches scripts/run_demandclean.py:
        all_idx = np.arange(n_total)
        train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    Args:
        n_total: total number of rows
        seed: random seed (default 42)

    Returns:
        (train_idx, val_idx, test_idx) numpy arrays
    """
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


def get_demandclean_split(dirty_path: str, clean_path: str,
                           seed: int = SEED) -> Dict:
    """
    Load the data and return a 60/20/20 split identical to DemandClean's.

    Args:
        dirty_path: path to the dirty CSV
        clean_path: path to the clean CSV
        seed: random seed

    Returns:
        dict containing:
        - train_idx, val_idx, test_idx: split indices
        - dirty_train: dirty training set (60%)
        - clean_train: clean training set (60%)
        - clean_val: clean validation set (20%)
        - clean_test: clean test set (20%)
        - dirty_full: full dirty data
        - clean_full: full clean data
        - n_total: total number of rows
    """
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    assert len(dirty_df) == len(clean_df), \
        f"Dirty ({len(dirty_df)} rows) and clean ({len(clean_df)} rows) row counts differ"

    n_total = len(dirty_df)
    train_idx, val_idx, test_idx = compute_split_indices(n_total, seed)

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'dirty_train': dirty_df.iloc[train_idx].reset_index(drop=True),
        'dirty_val': dirty_df.iloc[val_idx].reset_index(drop=True),
        'dirty_test': dirty_df.iloc[test_idx].reset_index(drop=True),
        'clean_train': clean_df.iloc[train_idx].reset_index(drop=True),
        'clean_val': clean_df.iloc[val_idx].reset_index(drop=True),
        'clean_test': clean_df.iloc[test_idx].reset_index(drop=True),
        'dirty_full': dirty_df,
        'clean_full': clean_df,
        'n_total': n_total,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
    }


def save_split_csvs(split: Dict, output_dir: str, prefix: str = ''):
    """Save the split subsets as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for key in ['dirty_train', 'clean_train', 'clean_val', 'clean_test']:
        filename = f"{prefix}{key}.csv" if prefix else f"{key}.csv"
        split[key].to_csv(os.path.join(output_dir, filename), index=False)


def verify_split_consistency(dirty_path: str, clean_path: str,
                              demandclean_dirty_train_path: Optional[str] = None):
    """
    Verify that the split matches DemandClean's.

    If DemandClean has already produced dirty_train_60pct.csv, pass it in to diff.
    """
    split = get_demandclean_split(dirty_path, clean_path)
    print(f"Total rows: {split['n_total']}")
    print(f"Train: {split['n_train']} ({split['n_train']/split['n_total']*100:.1f}%)")
    print(f"Val:   {split['n_val']} ({split['n_val']/split['n_total']*100:.1f}%)")
    print(f"Test:  {split['n_test']} ({split['n_test']/split['n_total']*100:.1f}%)")

    if demandclean_dirty_train_path and os.path.exists(demandclean_dirty_train_path):
        dc_train = pd.read_csv(demandclean_dirty_train_path)
        our_train = split['dirty_train']
        if len(dc_train) == len(our_train):
            match = (dc_train.values == our_train.values).all()
            print(f"\nDiff vs. DemandClean dirty_train_60pct.csv: {'exact match' if match else 'mismatch'}")
        else:
            print(f"\nRow count mismatch: DemandClean={len(dc_train)}, ours={len(our_train)}")

    return split


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify the DemandClean data split')
    parser.add_argument('--dirty', required=True, help='Path to the dirty data')
    parser.add_argument('--clean', required=True, help='Path to the clean data')
    parser.add_argument('--dc_train', default=None, help='Path to DemandClean dirty_train_60pct.csv (optional)')
    args = parser.parse_args()

    verify_split_consistency(args.dirty, args.clean, args.dc_train)
