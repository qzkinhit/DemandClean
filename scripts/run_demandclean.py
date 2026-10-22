#!/usr/bin/env python
"""
Run DemandClean on one dataset
==============================

    python scripts/run_demandclean.py --config configs/datasets/beers.yaml
    python scripts/run_demandclean.py --config configs/datasets/beers.yaml --variant ngt
    python scripts/run_demandclean.py --config configs/datasets/beers.yaml --checkpoint results/main_table/beers/demandclean/model

Variants (configs/variants.yaml, the configurations of Figure 5):
    full    DemandClean        hybrid detection, two-stage dueling policy
    ngt     DemandClean-NGT    as full, without ground-truth repair (max_truth_budget 0)
    nd      DemandClean-ND     oracle detection, two-stage dueling policy
    arch    DemandClean-Arch   hybrid detection, two-stage policy with plain DQN heads
    stage   DemandClean-Stage  hybrid detection, single-stage dueling policy

A run writes into its output directory
    run.log               console output of the run
    manifest.json         command, arguments, seed, configuration, package versions, start and end time
    summary.json          action counts, ground-truth cells, detection accuracy, timings
    cleaned.csv           cleaned training split in the original CSV format
    cleaned_encoded.npz   cleaned training split in the encoded space (input of the scorer)
    model/                trained policy (agent_stage1.pt, agent_stage2.pt)
    training_history.json score, reward, epsilon and operation counts of every episode (training runs)
    split/                dirty and clean training split of the 60/20/20 split (input of the detector)
    demandclean_<YYYYmmdd_HHMMSS>.log
                          records of the Python logging module of the package (detector, clean base
                          selection, episodes) with time stamps, usually one file per run
    scores.json           evaluation report written by evaluation/paper_score.py
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from demandclean.api import DemandClean
from demandclean.utils.model_io import ModelIO
from demandclean.evaluation.detection import evaluate_detector_accuracy
from demandclean.utils.edit_distance import find_nearest_known

# Categories (and class labels) absent from the dirty training split are encoded as missing
# by default, which is the encoding of the reported runs. --unseen-categories nearest maps them
# to the nearest known category by edit distance instead.
_find_nearest_known = None

# Dataset configuration, filled from --config
DATASETS = {}


def parse_rules_file(rules_path: str) -> Dict[str, List[str]]:
    """
    Parse the unified rule file.

    Args:
        rules_path: path to rules.txt

    Returns:
        Dict keyed by section name, with lists of rule strings as values.
    """
    if not os.path.exists(rules_path):
        return {}

    sections = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith('#'):
                continue

            # Detect section headers
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    return sections


def get_horizon_fds(rules_path: str) -> List[Tuple[str, str]]:
    """
    Return Horizon functional-dependency rules.

    Returns:
        List of (lhs, rhs) tuples.
    """
    sections = parse_rules_file(rules_path)
    fds = []

    for line in sections.get('HORIZON_FD', []):
        # Accept both "=>" and the Unicode arrow
        if '=>' in line:
            parts = line.split('=>')
        elif '\u21d2' in line:
            parts = line.split('\u21d2')
        else:
            continue

        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            fds.append((lhs, rhs))

    return fds


class TeeLogger:
    """Write stdout/stderr to both terminal and log file (fault-tolerant)."""

    def __init__(self, log_path: str, stream=None):
        self.terminal = stream or sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # line buffered

    @property
    def _file_ok(self):
        """Check whether log_file is still writable."""
        try:
            return self.log_file is not None and not self.log_file.closed
        except Exception:
            return False

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            # terminal is gone too; fall back to original stdout
            if sys.__stdout__ is not None and not sys.__stdout__.closed:
                sys.__stdout__.write(message)
        if self._file_ok:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (ValueError, IOError, OSError):
                pass  # silent failure; degrade to terminal-only output

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        if self._file_ok:
            try:
                self.log_file.flush()
            except (ValueError, IOError, OSError):
                pass

    def close(self):
        if self._file_ok:
            try:
                self.log_file.close()
            except (ValueError, IOError, OSError):
                pass

    def fileno(self):
        return self.terminal.fileno()


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch RNGs (error injection, exploration, shuffling)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


REWARD_MODEL_KWARGS = {
    'random_forest': {'n_estimators': 10, 'max_depth': 8},
    'xgboost':       {'n_estimators': 10, 'max_depth': 3},
    'xgboost_reg':   {'n_estimators': 10, 'max_depth': 3},
    'ridge':         {},  # linear models already lightweight
    'linear':        {},  # linear models already lightweight
    'svm':           {},  # linear kernel SVM is fine on small datasets
    'kmeans':        {'n_init': 3},  # fewer KMeans initializations
}


def preprocess_data(dataset_name: str, dirty_df: pd.DataFrame = None, clean_df: pd.DataFrame = None):
    """
    Load and preprocess the specified dataset.

    Steps:
      1. Read dirty_with_index.csv and clean_with_index.csv (or use supplied DataFrames)
      2. Replace "empty"/"Empty"/"EMPTY" with NaN
      3. Fit LabelEncoder on dirty data only (not on clean)
      4. Fit StandardScaler on dirty data only
      5. Auto-derive and parse FD rules

    Args:
        dataset_name: dataset name (e.g. 'beers')
        dirty_df: optional dirty-data DataFrame (skip file read if provided)
        clean_df: optional clean-data DataFrame (used for encoding; None skips clean encoding)

    Returns:
        (X_dirty_scaled, y_dirty, X_clean_scaled_or_None, y_clean_or_None,
         column_names, fd_rules, rules_path,
         dirty_csv_path, clean_csv_path, csv_columns,
         scaler, label_encoders, categorical_cols,
         dirty_df, clean_df_or_None)
    """
    ds_cfg = DATASETS[dataset_name]
    label_col = ds_cfg['label_col']

    # --- Paths ---
    data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(data_dir, 'clean_with_index.csv')
    # DC_RULES_PATH (set by --rules) swaps in an alternative rule library
    rules_path = os.environ.get('DC_RULES_PATH') or os.path.join(data_dir, 'rules.txt')

    # --- Load (or use supplied DataFrame) ---
    read_clean_from_file = (dirty_df is None and clean_df is None)  # default: read both from files
    if dirty_df is None:
        dirty_df = pd.read_csv(dirty_path)
    if clean_df is None and read_clean_from_file:
        clean_df = pd.read_csv(clean_path)
    # Ensure dirty_df has clean column names
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    if clean_df is not None:
        clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]

    # --- Replace empty with NaN ---
    dirty_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)
    if clean_df is not None:
        clean_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    # --- Determine feature columns ---
    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # --- Identify categorical vs numeric columns ---
    config_cat_cols = ds_cfg.get('categorical_cols')
    if config_cat_cols is not None:
        categorical_cols = set(config_cat_cols) & set(feature_cols)
    else:
        categorical_cols = set()
        for col in feature_cols:
            vals = dirty_df[col].dropna()
            vals = vals[~vals.astype(str).str.strip().isin(
                ['?', '', 'N/A', 'NA', 'nan', 'NaN', 'null', 'None', '-', 'empty', 'Empty', 'EMPTY']
            )]
            if len(vals) == 0:
                categorical_cols.add(col)
                continue
            try:
                pd.to_numeric(vals, errors='raise')
            except (ValueError, TypeError):
                categorical_cols.add(col)

    # --- Encoding helpers ---
    # LabelEncoder is fit on dirty data only (no dependency on clean)
    label_encoders = {}

    def encode_df(df, feature_cols, label_col, fit_le=False):
        """Encode features and label of a DataFrame, returning (X, y).

        Args:
            df: DataFrame to encode
            feature_cols: list of feature column names
            label_col: label column name
            fit_le: whether to fit LabelEncoder (set True only for dirty data)
        """
        X_df = df[feature_cols].copy()
        y_series = df[label_col].copy()

        for col in feature_cols:
            if col in categorical_cols:
                if col not in label_encoders and fit_le:
                    le = LabelEncoder()
                    # Fit LE on dirty data only
                    all_vals = dirty_df[col].dropna().astype(str).unique()
                    le.fit(all_vals)
                    label_encoders[col] = le

                nan_mask = X_df[col].isna()
                non_nan_values = X_df.loc[~nan_mask, col].astype(str)
                le = label_encoders[col]

                if _find_nearest_known is not None:
                    known_set = set(le.classes_)
                    known_list = list(le.classes_)

                    def _safe_encode(val, _le=le, _ks=known_set, _kl=known_list):
                        if val in _ks:
                            return _le.transform([val])[0]
                        nearest = _find_nearest_known(val, _kl, threshold=0.3)
                        if nearest is not None:
                            return _le.transform([nearest])[0]
                        return np.nan  # unreachable when LE is dirty-fit

                    X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_encode)
                else:
                    X_df.loc[~nan_mask, col] = le.transform(non_nan_values)
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # Encode label
        label_is_categorical = False
        if fit_le:
            combined_labels = dirty_df[label_col].dropna()
        else:
            combined_labels = df[label_col].dropna()
        try:
            pd.to_numeric(combined_labels, errors='raise')
        except (ValueError, TypeError):
            label_is_categorical = True

        if label_is_categorical:
            if label_col not in label_encoders and fit_le:
                le = LabelEncoder()
                all_labels = dirty_df[label_col].dropna().astype(str).unique()
                le.fit(all_labels)
                label_encoders[label_col] = le
            if label_col in label_encoders:
                nan_mask_y = y_series.isna()
                y_encoded = y_series.copy()
                le_label = label_encoders[label_col]
                known_set_y = set(le_label.classes_)

                def _safe_encode_label(val, _le=le_label, _ks=known_set_y):
                    val_str = str(val)
                    if val_str in _ks:
                        return _le.transform([val_str])[0]
                    if _find_nearest_known is not None:
                        nearest = _find_nearest_known(val_str, list(_le.classes_), threshold=0.3)
                        if nearest is not None:
                            return _le.transform([nearest])[0]
                    return np.nan

                y_encoded.loc[~nan_mask_y] = y_series.loc[~nan_mask_y].astype(str).map(_safe_encode_label)
                y_series = pd.to_numeric(y_encoded, errors='coerce')

        X = X_df.values.astype(float)
        y = y_series.values.astype(float)
        return X, y

    # Encode dirty (also fits LE)
    X_dirty, y_dirty = encode_df(dirty_df, feature_cols, label_col, fit_le=True)

    # Encode clean (using already-fit LE)
    X_clean_scaled = None
    y_clean = None
    if clean_df is not None:
        X_clean, y_clean = encode_df(clean_df, feature_cols, label_col, fit_le=False)

    # --- Standardize (fit on dirty only) ---
    scaler = StandardScaler()
    X_dirty_for_fit = X_dirty.copy()
    col_means = np.nanmean(X_dirty_for_fit, axis=0)
    for j in range(X_dirty_for_fit.shape[1]):
        mask = np.isnan(X_dirty_for_fit[:, j])
        X_dirty_for_fit[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    scaler.fit(X_dirty_for_fit)

    # Standardize dirty (preserving NaN positions)
    def scale_with_nan(X, scaler):
        X_out = X.copy()
        col_m = np.nanmean(X_out, axis=0)
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
        X_out = scaler.transform(X_out)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X_out[nan_mask, j] = np.nan
        return X_out

    X_dirty_scaled = scale_with_nan(X_dirty, scaler)

    # Standardize clean (if available)
    if clean_df is not None:
        X_clean_for_scale = X_clean.copy()
        col_means_c = np.nanmean(X_clean_for_scale, axis=0)
        for j in range(X_clean_for_scale.shape[1]):
            mask = np.isnan(X_clean_for_scale[:, j])
            X_clean_for_scale[mask, j] = col_means_c[j] if not np.isnan(col_means_c[j]) else 0.0
        X_clean_scaled = scaler.transform(X_clean_for_scale)

    # --- Parse FD rules ---
    fd_rules = []
    if os.path.exists(rules_path) and get_horizon_fds is not None:
        try:
            fd_rules = get_horizon_fds(rules_path)
        except Exception as e:
            print(f"  [warn] FD rule parse failed: {e}")

    print(f"  Dataset: {dataset_name}")
    print(f"  Dirty:   {X_dirty_scaled.shape}")
    if X_clean_scaled is not None:
        print(f"  Clean:   {X_clean_scaled.shape}")
    print(f"  Features: {len(feature_cols)}, label: {label_col}")
    print(f"  Categorical cols: {categorical_cols if categorical_cols else '(none)'}")
    print(f"  LE fit: dirty only ({len(dirty_df)} rows)")
    print(f"  SS fit: dirty only ({len(dirty_df)} rows)")
    print(f"  FD rules: {len(fd_rules)}")

    csv_columns = list(dirty_df.columns)

    return (
        X_dirty_scaled, y_dirty,
        X_clean_scaled, y_clean,
        feature_cols, fd_rules, rules_path,
        dirty_path, clean_path, csv_columns,
        scaler, label_encoders, categorical_cols,
        dirty_df, clean_df,
    )


def encode_subset(df, feature_cols, label_col, label_encoders, scaler,
                  categorical_cols, dataset_name=None):
    """Encode a val/test subset with already-fit LE/SS; return (X_scaled, y).

    Args:
        df: DataFrame to encode
        feature_cols: feature column names
        label_col: label column name
        label_encoders: already-fit {col_name: LabelEncoder}
        scaler: already-fit StandardScaler
        categorical_cols: set of categorical column names
        dataset_name: optional; used for logs

    Returns:
        (X_scaled, y), standardized feature matrix and label array
    """
    df = df.copy()
    df.columns = [c.strip().strip('\ufeff') for c in df.columns]
    df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    X_df = df[feature_cols].copy()
    y_series = df[label_col].copy()

    for col in feature_cols:
        if col in categorical_cols and col in label_encoders:
            le = label_encoders[col]
            nan_mask = X_df[col].isna()
            non_nan_values = X_df.loc[~nan_mask, col].astype(str)

            if _find_nearest_known is not None:
                known_set = set(le.classes_)
                known_list = list(le.classes_)

                def _safe_enc(val, _le=le, _ks=known_set, _kl=known_list):
                    if val in _ks:
                        return _le.transform([val])[0]
                    nearest = _find_nearest_known(val, _kl, threshold=0.3)
                    if nearest is not None:
                        return _le.transform([nearest])[0]
                    return np.nan

                X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_enc)
            else:
                known_set = set(le.classes_)
                encoded = []
                for v in non_nan_values:
                    if v in known_set:
                        encoded.append(le.transform([v])[0])
                    else:
                        encoded.append(np.nan)
                X_df.loc[~nan_mask, col] = encoded
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    # Encode label
    if label_col in label_encoders:
        le_label = label_encoders[label_col]
        nan_mask_y = y_series.isna()
        y_encoded = y_series.copy()
        known_set_y = set(le_label.classes_)

        def _safe_enc_label(val, _le=le_label, _ks=known_set_y):
            val_str = str(val)
            if val_str in _ks:
                return _le.transform([val_str])[0]
            if _find_nearest_known is not None:
                nearest = _find_nearest_known(val_str, list(_le.classes_), threshold=0.3)
                if nearest is not None:
                    return _le.transform([nearest])[0]
            return np.nan

        y_encoded.loc[~nan_mask_y] = y_series.loc[~nan_mask_y].astype(str).map(_safe_enc_label)
        y_series = pd.to_numeric(y_encoded, errors='coerce')
    else:
        y_series = pd.to_numeric(y_series, errors='coerce')

    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    # Standardize (NaN-safe)
    X_out = X.copy()
    col_m = np.nanmean(X_out, axis=0)
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
    X_scaled = scaler.transform(X_out)

    return X_scaled, y


def compute_auth_div(X_result, X_clean, X_dirty):
    """Authenticity and diversity of a cleaned table.

    Args:
        X_result: cleaned data (may have fewer rows than X_clean)
        X_clean: clean data
        X_dirty: dirty data

    Returns:
        (auth, div), authenticity and diversity in [0, 1]
    """
    n = len(X_result)
    n_total = len(X_clean)
    if n == 0:
        return 0.0, 0.0

    # Authenticity: fraction of rows matching clean data exactly
    compare_len = min(n, n_total)
    correct = 0
    for i in range(compare_len):
        if np.allclose(X_result[i], X_clean[i], atol=0.01, equal_nan=False):
            correct += 1
    auth = correct / n if n > 0 else 0.0

    # Diversity: sample retention rate x noise retention rate
    sample_ret = n / n_total
    X_dirty_valid = X_dirty[~np.isnan(X_dirty).any(axis=1)]
    if len(X_result) > 1 and len(X_dirty_valid) > 1:
        result_var = np.mean(np.var(X_result, axis=0))
        clean_var = np.mean(np.var(X_clean, axis=0))
        dirty_var = np.mean(np.var(X_dirty_valid, axis=0))
        if dirty_var > clean_var + 1e-6:
            noise_ret = np.clip((result_var - clean_var) / (dirty_var - clean_var), 0, 1)
        else:
            noise_ret = 1.0
        div = sample_ret * noise_ret
    else:
        div = 0.0

    return round(auth, 4), round(div, 4)


def _inverse_transform_to_csv(
    X_result: np.ndarray,
    y_result: np.ndarray,
    feature_cols: list,
    label_col: str,
    scaler,
    label_encoders: dict,
    categorical_cols: set,
    dirty_df: pd.DataFrame,
    inference_mode: str = 'single_phase',
    keep_mask=None,
) -> pd.DataFrame:
    """
    Convert DemandClean's standardized numpy arrays back to a DataFrame in original CSV format.

    Strategy:
      1. Start from original dirty_df (keeping index column and raw formatting)
      2. Filter rows by keep_mask
      3. scaler.inverse_transform on standardized X_result
      4. LabelEncoder.inverse_transform on encoded categorical columns
      5. Inverse-transform the label column as well

    Args:
        X_result: cleaned standardized feature matrix (n_kept, n_features)
        y_result: cleaned label vector (n_kept,)
        feature_cols: feature column names
        label_col: label column name
        scaler: StandardScaler
        label_encoders: {col_name: LabelEncoder}
        categorical_cols: set of categorical column names
        dirty_df: original dirty DataFrame (with index column)
        inference_mode: inference mode
        keep_mask: boolean array marking which original rows are kept

    Returns:
        Cleaned DataFrame in original format (columns match dirty_df, includes index)
    """
    n_result = len(X_result)

    # Edge case: agent deleted all rows -> return empty DataFrame preserving dirty_df columns
    if n_result == 0:
        return pd.DataFrame(columns=dirty_df.columns)

    # Step 1: Inverse standardize
    X_inv = scaler.inverse_transform(X_result)

    # Step 2: Build the inverse-transformed DataFrame
    result_df = pd.DataFrame(X_inv, columns=feature_cols)

    # Step 3: Inverse-encode categorical columns
    if label_encoders and categorical_cols:
        for col in feature_cols:
            if col in categorical_cols and col in label_encoders:
                le = label_encoders[col]
                col_vals = result_df[col].values
                n_classes = len(le.classes_)

                if n_classes == 0:
                    continue
                # result_df already holds scaler.inverse_transform(X_result) (Step 1), so the
                # values are in integer-code space: round and clip to [0, n_classes-1].
                col_vals = np.asarray(col_vals, dtype=float)
                finite = np.isfinite(col_vals)
                col_rounded = np.clip(np.round(np.where(finite, col_vals, 0.0)).astype(int),
                                      0, n_classes - 1)

                try:
                    decoded = le.inverse_transform(col_rounded).astype(object)
                    decoded[~finite] = np.nan  # keep missing cells missing
                    result_df[col] = decoded
                except Exception:
                    # If inverse-transform fails, try edit-distance safe encoding
                    if _find_nearest_known is not None:
                        safe_vals = []
                        known = list(le.classes_)
                        for v in col_vals:
                            nearest = _find_nearest_known(str(v), known)
                            safe_vals.append(nearest if nearest else known[0])
                        result_df[col] = safe_vals
                    # Otherwise keep the numeric values

    # Step 4: Inverse-encode the label column
    y_inv = y_result[:n_result].copy()
    if label_col in label_encoders:
        le = label_encoders[label_col]
        n_classes = len(le.classes_)

        if n_classes > 0 and scaler is not None and label_col in feature_cols:
            # Label column may have been scaled (if part of features); nearest-neighbor match
            col_idx = feature_cols.index(label_col)
            mean_val = scaler.mean_[col_idx]
            scale_val = scaler.scale_[col_idx]
            valid_scaled = np.array([(i - mean_val) / scale_val for i in range(n_classes)])
            y_rounded = np.zeros(len(y_inv), dtype=int)
            for i, v in enumerate(y_inv):
                nearest_idx = np.argmin(np.abs(valid_scaled - v))
                y_rounded[i] = nearest_idx
        else:
            # Label column is usually not scaled; plain round + clip
            y_rounded = np.round(y_inv).astype(int)
            y_rounded = np.clip(y_rounded, 0, n_classes - 1)

        try:
            y_inv = le.inverse_transform(y_rounded)
        except Exception:
            pass
    result_df[label_col] = y_inv

    # Step 5: Restore the index column
    if 'index' in dirty_df.columns:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            # keep_mask matches rows in original dirty_df
            kept_indices = dirty_df.loc[keep_mask, 'index'].values[:n_result]
        else:
            # Default: take the first n_result indices
            kept_indices = dirty_df['index'].values[:n_result]
        result_df.insert(0, 'index', kept_indices)

    # Step 5.5: Restore non-feature columns dropped by preprocess_data (e.g. 'id').
    #   Columns present in dirty_df but not in feature_cols / label_col / 'index'
    #   are copied back from the corresponding rows of dirty_df.
    extra_cols = [c for c in dirty_df.columns
                  if c not in result_df.columns and c not in ('index',)]
    if extra_cols:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            src = dirty_df.loc[keep_mask].iloc[:n_result]
        else:
            src = dirty_df.iloc[:n_result]
        for col in extra_cols:
            result_df[col] = src[col].values

    # Step 6: Match column order to dirty_df
    target_cols = [c for c in dirty_df.columns if c in result_df.columns]
    result_df = result_df[target_cols]

    return result_df


# ============================================================================
# Configuration
# ============================================================================
INFERENCE_MODES = {'online': 'single_phase', 'offline': 'two_phase'}
DETECTORS = {'hybrid': 'auto', 'oracle': 'oracle'}
ACTION_NAMES = {'no_action': 'no_op', 'repair_value': 'gt_repair',
                'delete': 'delete', 'replace_nearby': 've_replace'}


def load_config(config_path, variant):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(_PROJECT_ROOT, 'configs', 'variants.yaml')) as f:
        variants = yaml.safe_load(f)
    if variant not in variants:
        raise SystemExit(f"unknown variant {variant!r}, choose from {sorted(variants)}")
    var = variants[variant]
    training = dict(cfg.get('training') or {})
    training.update(var.get('training') or {})
    return cfg, var, training


def package_versions():
    out = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'sklearn', 'scipy', 'torch', 'raha', 'yaml'):
        try:
            mod = __import__(name)
            out['scikit-learn' if name == 'sklearn' else name] = getattr(mod, '__version__', 'unknown')
        except Exception:
            out[name] = None
    return out


def code_revision():
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            return {'commit': rev.stdout.strip(), 'modified': bool(dirty.stdout.strip())}
    except Exception:
        pass
    return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _portable(value):
    """Absolute paths inside the project become relative, other absolute paths keep their name only."""
    if isinstance(value, str) and os.path.isabs(value):
        r = os.path.relpath(value, _PROJECT_ROOT)
        return r if not r.startswith('..') else os.path.basename(value.rstrip('/'))
    return value


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def _safe_prepare(X, y):
    """Drop rows with NaN label or all-NaN features, fill the remaining NaN with column means."""
    X_c = X.copy()
    y_c = y.copy() if len(y) == len(X) else y[:len(X)].copy()
    min_len = min(len(X_c), len(y_c))
    X_c = X_c[:min_len]
    y_c = y_c[:min_len]
    y_valid = ~np.isnan(y_c)
    X_c = X_c[y_valid]
    y_c = y_c[y_valid]
    x_valid = ~np.isnan(X_c).all(axis=1)
    X_c = X_c[x_valid]
    y_c = y_c[x_valid]
    col_means = np.nanmean(X_c, axis=0)
    for j in range(X_c.shape[1]):
        mask = np.isnan(X_c[:, j])
        X_c[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    return X_c, y_c


# ============================================================================
# Run
# ============================================================================
def run(args):
    global _find_nearest_known
    cfg, var, training = load_config(args.config, args.variant)
    if args.seed is None:
        args.seed = cfg.get('seed', 0)
    if args.unseen_categories is None:
        args.unseen_categories = (cfg.get('evaluation') or {}).get('unseen_categories', 'nan')
    _find_nearest_known = find_nearest_known if args.unseen_categories == 'nearest' else None
    dataset_name = cfg['dataset']
    DATASETS[dataset_name] = {
        'task_type': cfg['task'],
        'model_type': cfg['reward_model'],
        'label_col': cfg['label_col'],
        'categorical_cols': set(cfg.get('categorical_cols') or []),
    }
    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']
    n_episodes = args.n_episodes if args.n_episodes is not None else training.get('n_episodes', 300)
    detector_mode = DETECTORS[var.get('detector', 'hybrid')]
    agent_type = var.get('agent', 'dueling_two_stage')
    inference = var.get('inference') or cfg.get('inference', 'online')
    inference_mode = INFERENCE_MODES[inference]
    use_split = cfg.get('split', 'train_val_test') == 'train_val_test'

    started = datetime.now(timezone.utc).astimezone()
    run_id = args.run_id or f"{dataset_name}_{args.variant}_s{args.seed}_{started.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.abspath(args.out or os.path.join(_PROJECT_ROOT, 'results', 'runs', run_id))
    model_dir = os.path.join(out_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    if args.rules:
        os.environ['DC_RULES_PATH'] = os.path.abspath(args.rules)

    tee_out = TeeLogger(os.path.join(out_dir, 'run.log'), stream=sys.stdout)
    sys.stdout = tee_out
    sys.stderr = tee_out

    manifest = {
        'run_id': run_id,
        'method': var.get('name', 'DemandClean'),
        'variant': args.variant,
        'dataset': dataset_name,
        'command': ' '.join(['python', os.path.relpath(os.path.abspath(sys.argv[0]), _PROJECT_ROOT)]
                            + [_portable(a) for a in sys.argv[1:]]),
        'arguments': {k: _portable(v) for k, v in vars(args).items()},
        'seed': args.seed,
        'configuration': {'dataset': cfg, 'variant': var, 'training': training,
                          'n_episodes': n_episodes, 'detector': var.get('detector', 'hybrid'),
                          'agent': agent_type, 'inference': inference},
        'versions': package_versions(),
        'code': code_revision(),
        'platform': {'system': platform.system(), 'machine': platform.machine(),
                     'processor': platform.processor() or None},
        'start_time': started.isoformat(timespec='seconds'),
        'end_time': None,
        'status': 'running',
    }
    write_json(os.path.join(out_dir, 'manifest.json'), manifest)

    print("=" * 70)
    print(f"DemandClean | dataset: {dataset_name} | variant: {args.variant} | seed: {args.seed}")
    print(f"Task: {task_type} | reward model: {ds_cfg['model_type']} | detector: {var.get('detector', 'hybrid')} | "
          f"agent: {agent_type} | inference: {inference}")
    print(f"Episodes: {n_episodes} | training parameters: {training}")
    print("=" * 70)

    if args.seed is not None:
        seed_everything(args.seed)

    step_times = {}
    summary = {'run_id': run_id, 'dataset': dataset_name, 'variant': args.variant,
               'task': task_type, 'n_episodes': n_episodes}
    t_start = time.time()

    # ------------------------------------------------------------------
    # Data: strict 60/20/20 split of the raw CSV, encoders fitted on the dirty training split
    # ------------------------------------------------------------------
    if use_split:
        from sklearn.model_selection import train_test_split as _tts
        data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
        raw_dirty_df = pd.read_csv(os.path.join(data_dir, 'dirty_index.csv'))
        raw_clean_df = pd.read_csv(os.path.join(data_dir, 'clean_index.csv'))
        all_idx = np.arange(len(raw_dirty_df))
        train_idx, temp_idx = _tts(all_idx, test_size=0.4, random_state=42)
        val_idx, test_idx = _tts(temp_idx, test_size=0.5, random_state=42)
        dirty_train_df = raw_dirty_df.iloc[train_idx].reset_index(drop=True)
        clean_train_df = raw_clean_df.iloc[train_idx].reset_index(drop=True)
        print(f"\n  60/20/20 split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        split_dir = os.path.join(out_dir, 'split')
        os.makedirs(split_dir, exist_ok=True)
        dirty_csv_path = os.path.join(split_dir, 'dirty_train.csv')
        clean_csv_path = os.path.join(split_dir, 'clean_train.csv')
        dirty_train_df.to_csv(dirty_csv_path, index=False)
        clean_train_df.to_csv(clean_csv_path, index=False)

        (X_dirty, y_dirty, _, _, column_names, fd_rules, rules_path, _, _, csv_columns,
         data_scaler, label_encoders, categorical_cols, dirty_df, _) = preprocess_data(
            dataset_name, dirty_df=dirty_train_df, clean_df=None)

        X_clean, y_clean = encode_subset(clean_train_df, column_names, ds_cfg['label_col'],
                                         label_encoders, data_scaler, categorical_cols)
        X_clean_val, y_clean_val = encode_subset(
            raw_clean_df.iloc[val_idx].reset_index(drop=True), column_names, ds_cfg['label_col'],
            label_encoders, data_scaler, categorical_cols)
        X_clean_test, y_clean_test = encode_subset(
            raw_clean_df.iloc[test_idx].reset_index(drop=True), column_names, ds_cfg['label_col'],
            label_encoders, data_scaler, categorical_cols)
        clean_df = clean_train_df
        print(f"  encoded: X_dirty_train={X_dirty.shape}, X_clean_train={X_clean.shape}, "
              f"X_clean_val={X_clean_val.shape}, X_clean_test={X_clean_test.shape}")
    else:
        (X_dirty, y_dirty, X_clean, y_clean, column_names, fd_rules, rules_path,
         dirty_csv_path, clean_csv_path, csv_columns, data_scaler, label_encoders,
         categorical_cols, dirty_df, clean_df) = preprocess_data(dataset_name)
        X_clean_val = y_clean_val = X_clean_test = y_clean_test = None

    # ------------------------------------------------------------------
    # Step 1: DemandClean instance
    # ------------------------------------------------------------------
    print("\n[Step 1] creating DemandClean instance...")
    t_step = time.time()
    reward_model_type = training.get('reward_model_type') or ds_cfg['model_type']
    extra_kwargs = {}
    for key in ('min_repair_ratio', 'max_repair_ratio', 'repair_sensitivity',
                'max_truth_budget', 'repair_lambda', 'delete_shaping_reward', 'keep_rate_weight'):
        if training.get(key) is not None:
            extra_kwargs[key] = training[key]
    if training.get('raha_detector') is False:
        extra_kwargs['disable_raha'] = True
    if task_type == 'regression':
        extra_kwargs.setdefault('delete_shaping_reward', -0.05)
        extra_kwargs.setdefault('keep_rate_weight', 1.0)
        extra_kwargs.setdefault('regression_log_normalize', True)
    if use_split:
        extra_kwargs['use_clean_validation'] = True

    dc = DemandClean(
        task_type=task_type,
        model_type=reward_model_type,
        agent_type=agent_type,
        detector_mode=detector_mode,
        inference_mode=inference_mode,
        n_episodes=n_episodes,
        rules_path=rules_path,
        fd_rules=fd_rules,
        column_names=column_names,
        dirty_csv_path=dirty_csv_path,
        clean_csv_path=clean_csv_path,
        csv_columns=csv_columns,
        label_col=ds_cfg['label_col'],
        save_path=os.path.relpath(out_dir),
        apply_raha_truth=True,
        count_raha_cost=True,
        model_kwargs=REWARD_MODEL_KWARGS.get(reward_model_type, {}),
        eval_sample_ratio=1.0,
        encoding_label_encoders=label_encoders,
        encoding_scaler=data_scaler,
        encoding_categorical_cols=categorical_cols,
        encoding_dirty_df=dirty_df,
        encoding_clean_df=clean_df,
        **extra_kwargs,
    )
    step_times['init'] = round(time.time() - t_step, 2)

    # ------------------------------------------------------------------
    # Step 2: training, or loading a trained policy
    # ------------------------------------------------------------------
    model_path = os.path.join(model_dir, 'agent.pt')
    t_step = time.time()
    if args.checkpoint:
        src = os.path.join(os.path.abspath(args.checkpoint), 'agent.pt')
        if not ModelIO.agent_model_exists(src):
            raise FileNotFoundError(f"no trained policy in {args.checkpoint}")
        dc.load(src)
        print(f"\n[Step 2] training skipped, policy loaded from {os.path.relpath(src, _PROJECT_ROOT)}")
        manifest['checkpoint'] = {os.path.basename(p): sha256_of(p) for p in
                                  (src.replace('.pt', '_stage1.pt'), src.replace('.pt', '_stage2.pt'))
                                  if os.path.exists(p)}
        step_times['train'] = 0.0
    else:
        print(f"\n[Step 2] training ({n_episodes} episodes)...")
        dc.fit(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean, n_episodes=n_episodes,
               verbose=args.verbose, X_clean_val=X_clean_val, y_clean_val=y_clean_val)
        dc.save(os.path.relpath(model_path))
        history = dc.get_training_history()
        if history and history.get('score'):
            write_json(os.path.join(out_dir, 'training_history.json'),
                       {k: (v if k == 'action_probs' else [float(x) for x in v])
                        for k, v in history.items()})
        step_times['train'] = round(time.time() - t_step, 2)
        print(f"  [time] training: {step_times['train']:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: inference
    # ------------------------------------------------------------------
    print(f"\n[Step 3] inference ({'online' if inference_mode == 'single_phase' else 'offline'})...")
    t_step = time.time()
    if inference_mode == 'single_phase':
        X_result, y_result, stats = dc.clean(X_dirty, y_dirty, X_clean, y_clean=y_clean,
                                             verbose=args.verbose)
        ground_truth_used = stats.get('truth_cost', 0)
        action_counts = stats.get('action_counts', {})
        keep_mask = stats.get('keep_mask', None)
    else:
        repair_plan = dc.plan(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean, verbose=args.verbose)
        print(f"  repair plan: {len(repair_plan)} ground-truth values requested")
        true_values = {}
        for item in repair_plan:
            idx, col = item['idx'], item['col']
            if col == -1:
                if idx < len(y_clean):
                    true_values[(idx, col)] = y_clean[idx]
            elif idx < X_clean.shape[0] and col < X_clean.shape[1]:
                true_values[(idx, col)] = X_clean[idx, col]
        X_result, y_result, keep_mask = dc.execute(X_dirty, true_values, verbose=args.verbose,
                                                   y_dirty=y_dirty)
        ground_truth_used = len(true_values)
        action_counts = dc.get_action_counts()
    step_times['inference'] = round(time.time() - t_step, 2)
    print(f"  ground-truth repairs: {ground_truth_used}")
    print(f"  operations: {{{', '.join(f'{ACTION_NAMES.get(k, k)}: {v}' for k, v in action_counts.items())}}}")
    print(f"  [time] inference: {step_times['inference']:.2f}s")

    cleaned_df = _inverse_transform_to_csv(
        X_result=X_result, y_result=y_result, feature_cols=column_names,
        label_col=ds_cfg['label_col'], scaler=data_scaler, label_encoders=label_encoders,
        categorical_cols=categorical_cols, dirty_df=dirty_df, inference_mode=inference_mode,
        keep_mask=keep_mask)
    cleaned_df.to_csv(os.path.join(out_dir, 'cleaned.csv'), index=False)
    np.savez(os.path.join(out_dir, 'cleaned_encoded.npz'), X_result=X_result, y_result=y_result,
             column_names=np.array(column_names), label_col=np.array([ds_cfg['label_col']]))

    # ------------------------------------------------------------------
    # Step 4: detection accuracy
    # ------------------------------------------------------------------
    print("\n[Step 4] detection accuracy...")
    t_step = time.time()
    detected_errors = dc.detect_errors(X_dirty, X_clean, y_dirty=y_dirty, y_clean=y_clean,
                                       verbose=args.verbose)
    step_times['detection'] = round(time.time() - t_step, 2)
    detection = evaluate_detector_accuracy(detected_errors, X_dirty, X_clean, verbose=args.verbose,
                                           y_dirty=y_dirty, y_clean=y_clean)
    o = detection['overall']
    print(f"  P_d={o['precision']:.4f} R_d={o['recall']:.4f} F1={o['f1']:.4f}")

    # ------------------------------------------------------------------
    # Step 5: ground-truth cells and summary
    # ------------------------------------------------------------------
    seed_info = {}
    if getattr(dc, 'detector', None) is not None and hasattr(dc.detector, 'raha_cost_info'):
        seed_info = dc.detector.raha_cost_info or {}
    # Calibration supplies complete records. Count their non-index cells, including
    # the target column, rather than treating the number of labeled rows as cells.
    seed_rows = 0 if detector_mode == 'oracle' else int(seed_info.get('total_labeled_rows', 0))
    calibration_columns = sum(c != 'index' for c in dirty_df.columns)
    seed_cells = seed_rows * calibration_columns
    total_cells_train = X_dirty.shape[0] * calibration_columns
    gt_cells = seed_cells + ground_truth_used
    print(f"\n[Step 5] ground-truth cells: seed set {seed_cells} + repairs {ground_truth_used} = {gt_cells}")

    if use_split:
        X_dc, _ = _safe_prepare(X_result, y_result)
        auth, div = compute_auth_div(X_dc, X_clean, X_dirty)
    else:
        auth = div = None

    summary.update({
        'data_shape': list(X_dirty.shape),
        'cleaned_shape': list(X_result.shape),
        'operations': {ACTION_NAMES.get(k, k): int(v) for k, v in action_counts.items()},
        'ground_truth_cells': {'seed_set': int(seed_cells), 'repairs': int(ground_truth_used),
                               'total': int(gt_cells)},
        'calibration': {'labeled_rows': seed_rows, 'non_index_columns': calibration_columns},
        'ground_truth_fraction_of_train_split': round(gt_cells / total_cells_train, 6) if total_cells_train else None,
        'detection': {k: detection[k] for k in ('overall', 'missing', 'syntactic', 'semantic',
                                                 'value_error_combined', 'label_noise')},
        'authenticity': auth,
        'diversity': div,
        'time_seconds': {'detection': step_times.get('detection'),
                         'training': step_times.get('train'),
                         'inference': step_times.get('inference'),
                         'total': round(time.time() - t_start, 2)},
    })
    write_json(os.path.join(out_dir, 'summary.json'), summary)
    print(f"  time: {summary['time_seconds']}")

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------
    if not args.no_score:
        from evaluation.paper_score import score_run, write_scores
        print("\n[Evaluation] scoring the cleaned training split...")
        scores = score_run(out_dir, os.path.join(_PROJECT_ROOT, 'data'))
        write_scores(out_dir, scores)
        print(f"  {cfg['downstream_model']} {scores['metric']}: test={scores['test']:.4f}"
              + (f" | GT%={scores['gt_pct']:.4f}" if scores.get('gt_pct') is not None else ''))

    manifest['end_time'] = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    manifest['status'] = 'finished'
    manifest['outputs'] = sorted(os.path.relpath(os.path.join(r, f), out_dir)
                                 for r, _, fs in os.walk(out_dir) for f in fs)
    write_json(os.path.join(out_dir, 'manifest.json'), manifest)
    print(f"\nDone in {time.time() - t_start:.1f}s. Output: {os.path.relpath(out_dir, _PROJECT_ROOT)}")

    sys.stdout = tee_out.terminal
    sys.stderr = sys.__stderr__
    tee_out.close()
    return out_dir


def main():
    ap = argparse.ArgumentParser(description='Run DemandClean on one dataset.')
    ap.add_argument('--config', required=True, help='configs/datasets/<dataset>.yaml')
    ap.add_argument('--variant', default='full', choices=['full', 'ngt', 'nd', 'arch', 'stage'],
                    help='DemandClean or one of its ablations (configs/variants.yaml)')
    ap.add_argument('--seed', type=int, default=None,
                    help='seed of Python, NumPy and PyTorch (config seed, otherwise 0)')
    ap.add_argument('--n_episodes', type=int, default=None, help='overrides training.n_episodes of the config')
    ap.add_argument('--checkpoint', default=None, help='directory with agent_stage1.pt and agent_stage2.pt, skips training')
    ap.add_argument('--rules', default=None, help='alternative rule library')
    ap.add_argument('--unseen-categories', choices=['nan', 'nearest'], default=None,
                    help='override evaluation.unseen_categories from the dataset config')
    ap.add_argument('--out', default=None, help='output directory (default results/runs/<run_id>)')
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--no_score', action='store_true', help='skip the evaluation report')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
