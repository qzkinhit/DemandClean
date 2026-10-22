"""
Oracle detector
===============

Generates full error labels by comparing X_dirty against X_clean directly,
bypassing automatic detection. Used in ablation studies as an upper-bound
baseline for error detection.

Error classification:
1. Missing values: NaN or the string "empty"
2. Syntactic errors: diff > 3 * std
3. Semantic errors: any remaining value mismatch
"""

from typing import Dict, List, Optional
import os
import pickle
import numpy as np


class OracleDetector:
    """
    Oracle detector (for ablation studies).

    Compares X_dirty and X_clean directly to produce complete error labels.
    API-compatible with AutoDetector.
    """

    def __init__(self, column_names: Optional[List[str]] = None):
        """
        Initialize the Oracle detector.

        Args:
            column_names: optional column names (used only for logging)
        """
        self.column_names = column_names
        self.col_stats: Dict[int, Dict[str, float]] = {}
        self.is_fitted = True  # Oracle requires no training

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """Compute per-column statistics (mean / std) from clean data."""
        for col in range(X.shape[1]):
            valid = X[:, col][~np.isnan(X[:, col])]
            if len(valid) > 0:
                self.col_stats[col] = {
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid) + 1e-6),
                    'q1': float(np.percentile(valid, 25)),
                    'q3': float(np.percentile(valid, 75)),
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'median': float(np.median(valid))
                }

    @staticmethod
    def _is_missing(value) -> bool:
        """
        Check whether a value is missing.

        Matches on:
        - np.isnan (numeric NaN)
        - the string "empty" (case-insensitive)
        """
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and value.strip().lower() == "empty":
            return True
        return False

    # ------------------------------------------------------------------
    # Public interface (aligned with AutoDetector)
    # ------------------------------------------------------------------

    def fit(self, X_clean_subset: np.ndarray = None, verbose: bool = True) -> 'OracleDetector':
        """
        Fit (no-op).

        The Oracle detector does not require training; this method exists for
        interface compatibility.

        Args:
            X_clean_subset: clean-data subset (ignored)
            verbose: whether to print progress

        Returns:
            self
        """
        if verbose:
            print("[OracleDetector] fit() is a no-op; Oracle requires no training")
        self.is_fitted = True
        return self

    def detect(self,
               X_dirty: np.ndarray,
               X_clean: np.ndarray,
               y_dirty: Optional[np.ndarray] = None,
               y_clean: Optional[np.ndarray] = None,
               verbose: bool = True) -> Dict[str, List]:
        """
        Compare X_dirty and X_clean cell by cell to produce full error labels.
        Also detects label noise (y_dirty vs. y_clean).

        Args:
            X_dirty: dirty data, shape = (n, d)
            X_clean: clean data, shape = (n, d)
            y_dirty: dirty label vector (optional)
            y_clean: clean label vector (optional)
            verbose: whether to print details

        Returns:
            detected: {
                'missing':     [(idx, col, clean_val), ...],
                'semantic':    [(idx, col, clean_val, dirty_val), ...],
                'syntactic':   [(idx, col, clean_val, noise), ...],
                'label_noise': [(idx, -1, clean_label, dirty_label), ...]
            }
        """
        assert X_dirty.shape == X_clean.shape, (
            f"X_dirty {X_dirty.shape} and X_clean {X_clean.shape} shapes differ"
        )

        n_rows, n_cols = X_dirty.shape

        # Compute column statistics from clean data
        self._compute_col_stats(X_clean)

        detected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': []
        }

        # ---- Feature-error detection ----
        for i in range(n_rows):
            for col in range(n_cols):
                dirty_val = X_dirty[i, col]
                clean_val = X_clean[i, col]

                # ---- 1. Missing value ----
                if self._is_missing(dirty_val):
                    estimated_val = clean_val if not np.isnan(clean_val) else \
                        self.col_stats.get(col, {}).get('mean', 0)
                    detected['missing'].append((i, col, estimated_val))
                    continue

                # ---- Skip cells with no difference ----
                if not np.isnan(dirty_val) and not np.isnan(clean_val):
                    if dirty_val == clean_val:
                        continue
                else:
                    if np.isnan(clean_val):
                        continue

                # ---- 2. Values differ, classify as syntactic vs. semantic ----
                diff = abs(dirty_val - clean_val)
                col_std = self.col_stats.get(col, {}).get('std', 1.0)

                if diff > 3 * col_std:
                    noise = dirty_val - clean_val
                    detected['syntactic'].append((i, col, clean_val, noise))
                else:
                    detected['semantic'].append((i, col, clean_val, dirty_val))

        # ---- Label-noise detection ----
        if y_dirty is not None and y_clean is not None:
            assert len(y_dirty) == len(y_clean), (
                f"y_dirty ({len(y_dirty)}) and y_clean ({len(y_clean)}) lengths differ"
            )
            for i in range(len(y_dirty)):
                d_val = y_dirty[i]
                c_val = y_clean[i]
                # Skip when both are NaN
                if np.isnan(d_val) and np.isnan(c_val):
                    continue
                # Labels disagree
                if np.isnan(d_val) or np.isnan(c_val) or d_val != c_val:
                    detected['label_noise'].append((i, -1, c_val, d_val))

        if verbose:
            feat_total = (len(detected['missing'])
                          + len(detected['semantic'])
                          + len(detected['syntactic']))
            label_total = len(detected['label_noise'])
            total = feat_total + label_total
            total_cells = n_rows * n_cols
            print(f"\n[OracleDetector] Detection finished:")
            print(f"  Missing:      {len(detected['missing'])}")
            print(f"  Semantic:     {len(detected['semantic'])}")
            print(f"  Syntactic:    {len(detected['syntactic'])}")
            print(f"  Label noise:  {label_total}")
            print(f"  Total:        {total} errors  "
                  f"(features: {feat_total}/{total_cells}={feat_total/max(total_cells,1):.2%}, "
                  f"labels: {label_total}/{n_rows}={label_total/max(n_rows,1):.2%})")

        return detected

    def build_error_list(self,
                         detected: Dict[str, List],
                         X_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Convert detected errors into the format expected by the cleaning environment.

        Args:
            detected: error dict returned by detect()
            X_clean: clean data (used to fetch ground truths; detected already
                     contains truths under Oracle, and this takes precedence when
                     provided)

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
                type: 0=missing, 1=semantic, 2=syntactic
        """
        error_list = []

        # Missing errors (type=0)
        for item in detected['missing']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': repair_value
            })

        # Semantic errors (type=1)
        for item in detected['semantic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': repair_value
            })

        # Syntactic errors (type=2)
        for item in detected['syntactic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': repair_value
            })

        # Label noise errors (type=3, col=-1)
        for item in detected.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                # item format: (idx, -1, clean_label, dirty_label)
                clean_val = item[2] if len(item) > 2 else float('nan')
                repair_value = clean_val if not (isinstance(clean_val, float) and np.isnan(clean_val)) else float('nan')
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': repair_value
                })

        return error_list

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save detector parameters."""
        data = {
            'column_names': self.column_names,
            'col_stats': self.col_stats,
            'is_fitted': self.is_fitted
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  [OracleDetector] saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'OracleDetector':
        """Load detector parameters."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(column_names=data.get('column_names'))
        detector.col_stats = data.get('col_stats', {})
        detector.is_fitted = data.get('is_fitted', True)
        print(f"  [OracleDetector] loaded: {path}")
        return detector
