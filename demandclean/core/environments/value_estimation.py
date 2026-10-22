"""
Value estimator
===============

Unified value-estimation algorithm shared by CleaningEnv and
TwoPhaseCleaningEnv.

Priority chain: FD-rule derivation -> DC-rule derivation -> CFD baseline
  regression -> numeric extraction -> edit-distance estimation ->
  multi-dim KNN (k=5) weighted mean -> DOMAIN-range clipping.

Fallback strategy:
  - Missing fd_rules / column_names -> skip FD derivation.
  - Missing rich_rules['dc_rules'] -> skip DC derivation.
  - Missing rich_rules['cfd_rules'] -> skip CFD derivation.
  - Missing dirty_df / label_encoders -> skip edit-distance estimation.
  - Missing rich_rules -> skip DOMAIN clipping.
  - Missing scaler -> DOMAIN/DC/CFD use raw-space values.
  - KNN finds no neighbors (k=0) -> fall back to col_means.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import re
import logging
import numpy as np
import pandas as pd

from ...config import DemandCleanConfig
from ...utils.edit_distance import find_nearest_known


class ValueEstimator:
    """
    Unified value estimator.

    Estimates the value of a given cell from the FD rules, the DOMAIN rules and
    the scaler information held in the configuration.

    Usage:
        estimator = ValueEstimator(config)
        estimated = estimator.estimate_feature_value(X, idx, col, deleted_rows, col_means)
    """

    def __init__(self, config: DemandCleanConfig):
        """
        Initialise the value estimator.

        Args:
            config: configuration object holding fd_rules, column_names, rich_rules, scaler and related fields
        """
        self.config = config

        # Indices of categorical columns (column name -> column index)
        self.categorical_col_indices: Set[int] = set()
        self._build_categorical_index()

        # FD index: {rhs_col_idx: [(lhs_col_idx_list), ...]}
        self.fd_index: Dict[int, List[List[int]]] = {}
        self._build_fd_index()

        # DOMAIN ranges: {col_idx: (encoded_min, encoded_max)} or None
        self.domain_ranges: Dict[int, Tuple[float, float]] = {}
        self._build_domain_index()

        # DC index: {col_idx: [dc_rule_dict, ...]}
        self.dc_index: Dict[int, List[Dict]] = {}
        self._build_dc_index()

        # CFD index: {col_idx: [cfd_rule_dict, ...]}
        self.cfd_index: Dict[int, List[Dict]] = {}
        self._build_cfd_index()

        # Raw values of categorical columns: col_idx -> [str, ...] (used by the edit-distance estimate)
        self._cat_col_original_values: Dict[int, List[str]] = {}
        self._build_cat_col_original_values()

    # ====================================================================
    # Index construction
    # ====================================================================

    def _build_categorical_index(self) -> None:
        """Build the categorical column index: set of column names -> set of column indices."""
        if not self.config.categorical_cols or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for col_name in self.config.categorical_cols:
            idx = col_name_to_idx.get(col_name)
            if idx is not None:
                self.categorical_col_indices.add(idx)

    def _is_categorical(self, col: int) -> bool:
        """Return whether a column is categorical."""
        return col in self.categorical_col_indices

    def _build_fd_index(self) -> None:
        """Build the FD rule index: rhs_col_idx -> [lhs_col_idx_list, ...]."""
        if not self.config.fd_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }

        for lhs_str, rhs_str in self.config.fd_rules:
            # The LHS may list several comma-separated columns
            lhs_cols = [c.strip() for c in lhs_str.split(',')]
            lhs_indices = []
            for c in lhs_cols:
                if c in col_name_to_idx:
                    lhs_indices.append(col_name_to_idx[c])

            rhs_idx = col_name_to_idx.get(rhs_str.strip())

            if lhs_indices and rhs_idx is not None:
                self.fd_index.setdefault(rhs_idx, []).append(lhs_indices)

    def _build_domain_index(self) -> None:
        """Build the DOMAIN range index: col_idx -> (encoded_min, encoded_max)."""
        if not self.config.rich_rules or not self.config.column_names:
            return

        domain_rules = self.config.rich_rules.get('domain_rules', [])
        if not domain_rules:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }

        for rule in domain_rules:
            col_name = rule.get('column', '')
            dtype = rule.get('dtype', '')
            min_val = rule.get('min_val')
            max_val = rule.get('max_val')

            # Range constraints only for INT/FLOAT columns (ENUM columns are not clipped)
            if dtype not in ('INT', 'FLOAT') or min_val is None or max_val is None:
                continue

            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                continue

            # Convert to the encoded space
            encoded_min, encoded_max = self._transform_to_encoded_space(
                col_idx, min_val, max_val
            )
            if encoded_min is not None:
                self.domain_ranges[col_idx] = (encoded_min, encoded_max)

    def _build_dc_index(self) -> None:
        """Build the DC rule index: col_idx -> [dc_rule_dict, ...].

        A serialised DC rule holds
        - raw: the rule string
        - clauses: the list of clauses
        - mark_cols: the target columns marked with MARK
        - involved_cols: every column the rule mentions

        Indexing:
        - with mark_cols, the mark_cols are the target columns
        - without mark_cols, every involved column is a potential target
        """
        if not self.config.rich_rules:
            return
        dc_rules = self.config.rich_rules.get('dc_rules', [])
        if not dc_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for rule in dc_rules:
            target_cols = rule.get('mark_cols', []) or rule.get('involved_cols', [])
            for col_name in target_cols:
                if col_name in col_name_to_idx:
                    col_idx = col_name_to_idx[col_name]
                    self.dc_index.setdefault(col_idx, []).append(rule)

    def _build_cfd_index(self) -> None:
        """Build the CFD rule index: col_idx -> [cfd_rule_dict, ...].

        A serialised CFD rule holds
        - conditions: [(col, op, val), ...]
        - target_col: name of the target column
        - direction: 'EXCESS' or 'DEFICIT'
        - threshold: deviation threshold
        - baseline: baseline value
        """
        if not self.config.rich_rules:
            return
        cfd_rules = self.config.rich_rules.get('cfd_rules', [])
        if not cfd_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for rule in cfd_rules:
            col_name = rule.get('target_col', '')
            if col_name in col_name_to_idx:
                col_idx = col_name_to_idx[col_name]
                self.cfd_index.setdefault(col_idx, []).append(rule)

    def _build_cat_col_original_values(self) -> None:
        """Map each categorical col_idx to the list of its raw string values.

        Requires config.label_encoders, config.categorical_cols and config.column_names.
        """
        label_encoders = getattr(self.config, 'label_encoders', None)
        cat_cols = self.config.categorical_cols
        col_names = self.config.column_names

        if not label_encoders or not cat_cols or not col_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(col_names)
        }
        for col_name in cat_cols:
            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                continue
            le = label_encoders.get(col_name)
            if le is None or not hasattr(le, 'classes_'):
                continue
            self._cat_col_original_values[col_idx] = list(le.classes_)

    def _transform_to_encoded_space(
            self, col_idx: int, raw_min: float, raw_max: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Convert a raw-space range [min, max] to the encoded space."""
        scaler = self.config.scaler
        if scaler is None:
            # No scaler: use the raw values
            return raw_min, raw_max

        try:
            # StandardScaler: encoded = (raw - mean) / scale
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                mean = scaler.mean_[col_idx]
                scale = scaler.scale_[col_idx]
                if scale == 0 or np.isnan(scale):
                    return raw_min, raw_max
                encoded_min = (raw_min - mean) / scale
                encoded_max = (raw_max - mean) / scale
                return float(encoded_min), float(encoded_max)
        except (IndexError, AttributeError):
            pass

        # Incompatible scaler: use the raw values
        return raw_min, raw_max

    # ====================================================================
    # Core estimation
    # ====================================================================

    def estimate_feature_value(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate the feature value of one cell.

        Order of strategies: FD derivation, DC derivation, CFD baseline, numeric extraction,
        edit-distance estimate, multivariate KNN, DOMAIN clipping.

        Args:
            X: current data matrix
            idx: target row index
            col: target column index
            deleted_rows: set of deleted rows
            col_means: column means (fallback)
            dirty_df_row_indices: array mapping rows of X to rows of dirty_df.
                Required when X is a subset of dirty_df (for example X_base after DeleteFix)
                so that the edit-distance and numeric-extraction strategies find the raw dirty string.
                With None the behaviour is unchanged (requires len(X) == len(dirty_df)).

        Returns:
            the estimated feature value
        """
        # Step 1: FD derivation
        fd_result = self._try_fd_derivation(X, idx, col, deleted_rows)
        if fd_result is not None:
            return self._clip_domain(col, fd_result)

        # Step 2: DC derivation
        dc_result = self._try_dc_derivation(X, idx, col)
        if dc_result is not None:
            return self._clip_domain(col, dc_result)

        # Step 3: CFD derivation
        cfd_result = self._try_cfd_derivation(col)
        if cfd_result is not None:
            return self._clip_domain(col, cfd_result)

        # Step 3.5: numeric extraction (syntactic error: take the number out of the raw string)
        numeric_result = self._try_numeric_extraction(X, idx, col, dirty_df_row_indices)
        if numeric_result is not None:
            return self._clip_domain(col, numeric_result)

        # Step 3.6: edit-distance estimate (typo repair in categorical columns)
        edit_dist_result = self._try_edit_distance_estimation(X, idx, col, dirty_df_row_indices)
        if edit_dist_result is not None:
            return self._clip_domain(col, edit_dist_result)

        # Step 4: multivariate KNN
        knn_result = self._multidim_knn(X, idx, col, deleted_rows, col_means)
        if knn_result is not None:
            return self._clip_domain(col, knn_result)

        # Step 5: Fallback
        # Categorical column: most frequent value of the column (mode)
        # Numeric column: column mean
        if self._is_categorical(col):
            fallback = self._col_mode(X, col, deleted_rows)
        else:
            fallback = col_means[col] if not np.isnan(col_means[col]) else 0.0
        return self._clip_domain(col, fallback)

    # ====================================================================
    # FD derivation
    # ====================================================================

    def _try_fd_derivation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
    ) -> Optional[float]:
        """
        Derive the value of the target column from the FD rules.

        If col is the RHS of an FD:
        1. take the LHS values of the current row
        2. find every row with the same LHS values (encoded space, |a-b| < 1e-6)
        3. exclude the row itself and deleted rows
        4. require at least 2 matching rows

        Returns:
            the derived value, or None (falls back to KNN)
        """
        if col not in self.fd_index:
            return None

        for lhs_cols in self.fd_index[col]:
            # LHS values of the target row
            lhs_values = X[idx, lhs_cols]

            # Skip an LHS that contains NaN
            if np.any(np.isnan(lhs_values)):
                continue

            # Rows with the same LHS values
            candidates = []
            for i in range(len(X)):
                if i == idx or i in deleted_rows:
                    continue

                row_lhs = X[i, lhs_cols]
                if np.any(np.isnan(row_lhs)):
                    continue

                # Tolerance match in the encoded space
                if np.all(np.abs(row_lhs - lhs_values) < 1e-6):
                    rhs_val = X[i, col]
                    if not np.isnan(rhs_val):
                        candidates.append(rhs_val)

            # Require at least 2 matches
            if len(candidates) < 2:
                continue

            candidates = np.array(candidates)

            # Categorical column: always a majority vote (the output is a valid code)
            # Numeric column: majority vote for few distinct values, mean otherwise
            if self._is_categorical(col):
                unique, counts = np.unique(candidates, return_counts=True)
                return float(unique[np.argmax(counts)])

            unique_vals = np.unique(candidates)
            if len(unique_vals) <= 5:
                unique, counts = np.unique(candidates, return_counts=True)
                return float(unique[np.argmax(counts)])

            # Many distinct values: mean
            return float(np.mean(candidates))

        return None

    # ====================================================================
    # DC derivation
    # ====================================================================

    def _try_dc_derivation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
    ) -> Optional[float]:
        """
        Derive the repair value of the target column from the DC rules.

        DCs have denial semantics: the constraint is violated when every clause holds.
        The estimate makes the constraint hold again, i.e. makes at least one clause false.

        Procedure:
        - find the clause on the current (MARK) column, the target clause
        - check that every other clause holds (otherwise the rule does not apply)
        - invert the target clause: the value that makes it false is the repair value

        Returns:
            repair value in the encoded space, or None (falls back to the next strategy)
        """
        if col not in self.dc_index:
            return None

        col_name_to_idx = {
            name: i for i, name in enumerate(self.config.column_names)
        }

        for rule in self.dc_index[col]:
            col_name = self.config.column_names[col]
            clauses = rule.get('clauses', [])
            mark_cols = rule.get('mark_cols', [])

            # Derive only for MARK columns (when MARK is defined)
            if mark_cols and col_name not in mark_cols:
                continue

            # Split the clauses into the one on the current column (target) and the others
            target_clause = None
            other_clauses_all_met = True

            for clause in clauses:
                clause_cols = clause.get('columns', [])
                if col_name in clause_cols:
                    # Clause on the current column
                    target_clause = clause
                    continue

                # Evaluate the clauses on the other columns
                if not self._evaluate_dc_clause_encoded(clause, X, idx, col_name_to_idx):
                    other_clauses_all_met = False
                    break

            if not other_clauses_all_met or target_clause is None:
                continue

            # Invert the target clause to get the repair value
            result = self._derive_from_dc_clause(
                target_clause, X, idx, col, col_name, col_name_to_idx
            )
            if result is not None:
                return result

        return None

    def _evaluate_dc_clause_encoded(
            self,
            clause: Dict,
            X: np.ndarray,
            row_idx: int,
            col_name_to_idx: Dict[str, int],
    ) -> bool:
        """
        Evaluate one DC clause in the encoded space.

        simple/simple_str clauses: the raw value of the rule is moved to the encoded space first.
        abs_diff clauses: compared directly in the encoded space (encoding keeps the sign of a difference).

        Returns:
            True if the clause holds (one condition of a violation)
        """
        clause_type = clause.get('type', '')

        if clause_type == 'abs_diff':
            col1_name, col2_name = clause['col1'], clause['col2']
            col1_idx = col_name_to_idx.get(col1_name)
            col2_idx = col_name_to_idx.get(col2_name)
            if col1_idx is None or col2_idx is None:
                return False
            try:
                v1 = float(X[row_idx, col1_idx])
                v2 = float(X[row_idx, col2_idx])
            except (ValueError, TypeError, IndexError):
                return False
            if np.isnan(v1) or np.isnan(v2):
                return False
            # The abs_diff threshold is converted to the encoded space
            threshold = clause['value']
            encoded_threshold = self._transform_threshold_to_encoded_space(
                col1_name, threshold
            )
            actual = abs(v1 - v2)
            return self._dc_compare(actual, clause['op'], encoded_threshold)

        elif clause_type in ('simple', 'simple_str'):
            col_name = clause['col']
            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                return False
            try:
                actual_encoded = float(X[row_idx, col_idx])
            except (ValueError, TypeError, IndexError):
                return False
            if np.isnan(actual_encoded):
                return False
            # Convert the raw value of the rule to the encoded space
            expected_encoded = self._transform_single_value_to_encoded_space(
                col_name, clause['value']
            )
            if expected_encoded is None:
                return False
            return self._dc_compare(actual_encoded, clause['op'], expected_encoded)

        return False

    @staticmethod
    def _dc_compare(actual: float, op: str, expected: float) -> bool:
        """Numeric DC comparison (same logic as in auto_detector)."""
        if op == 'GT':
            return actual > expected
        elif op == 'GTE':
            return actual >= expected
        elif op == 'LT':
            return actual < expected
        elif op == 'LTE':
            return actual <= expected
        elif op in ('EQ', 'IQ'):
            return abs(actual - expected) < 1e-6
        elif op == 'NEQ':
            return abs(actual - expected) > 1e-6
        return False

    def _derive_from_dc_clause(
            self,
            clause: Dict,
            X: np.ndarray,
            idx: int,
            col: int,
            col_name: str,
            col_name_to_idx: Dict[str, int],
    ) -> Optional[float]:
        """
        Derive the repair value from the target clause of a DC.

        Denial semantics: the clause holds when the constraint is violated.
        A repair makes the clause false.

        Cases:
        - NEQ(mark_col, val) holds when mark_col != val, repair to val
        - GT(mark_col, threshold) holds when mark_col > threshold, repair to threshold
        - LT(mark_col, threshold) holds when mark_col < threshold, repair to threshold
        - EQ(mark_col, val) holds when mark_col == val, no exact repair, skipped
        - abs_diff GT(ABS(col1-col2), t): set the column to the other column's value (difference 0)

        Returns:
            repair value in the encoded space, or None
        """
        clause_type = clause.get('type', '')
        op = clause.get('op', '')

        if clause_type in ('simple', 'simple_str'):
            clause_col = clause.get('col', '')
            raw_value = clause.get('value')

            if clause_col != col_name:
                return None

            if op == 'NEQ':
                # NEQ(col, val) holds when col != val: repair to val
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            elif op in ('GT', 'GTE'):
                # GT(col, threshold) holds when col > threshold: repair to threshold
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            elif op in ('LT', 'LTE'):
                # LT(col, threshold) holds when col < threshold: repair to threshold
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            # EQ holds when col == val. A repair needs col != val, which has no exact value
            return None

        elif clause_type == 'abs_diff':
            # GT(ABS(col1 - col2), threshold)
            col1_name = clause.get('col1', '')
            col2_name = clause.get('col2', '')

            # Current column is col1: repair to X[idx, col2_idx] (difference 0, within the threshold)
            if col1_name == col_name:
                other_idx = col_name_to_idx.get(col2_name)
                if other_idx is not None:
                    other_val = X[idx, other_idx]
                    if not np.isnan(other_val):
                        return float(other_val)

            # Current column is col2: repair to X[idx, col1_idx]
            if col2_name == col_name:
                other_idx = col_name_to_idx.get(col1_name)
                if other_idx is not None:
                    other_val = X[idx, other_idx]
                    if not np.isnan(other_val):
                        return float(other_val)

        return None

    # ====================================================================
    # CFD derivation
    # ====================================================================

    def _try_cfd_derivation(self, col: int) -> Optional[float]:
        """
        Derive the repair value of the target column from the CFD rules.

        A CFD flags a column that deviates from its baseline, and the estimate is the baseline.
        The baseline is defined in the raw CSV space and is converted to the encoded space.

        Returns:
            repair value in the encoded space, or None (falls back to the next strategy)
        """
        if col not in self.cfd_index:
            return None

        for rule in self.cfd_index[col]:
            baseline = rule.get('baseline')
            if baseline is not None:
                # Convert the baseline from the raw space to the encoded space
                col_name = self.config.column_names[col]
                encoded_baseline = self._transform_single_value_to_encoded_space(
                    col_name, baseline
                )
                if encoded_baseline is not None:
                    return encoded_baseline

        return None

    # ====================================================================
    # Numeric extraction (syntactic error repair)
    # ====================================================================

    # Numeric part of a string (integer or float, optional minus sign)
    _NUMERIC_RE = re.compile(r'-?\d+(?:\.\d+)?')

    def _try_numeric_extraction(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Extract the numeric part of a raw dirty string (syntactic error repair).

        Applies when pd.to_numeric produced NaN during encoding but the raw value contains a number,
        for example "12.0 oz" -> 12.0 and "0.05%" -> 0.05.

        Conditions:
        - idx can be mapped to a row of dirty_df (via dirty_df_row_indices or equal row counts)
        - the current cell is NaN (a successfully encoded cell needs no extraction)
        - numeric columns only
        - a raw value of NaN, None or an empty string is a true missing value and is skipped

        Args:
            X: current data matrix
            idx: target row index
            col: target column index
            dirty_df_row_indices: optional array mapping rows of X to rows of dirty_df

        Returns:
            extracted value in the encoded space, or None (not applicable)
        """
        # Preconditions
        dirty_df = self.config.dirty_df
        if dirty_df is None or not self.config.column_names:
            return None

        # Row number in dirty_df
        if dirty_df_row_indices is not None:
            if idx >= len(dirty_df_row_indices):
                return None
            dirty_idx = int(dirty_df_row_indices[idx])
        else:
            # Row counts differ: idx cannot be mapped to dirty_df
            if len(X) != len(dirty_df):
                return None
            dirty_idx = idx

        # No numeric extraction for categorical columns
        if self._is_categorical(col):
            return None

        # Current value is not NaN: encoding succeeded, nothing to extract
        current_val = X[idx, col]
        if not np.isnan(current_val):
            return None

        # Raw dirty string
        col_name = self.config.column_names[col]
        if col_name not in dirty_df.columns:
            return None

        raw_value = dirty_df.iloc[dirty_idx][col_name]

        # True missing value (NaN, None or empty string): skip
        if pd.isna(raw_value):
            return None
        raw_str = str(raw_value).strip()
        if raw_str == '':
            return None

        # Try to extract a number from the string
        match = self._NUMERIC_RE.search(raw_str)
        if match is None:
            return None

        try:
            extracted = float(match.group())
        except ValueError:
            return None

        # Convert to the encoded space
        encoded = self._transform_single_value_to_encoded_space(col_name, extracted)
        if encoded is not None:
            logger = logging.getLogger('demandclean.value_estimation')
            logger.debug(
                f"NumericExtraction: row={idx}, col={col_name}, "
                f"raw='{raw_str}' → {extracted} → encoded={encoded:.4f}"
            )
        return encoded

    # ====================================================================
    # Edit-distance estimate (typo repair in categorical columns)
    # ====================================================================

    def _try_edit_distance_estimation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Repair a typo in a categorical column by edit distance.

        Categorical columns only. Takes the raw dirty string from dirty_df, finds the nearest known
        category with find_nearest_known() and converts it to the encoded space.

        Conditions:
        - categorical columns only
        - idx can be mapped to a row of dirty_df (via dirty_df_row_indices or equal row counts)
        - the raw dirty string differs from the nearest known value (it is a typo)
        - the current code is not the column mode (the mode is probably correct)

        The threshold of 0.6 is strict, so only clear typos are repaired.

        Args:
            X: current data matrix
            idx: target row index
            col: target column index
            dirty_df_row_indices: optional array mapping rows of X to rows of dirty_df

        Returns:
            repair value in the encoded space, or None (not applicable)
        """
        # Precondition: categorical columns only
        if not self._is_categorical(col):
            return None

        # Needs the raw values of the categorical column
        if col not in self._cat_col_original_values:
            return None

        known_values = self._cat_col_original_values[col]
        if not known_values:
            return None

        # Needs dirty_df and the column names
        dirty_df = self.config.dirty_df
        if dirty_df is None or not self.config.column_names:
            return None

        # Row number in dirty_df
        if dirty_df_row_indices is not None:
            if idx >= len(dirty_df_row_indices):
                return None
            dirty_idx = int(dirty_df_row_indices[idx])
        else:
            # Row counts differ: idx cannot be mapped to dirty_df
            if len(X) != len(dirty_df):
                return None
            dirty_idx = idx

        # Raw dirty string
        col_name = self.config.column_names[col]
        if col_name not in dirty_df.columns:
            return None

        raw_value = dirty_df.iloc[dirty_idx][col_name]
        if pd.isna(raw_value):
            return None
        raw_str = str(raw_value).strip()
        if not raw_str:
            return None

        # Edit-distance match (strict threshold)
        nearest = find_nearest_known(raw_str, known_values, threshold=0.6)
        if nearest is None:
            return None

        # Nearest value equals the raw value: no typo, skip
        if nearest == raw_str:
            return None

        # Mode check: skip if the current code is the column mode
        # (the most frequent value is usually correct)
        current_val = X[idx, col]
        if not np.isnan(current_val):
            col_vals = X[:, col]
            valid_vals = col_vals[~np.isnan(col_vals)]
            if len(valid_vals) > 0:
                unique, counts = np.unique(valid_vals, return_counts=True)
                mode_val = unique[np.argmax(counts)]
                if abs(current_val - mode_val) < 1e-6:
                    return None

        # Convert the nearest value to the encoded space
        label_encoders = getattr(self.config, 'label_encoders', None)
        if not label_encoders or col_name not in label_encoders:
            return None

        le = label_encoders[col_name]
        scaler = self.config.scaler

        try:
            le_val = le.transform([nearest])[0]
        except (ValueError, KeyError):
            return None

        # Move to the LE+SS space
        if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            try:
                mean = scaler.mean_[col]
                scale = scaler.scale_[col]
                if scale > 0 and not np.isnan(scale):
                    encoded = float((le_val - mean) / scale)
                else:
                    encoded = float(le_val)
            except (IndexError, AttributeError):
                encoded = float(le_val)
        else:
            encoded = float(le_val)

        logger = logging.getLogger('demandclean.value_estimation')
        logger.debug(
            f"EditDistance: row={idx}, col={col_name}, "
            f"raw='{raw_str}' → nearest='{nearest}' → encoded={encoded:.4f}"
        )
        return encoded

    # ====================================================================
    # Encoded-space conversion helpers
    # ====================================================================

    def _transform_single_value_to_encoded_space(
            self, col_name: str, raw_value: Any
    ) -> Optional[float]:
        """
        Convert one raw CSV value to the LE+SS encoded space.

        raw_value -> float -> (val - mean) / scale

        Args:
            col_name: column name
            raw_value: raw value (float, int or str)

        Returns:
            value in the encoded space, or None (not convertible)
        """
        try:
            val = float(raw_value)
        except (ValueError, TypeError):
            return None

        scaler = self.config.scaler
        if scaler is None:
            return val  # No scaler: return the raw value

        if not self.config.column_names:
            return val

        try:
            col_idx = list(self.config.column_names).index(col_name)
            # StandardScaler: encoded = (val - mean) / scale
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                mean = scaler.mean_[col_idx]
                scale = scaler.scale_[col_idx]
                if scale > 0 and not np.isnan(scale):
                    return float((val - mean) / scale)
            return val
        except (ValueError, IndexError):
            return val

    def _transform_threshold_to_encoded_space(
            self, col_name: str, raw_threshold: float
    ) -> float:
        """
        Convert an abs_diff threshold to the encoded space.

        With a StandardScaler the threshold is divided by scale and the mean is not subtracted,
        because |encoded_a - encoded_b| = |raw_a - raw_b| / scale.

        Args:
            col_name: reference column name (gives the scale)
            raw_threshold: threshold in the raw space

        Returns:
            threshold in the encoded space
        """
        scaler = self.config.scaler
        if scaler is None or not self.config.column_names:
            return raw_threshold

        try:
            col_idx = list(self.config.column_names).index(col_name)
            if hasattr(scaler, 'scale_'):
                scale = scaler.scale_[col_idx]
                if scale > 0 and not np.isnan(scale):
                    return raw_threshold / scale
        except (ValueError, IndexError):
            pass

        return raw_threshold

    # ====================================================================
    # Multivariate KNN
    # ====================================================================

    def _multidim_knn(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            k: int = 5,
    ) -> Optional[float]:
        """
        Multivariate KNN estimate (target column excluded).

        Computes Euclidean distances on every feature except the target column and returns
        the inverse-distance weighted mean of the k nearest neighbours.

        Args:
            X: data matrix
            idx: target row index
            col: target column index
            deleted_rows: set of deleted rows
            col_means: column means (NaN fill)
            k: number of neighbours

        Returns:
            weighted mean, or None
        """
        n_rows, n_cols = X.shape
        if n_rows < 2:
            return None

        # Feature mask (target column excluded)
        feature_mask = np.ones(n_cols, dtype=bool)
        feature_mask[col] = False

        # Filled feature matrix (only for the distance computation)
        X_feat = X[:, feature_mask].copy()
        for c in range(X_feat.shape[1]):
            nan_mask = np.isnan(X_feat[:, c])
            if nan_mask.any():
                # Fill with the mean of the column
                orig_col = np.where(feature_mask)[0][c]
                fill_val = col_means[orig_col] if not np.isnan(col_means[orig_col]) else 0.0
                X_feat[nan_mask, c] = fill_val

        target_feat = X_feat[idx]

        # Distances
        distances = np.linalg.norm(X_feat - target_feat, axis=1)
        distances[idx] = np.inf

        # Exclude deleted rows
        for d_idx in deleted_rows:
            if d_idx < n_rows:
                distances[d_idx] = np.inf

        # Exclude rows whose target column is NaN
        target_col_vals = X[:, col]
        nan_in_target = np.isnan(target_col_vals)
        distances[nan_in_target] = np.inf

        # k nearest neighbours
        valid_count = (distances < np.inf).sum()
        k_actual = min(k, valid_count)
        if k_actual == 0:
            return None

        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_dists = distances[nearest_indices]
        nearest_vals = target_col_vals[nearest_indices]

        # Inverse-distance weights
        weights = 1.0 / (nearest_dists + 1e-8)
        weights /= weights.sum()

        # Categorical column: weighted majority vote (the output is an existing valid code)
        if self._is_categorical(col):
            return self._weighted_majority_vote(nearest_vals, weights)

        # Numeric column: weighted mean
        return float(np.average(nearest_vals, weights=weights))

    # ====================================================================
    # Categorical vote
    # ====================================================================

    @staticmethod
    def _weighted_majority_vote(values: np.ndarray, weights: np.ndarray) -> float:
        """
        Weighted majority vote.

        Sums the weights of each distinct value and returns the value with the largest total.
        The result is a valid code that occurs in values.

        Args:
            values: values of the neighbours
            weights: inverse-distance weights

        Returns:
            the valid code with the most votes
        """
        unique_vals = np.unique(values)
        if len(unique_vals) == 1:
            return float(unique_vals[0])

        best_val = unique_vals[0]
        best_weight = -1.0

        for v in unique_vals:
            mask = np.abs(values - v) < 1e-8
            total_w = weights[mask].sum()
            if total_w > best_weight:
                best_weight = total_w
                best_val = v

        return float(best_val)

    def _col_mode(self, X: np.ndarray, col: int, deleted_rows: Set[int]) -> float:
        """
        Mode of a categorical column (fallback).

        The most frequent value among non-NaN cells of non-deleted rows.

        Args:
            X: data matrix
            col: column index
            deleted_rows: set of deleted rows

        Returns:
            the mode code (a valid value)
        """
        vals = X[:, col].copy()
        # Exclude deleted rows
        for d_idx in deleted_rows:
            if d_idx < len(vals):
                vals[d_idx] = np.nan
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            return 0.0
        unique, counts = np.unique(valid, return_counts=True)
        return float(unique[np.argmax(counts)])

    # ====================================================================
    # DOMAIN clipping
    # ====================================================================

    def _clip_domain(self, col: int, value: float) -> float:
        """
        DOMAIN range clipping.

        If the column has a DOMAIN rule (INT/FLOAT), the value is clipped to the valid range
        in the encoded space.

        Args:
            col: column index
            value: value to clip

        Returns:
            the clipped value
        """
        if col in self.domain_ranges:
            encoded_min, encoded_max = self.domain_ranges[col]
            value = float(np.clip(value, encoded_min, encoded_max))
        return value

    # ====================================================================
    # Diagnostics
    # ====================================================================

    def get_estimation_source(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, str]:
        """
        Estimate a value and return its source (for tests and diagnostics).

        Returns:
            (estimated_value, source), where source is 'fd', 'dc', 'cfd', 'numeric_extraction',
            'edit_distance', 'knn', 'mode' or 'mean'
        """
        # Step 1: FD
        fd_result = self._try_fd_derivation(X, idx, col, deleted_rows)
        if fd_result is not None:
            return self._clip_domain(col, fd_result), 'fd'

        # Step 2: DC
        dc_result = self._try_dc_derivation(X, idx, col)
        if dc_result is not None:
            return self._clip_domain(col, dc_result), 'dc'

        # Step 3: CFD
        cfd_result = self._try_cfd_derivation(col)
        if cfd_result is not None:
            return self._clip_domain(col, cfd_result), 'cfd'

        # Step 3.5: numeric extraction
        numeric_result = self._try_numeric_extraction(X, idx, col, dirty_df_row_indices)
        if numeric_result is not None:
            return self._clip_domain(col, numeric_result), 'numeric_extraction'

        # Step 3.6: edit-distance estimate
        edit_dist_result = self._try_edit_distance_estimation(X, idx, col, dirty_df_row_indices)
        if edit_dist_result is not None:
            return self._clip_domain(col, edit_dist_result), 'edit_distance'

        # Step 4: KNN
        knn_result = self._multidim_knn(X, idx, col, deleted_rows, col_means)
        if knn_result is not None:
            return self._clip_domain(col, knn_result), 'knn'

        # Step 5: Fallback
        if self._is_categorical(col):
            fallback = self._col_mode(X, col, deleted_rows)
        else:
            fallback = col_means[col] if not np.isnan(col_means[col]) else 0.0
        return self._clip_domain(col, fallback), 'mode' if self._is_categorical(col) else 'mean'

    def summary(self) -> str:
        """Return a summary of the estimator configuration."""
        parts = []
        if self.fd_index:
            total_rules = sum(len(v) for v in self.fd_index.values())
            parts.append(f"FD={total_rules} rules ({len(self.fd_index)} RHS cols)")
        if self.dc_index:
            total_dc = sum(len(v) for v in self.dc_index.values())
            parts.append(f"DC={total_dc} rules ({len(self.dc_index)} target cols)")
        if self.cfd_index:
            total_cfd = sum(len(v) for v in self.cfd_index.values())
            parts.append(f"CFD={total_cfd} rules ({len(self.cfd_index)} target cols)")
        if self.domain_ranges:
            parts.append(f"DOMAIN={len(self.domain_ranges)} cols")
        if self.categorical_col_indices:
            parts.append(f"CAT={len(self.categorical_col_indices)} cols")
        if not parts:
            parts.append("no rules (KNN only)")
        return f"ValueEstimator({', '.join(parts)})"
