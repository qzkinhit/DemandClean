"""
Error injector
==============

Inject different kinds of errors into data for self-supervised training.

Design: error injection is the inverse of detection.
  - Semantic errors: rule-based reverse injection (DOMAIN/CFD/FD), simulating
    logical violations that RAHA cannot detect.
  - Syntactic errors: RAHA-aware, statistics-driven (OD-Gaussian/Histogram/PVD);
    no rules involved.
  - Label errors: conditional injection that mirrors the label-flip pattern
    observed by the detector.
  - Missing values: simply set to NaN.

Error type codes:
  0 = missing, 1 = semantic, 2 = syntactic, 3 = label_noise
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field

from ..utils.edit_distance import generate_typo, find_nearest_known, find_top_k_nearest


# ============================================================================
# Label-error pattern analysis
# ============================================================================

@dataclass
class LabelErrorPattern:
    """Label-error pattern observed by the detector."""
    flip_matrix: Dict[Tuple, int] = field(default_factory=dict)  # (from_class, to_class) -> count
    error_rate: float = 0.0
    is_symmetric: bool = True
    unique_classes: List = field(default_factory=list)
    # Extra fields for regression tasks
    is_regression: bool = False
    noise_std: float = 0.0      # estimated label-noise std
    label_mean: float = 0.0     # label mean
    label_std: float = 1.0      # label std


def analyze_label_error_pattern(
    detected_label_errors: List,
    y_dirty: np.ndarray,
    task_type: str = 'classification',
) -> LabelErrorPattern:
    """Analyze the label-error pattern returned by the detector.

    Args:
        detected_label_errors: label errors returned by the detector,
            formatted as [(row_idx, col=-1, ...), ...] or
            [{'idx': ..., 'col': -1}, ...]
        y_dirty: dirty label vector
        task_type: 'classification' or 'regression'

    Returns:
        LabelErrorPattern describing the flip distribution (classification) or
        noise distribution (regression).
    """
    pattern = LabelErrorPattern()
    valid_y = y_dirty[~np.isnan(y_dirty)]
    unique_classes = list(np.unique(valid_y))
    pattern.unique_classes = unique_classes

    if not detected_label_errors:
        return pattern

    # Extract error row indices
    error_indices = set()
    for item in detected_label_errors:
        if isinstance(item, (list, tuple)):
            error_indices.add(int(item[0]))
        elif isinstance(item, dict):
            error_indices.add(int(item.get('idx', item.get(0, -1))))

    if not error_indices:
        return pattern

    total_errors = len(error_indices)
    pattern.error_rate = total_errors / max(1, len(y_dirty))

    # ============================================================
    # Regression: estimate noise std rather than building a flip_matrix
    # ============================================================
    if task_type == 'regression':
        pattern.is_regression = True
        pattern.label_mean = float(np.mean(valid_y))
        pattern.label_std = float(np.std(valid_y)) + 1e-8

        # Estimate noise magnitude from how much error-flagged labels deviate
        # from the global distribution.
        error_labels = np.array([y_dirty[i] for i in error_indices
                                  if i < len(y_dirty) and not np.isnan(y_dirty[i])])
        if len(error_labels) > 0:
            # Estimate: half the mean absolute deviation (conservative).
            deviations = np.abs(error_labels - pattern.label_mean)
            pattern.noise_std = float(np.mean(deviations)) * 0.5  # conservative
            # Lower bound: at least 10% of the label std
            pattern.noise_std = max(pattern.noise_std, pattern.label_std * 0.1)
        else:
            pattern.noise_std = pattern.label_std * 0.2

        return pattern

    # ============================================================
    # Classification: build the flip_matrix (original logic)
    # ============================================================
    # Per-class error counts
    class_error_count = defaultdict(int)
    for idx in error_indices:
        if idx < len(y_dirty) and not np.isnan(y_dirty[idx]):
            class_error_count[y_dirty[idx]] += 1

    if len(unique_classes) == 2:
        c0, c1 = unique_classes[0], unique_classes[1]
        n_c0_err = class_error_count.get(c0, 0)
        n_c1_err = class_error_count.get(c1, 0)
        if n_c0_err > 0:
            pattern.flip_matrix[(c0, c1)] = n_c0_err
        if n_c1_err > 0:
            pattern.flip_matrix[(c1, c0)] = n_c1_err
        pattern.is_symmetric = abs(n_c0_err - n_c1_err) < max(1, total_errors * 0.3)
    else:
        # Multi-class: uniformly flip to every other class
        for cls_from, count in class_error_count.items():
            others = [c for c in unique_classes if c != cls_from]
            per_other = max(1, count // len(others)) if others else 0
            for cls_to in others:
                pattern.flip_matrix[(cls_from, cls_to)] = per_other

    return pattern


# ============================================================================
# ErrorInjector main class
# ============================================================================

class ErrorInjector:
    """
    Error injector.

    Injects four types of errors onto the base data for training:
    - Missing (type=0): set the value to NaN.
    - Semantic (type=1): rule-based reverse injection (DOMAIN/CFD/FD);
      falls back to random replacement when no rules are available.
    - Syntactic (type=2): RAHA-aware, statistics-driven (simulating
      OD-Gaussian / Histogram / PVD).
    - Label (type=3): conditional label flips that mirror the detected pattern.
    """

    def __init__(self, X_base: np.ndarray, y_base: np.ndarray,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 column_names: Optional[List[str]] = None,
                 rich_rules: Optional[Dict[str, Any]] = None,
                 label_encoders: Optional[Dict[str, Any]] = None,
                 scaler: Optional[Any] = None,
                 categorical_cols: Optional[set] = None,
                 dirty_df: Optional[Any] = None,
                 label_col: Optional[str] = None):
        """
        Initialize the error injector.

        Args:
            X_base: base data (dirty data with missing rows removed, treated as
                relatively clean)
            y_base: labels
            fd_rules: list of FD rules [("lhs_col", "rhs_col"), ...]
            column_names: list of data column names
            rich_rules: rich-rules dict (from rule_parser.rules_to_dict()),
                containing domain_rules, cfd_rules, etc.
            label_encoders: {col_name: LabelEncoder} encoders
            scaler: StandardScaler
            categorical_cols: set of categorical column names
            dirty_df: dirty DataFrame in the raw CSV space
            label_col: label column name
        """
        self.X_base = X_base.copy()
        self.y_base = y_base.copy()
        self.fd_rules = fd_rules or []
        self.column_names = column_names or []
        self.rich_rules = rich_rules

        # Encoders
        self.label_encoders = label_encoders or {}
        self.scaler = scaler
        self.categorical_cols = categorical_cols or set()
        self.dirty_df = dirty_df
        self.label_col = label_col
        self._has_encoding_tools = bool(self.scaler is not None)

        # Compute column statistics
        self.col_means = np.nanmean(X_base, axis=0)
        self.col_stds = np.nanstd(X_base, axis=0)
        self.col_percentiles: Dict[int, Tuple[float, float]] = {}
        self.all_values: Dict[int, np.ndarray] = {}

        for col in range(X_base.shape[1]):
            valid = X_base[:, col][~np.isnan(X_base[:, col])]
            self.all_values[col] = valid
            if len(valid) > 0:
                self.col_percentiles[col] = (
                    np.percentile(valid, 1),
                    np.percentile(valid, 99),
                )

        # Build the FD column-index map
        self.fd_col_pairs: List[Tuple[List[int], int]] = []
        self._build_fd_index()

        # FD primary-key column set (high-frequency LHS: columns appearing in
        # >=2 FD rules). In principle syntactic injection should avoid these
        # columns to prevent cascading FD false positives, but empirically
        # excluding them concentrates errors in categorical columns and causes
        # RAHA to over-detect (8 -> 150+), increasing overall FP. The set
        # therefore stays empty.
        from collections import Counter
        lhs_counter = Counter()
        for lhs_indices, _rhs_idx in self.fd_col_pairs:
            for li in lhs_indices:
                lhs_counter[li] += 1
        self._fd_lhs_cols: Set[int] = set()  # disabled: enabling hurts RAHA
        # Enabled version: {col for col, cnt in lhs_counter.items() if cnt >= 2}

        # Build DOMAIN / CFD / DC column-index maps
        self._domain_col_map: Dict[int, Dict] = {}   # col_idx -> domain_rule_dict
        self._cfd_col_map: Dict[str, List[Dict]] = {} # class_val -> [cfd_rule_dict, ...]
        self._dc_rule_list: List[Dict] = []            # list of DC-rule dicts
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            self._build_rich_rule_index()

        # When encoders are available, pre-convert rule values into LE+SS space
        if self._has_encoding_tools:
            self._convert_rules_to_encoded_space()

        # Categorical columns: col_idx -> list of original string values
        # (from LabelEncoder.classes_). Used by typo injection
        # (inverse-transform -> typo -> forward-transform).
        self._cat_col_original_values: Dict[int, List[str]] = {}
        self._cat_col_idx_set: Set[int] = set()
        self._build_categorical_col_map()

    def _build_categorical_col_map(self):
        """Build the map: categorical col_idx -> list of original string values.

        Preconditions: self.label_encoders, self.categorical_cols, self.column_names.
        """
        if not self.label_encoders or not self.categorical_cols or not self.column_names:
            return

        for col_name in self.categorical_cols:
            if col_name not in self.column_names:
                continue
            col_idx = self.column_names.index(col_name)

            le = self.label_encoders.get(col_name)
            if le is None or not hasattr(le, 'classes_'):
                continue

            known_values = list(le.classes_)
            if len(known_values) >= 2:  # at least 2 categories are required
                self._cat_col_original_values[col_idx] = known_values
                self._cat_col_idx_set.add(col_idx)

    def _generate_categorical_typo_encoded(
            self, col: int, current_encoded: float
    ) -> Optional[float]:
        """Generate a syntactic anomaly for a categorical column by randomly
        substituting another valid LE category.

        Training/inference consistency: under a dirty-fit LE, typos at inference
        time are valid LE integers, so injected errors during training must also
        be valid LE integers (no more OOV extreme values).

        Args:
            col: column index
            current_encoded: current LE+SS encoded value

        Returns:
            New encoded value, or None.
        """
        if col not in self._cat_col_original_values:
            return None
        col_name = self.column_names[col]
        le = self.label_encoders.get(col_name)
        if le is None or self.scaler is None or col >= len(self.scaler.mean_):
            return None

        n_classes = len(le.classes_)
        if n_classes < 2:
            return None

        # Inverse-transform to recover the current LE integer
        scaler_mean = self.scaler.mean_[col]
        scaler_scale = self.scaler.scale_[col]
        current_le = int(round(current_encoded * scaler_scale + scaler_mean))
        current_le = max(0, min(current_le, n_classes - 1))

        # Randomly pick a different valid LE integer
        new_le = current_le
        for _ in range(20):
            new_le = np.random.randint(0, n_classes)
            if new_le != current_le:
                break
        if new_le == current_le:
            return None

        # LE -> SS encoding
        new_encoded = (new_le - scaler_mean) / scaler_scale
        return float(new_encoded)

    def _build_fd_index(self):
        """Build the FD rule column-index map."""
        if not self.fd_rules or not self.column_names:
            return

        for rule in self.fd_rules:
            if isinstance(rule, (list, tuple)) and len(rule) == 2:
                lhs_str, rhs_str = rule
                lhs_cols = [c.strip() for c in str(lhs_str).split(',')]
                rhs_col = str(rhs_str).strip()

                lhs_indices = []
                for c in lhs_cols:
                    if c in self.column_names:
                        lhs_indices.append(self.column_names.index(c))

                if rhs_col in self.column_names and lhs_indices:
                    rhs_idx = self.column_names.index(rhs_col)
                    self.fd_col_pairs.append((lhs_indices, rhs_idx))

    def _build_rich_rule_index(self):
        """Build DOMAIN / CFD rule column-index maps."""
        if not self.rich_rules:
            return

        # DOMAIN rules -> column index
        for rule in self.rich_rules.get('domain_rules', []):
            col_name = rule.get('column', '')
            if col_name in self.column_names:
                col_idx = self.column_names.index(col_name)
                self._domain_col_map[col_idx] = rule

        # CFD rules -> grouped by label value (supports any label column name)
        label_names = {'class'}
        if self.label_col:
            label_names.add(self.label_col)
        for rule in self.rich_rules.get('cfd_rules', []):
            for col, op, val in rule.get('conditions', []):
                if col in label_names and op == '=':
                    self._cfd_col_map.setdefault(val, []).append(rule)
                    break

        # DC rules (already a serialized list of dicts)
        self._dc_rule_list = self.rich_rules.get('dc_rules', [])

    def _convert_rules_to_encoded_space(self):
        """Pre-convert rule values from raw CSV space to LE+SS encoded space.

        Problem: DOMAIN/CFD rule values in rules.txt are in the raw CSV string
        space, while ErrorInjector operates on data encoded with
        LabelEncoder + StandardScaler.

        Strategy:
          - DOMAIN ENUM: use LabelEncoder to convert the enum list -> LE integers,
                        then StandardScaler -> LE+SS space.
          - DOMAIN INT/FLOAT: use StandardScaler to convert min_val/max_val
                              -> LE+SS space.
          - CFD class_val: use LabelEncoder to convert the class label value
                           -> LE space.
        """
        if not self.scaler:
            return

        scaler_mean = self.scaler.mean_
        scaler_scale = self.scaler.scale_

        # --- Convert DOMAIN rules ---
        for col_idx, rule in self._domain_col_map.items():
            col_name = rule.get('column', '')

            if rule.get('dtype') == 'ENUM':
                # ENUM: if it is a categorical column (has a LabelEncoder),
                # convert the enum values.
                if col_name in self.label_encoders and col_name in self.categorical_cols:
                    le = self.label_encoders[col_name]
                    enum_vals = rule.get('enum_vals', [])
                    if enum_vals:
                        try:
                            # Convert raw-string enum values to LE integers
                            le_vals = le.transform(enum_vals)
                            # Then StandardScaler -> LE+SS space
                            if col_idx < len(scaler_mean):
                                ss_vals = (le_vals - scaler_mean[col_idx]) / scaler_scale[col_idx]
                                # Store the encoded enum range (used to generate
                                # out-of-range injections).
                                rule['_encoded_enum_max'] = float(np.max(ss_vals))
                                rule['_encoded_enum_min'] = float(np.min(ss_vals))
                                rule['_encoded_enum_step'] = scaler_scale[col_idx]  # SS-space step per raw unit
                                rule['_encoding_converted'] = True
                        except (ValueError, KeyError):
                            # Some enum values are absent from the LabelEncoder
                            # (e.g. novel erroneous values).
                            rule['_encoding_converted'] = False
                else:
                    # Numeric-only ENUM (no LabelEncoder required)
                    try:
                        num_vals = [float(v) for v in rule.get('enum_vals', [])]
                        if num_vals and col_idx < len(scaler_mean):
                            ss_vals = [(v - scaler_mean[col_idx]) / scaler_scale[col_idx] for v in num_vals]
                            rule['_encoded_enum_max'] = max(ss_vals)
                            rule['_encoded_enum_min'] = min(ss_vals)
                            rule['_encoded_enum_step'] = 1.0 / scaler_scale[col_idx]
                            rule['_encoding_converted'] = True
                    except (ValueError, TypeError):
                        rule['_encoding_converted'] = False

            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                # INT/FLOAT: convert min_val, max_val into LE+SS space
                if col_idx < len(scaler_mean):
                    original_min = rule['min_val']
                    original_max = rule['max_val']
                    rule['_encoded_min'] = (original_min - scaler_mean[col_idx]) / scaler_scale[col_idx]
                    rule['_encoded_max'] = (original_max - scaler_mean[col_idx]) / scaler_scale[col_idx]
                    # SS-space step corresponding to 1 raw-space unit
                    rule['_encoded_unit_step'] = 1.0 / scaler_scale[col_idx]
                    rule['_encoding_converted'] = True

        # --- Convert CFD rules' class_val ---
        if self.label_col and self.label_col in self.label_encoders:
            le = self.label_encoders[self.label_col]
            new_cfd_map: Dict[str, List[Dict]] = {}
            for class_val, rules in self._cfd_col_map.items():
                try:
                    encoded_val = le.transform([class_val])[0]
                    encoded_key = str(int(encoded_val))
                    for rule in rules:
                        rule['_original_class_val'] = class_val
                        rule['_encoded_class_val'] = encoded_key
                    new_cfd_map.setdefault(encoded_key, []).extend(rules)
                except (ValueError, KeyError):
                    # class_val is not in the LabelEncoder; keep it as-is
                    new_cfd_map.setdefault(class_val, []).extend(rules)
            self._cfd_col_map = new_cfd_map

        # --- Convert DC rule clause values into LE+SS space ---
        for dc_rule in self._dc_rule_list:
            if dc_rule.get('_encoding_converted'):
                continue  # already converted

            all_cols_valid = True
            for clause in dc_rule.get('clauses', []):
                ctype = clause.get('type', '')

                if ctype == 'simple':
                    col_name = clause.get('col', '')
                    if col_name not in self.column_names:
                        all_cols_valid = False
                        break
                    col_idx = self.column_names.index(col_name)
                    raw_val = clause.get('value', 0.0)

                    if col_idx < len(scaler_mean):
                        # Categorical columns go through LabelEncoder first
                        if col_name in self.label_encoders and col_name in self.categorical_cols:
                            try:
                                le = self.label_encoders[col_name]
                                le_val = le.transform([str(int(raw_val))])[0]
                                encoded_val = (le_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                            except (ValueError, KeyError):
                                encoded_val = (raw_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                        else:
                            encoded_val = (raw_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                        clause['_encoded_value'] = encoded_val
                        clause['_col_idx'] = col_idx
                        clause['_scaler_scale'] = scaler_scale[col_idx]

                elif ctype == 'abs_diff':
                    col1_name = clause.get('col1', '')
                    col2_name = clause.get('col2', '')
                    if col1_name not in self.column_names or col2_name not in self.column_names:
                        all_cols_valid = False
                        break
                    col1_idx = self.column_names.index(col1_name)
                    col2_idx = self.column_names.index(col2_name)
                    raw_threshold = clause.get('value', 0.0)

                    # The abs_diff threshold is a difference, so both columns'
                    # scales must be considered; we approximate with the mean.
                    if col1_idx < len(scaler_scale) and col2_idx < len(scaler_scale):
                        avg_scale = (scaler_scale[col1_idx] + scaler_scale[col2_idx]) / 2.0
                        encoded_threshold = raw_threshold / avg_scale if avg_scale > 1e-10 else raw_threshold
                    else:
                        encoded_threshold = raw_threshold

                    clause['_encoded_value'] = encoded_threshold
                    clause['_col1_idx'] = col1_idx
                    clause['_col2_idx'] = col2_idx

            dc_rule['_encoding_converted'] = all_cols_valid

    def _has_cfd_for_label(self) -> bool:
        """Return True if any CFD rule covers the label column.

        Used to decide whether to allocate label injection from the semantic
        budget. Any CFD rule whose conditions include the label column
        (e.g. class=X => ...) lets AutoDetector detect label flips via CFD
        inference.
        """
        if not self.rich_rules or not self.rich_rules.get('has_rich_rules'):
            return False
        if not self.label_col:
            return False
        for rule in self.rich_rules.get('cfd_rules', []):
            for col, op, val in rule.get('conditions', []):
                if col == self.label_col or col == 'class':
                    return True
        return False

    # ====================================================================
    # Public interface
    # ====================================================================

    def inject_errors(self,
                      missing_rate: float = 0.05,
                      semantic_rate: float = 0.1,
                      syntactic_rate: float = 0.15,
                      label_rate: float = 0.0,
                      label_pattern: Optional[LabelErrorPattern] = None,
                      strict_semantic: bool = False,
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """
        Inject errors on top of the base data.

        Error taxonomy (aligned with AutoDetector detection channels):
          - syntactic: value-range / format anomalies = DOMAIN violations +
            RAHA-aware statistical anomalies.
          - semantic: logical violations = FD + CFD + DC violations.
          - missing: missing values.
          - label_noise: label flips.

        Args:
            missing_rate: fraction of missing values to inject
            semantic_rate: fraction of semantic errors (includes the label
                budget when CFD label rules exist)
            syntactic_rate: fraction of syntactic errors (DOMAIN violations +
                RAHA-aware)
            label_rate: deprecated label-error fraction; the label budget is
                now allocated from semantic_rate
            label_pattern: label-error pattern (from detector analysis or a
                uniform flip matrix)
            strict_semantic: strict mode, never fall back to undetectable
                random semantic injection

        Returns:
            (X_dirty, y_dirty, injected_errors)
            - X_dirty: features after injection
            - y_dirty: labels after injection
            - injected_errors: metadata
                {
                    'missing': [(idx, col, original_val), ...],
                    'semantic': [(idx, col, original_val, new_val), ...],
                    'syntactic': [(idx, col, original_val, noise), ...],
                    'label_noise': [(idx, -1, original_val, new_val), ...]
                }
        """
        X_dirty = self.X_base.copy()
        y_dirty = self.y_base.copy()
        n_samples, n_features = X_dirty.shape

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        used_indices: Set[Tuple[int, int]] = set()

        # 1. Inject missing values
        n_missing = int(n_samples * missing_rate)
        self._inject_missing(X_dirty, n_missing, used_indices, injected)

        # 2. Inject semantic + label errors
        #    Semantic = CFD + DC + FD (logical violations); DOMAIN is excluded.
        n_semantic_total = int(n_samples * semantic_rate)

        # Label-budget strategy:
        #   - label_rate > 0 (legacy callers, e.g. trainer.py): independent
        #     of the semantic budget.
        #   - label_rate == 0 (new callers): take ~20% from semantic_rate
        #     when CFD label rules exist.
        has_label_rules = self._has_cfd_for_label()
        n_label = 0
        label_from_semantic = False
        if label_rate > 0 and label_pattern is not None:
            # Backward compatible: use explicit label_rate, independent budget
            n_label = int(n_samples * label_rate)
        elif has_label_rules and label_pattern is not None:
            # New logic: allocate ~20% from the semantic budget
            n_label = max(1, int(n_semantic_total * 0.2))
            label_from_semantic = True

        n_semantic = n_semantic_total - n_label if label_from_semantic else n_semantic_total

        # Feature-column semantic injection (CFD + DC + FD only; no DOMAIN)
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            count = self._inject_semantic_no_domain(X_dirty, y_dirty, n_semantic, used_indices, injected)
            remaining = n_semantic - count
            if remaining > 0 and self.fd_col_pairs:
                self._inject_fd_violations(X_dirty, remaining, used_indices, injected,
                                           strict=strict_semantic)
            elif remaining > 0 and not strict_semantic:
                self._inject_random_semantic(X_dirty, remaining, used_indices, injected)
        elif self.fd_col_pairs:
            self._inject_fd_violations(X_dirty, n_semantic, used_indices, injected,
                                       strict=strict_semantic)
        elif not strict_semantic:
            self._inject_random_semantic(X_dirty, n_semantic, used_indices, injected)

        # 3. Inject syntactic errors = DOMAIN violations + RAHA-aware anomalies
        n_syntactic = int(n_samples * syntactic_rate)

        # DOMAIN violations (30% of the syntactic budget when rules exist)
        n_domain = 0
        if self._domain_col_map:
            n_domain = int(n_syntactic * 0.3)
            self._inject_domain_violations(X_dirty, n_domain, used_indices, injected)

        # RAHA-aware statistical anomalies (remaining syntactic budget)
        n_raha_syntactic = n_syntactic - n_domain
        self._inject_raha_aware_syntactic(X_dirty, n_raha_syntactic, used_indices, injected)

        # 4. Label errors (allocated from the semantic budget; only when CFD
        #    label rules exist).
        if n_label > 0 and label_pattern is not None:
            # Collect rows already carrying feature errors. Label injection
            # must avoid them; otherwise the detector's "feature-damaged row
            # exclusion" filters these out as TPs.
            feature_damaged_rows: Set[int] = set()
            for item in injected['missing']:
                feature_damaged_rows.add(item[0])
            for item in injected['semantic']:
                feature_damaged_rows.add(item[0])
            for item in injected['syntactic']:
                feature_damaged_rows.add(item[0])
            self._inject_label_noise(y_dirty, n_label, label_pattern, injected,
                                     exclude_rows=feature_damaged_rows)

        return X_dirty, y_dirty, injected

    def inject_on_dirty(self,
                        X_dirty: np.ndarray,
                        y_dirty: np.ndarray,
                        detected_cells: Set[Tuple[int, int]],
                        missing_rate: float = 0.05,
                        semantic_rate: float = 0.1,
                        syntactic_rate: float = 0.15,
                        label_rate: float = 0.0,
                        label_pattern: Optional[LabelErrorPattern] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """Inject additional errors on top of dirty data (for self-supervised training).

        Only injects into cells that are neither flagged by the detector nor NaN.

        Args:
            X_dirty: full dirty data (including detected errors)
            y_dirty: dirty labels
            detected_cells: set of positions already flagged by the detector,
                {(row, col), ...}
            The remaining parameters match inject_errors.

        Returns:
            (X_augmented, y_augmented, injected_new)
        """
        X_aug = X_dirty.copy()
        y_aug = y_dirty.copy()
        n_samples, n_features = X_aug.shape

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        # Neither already-detected positions nor NaN cells may be re-injected
        used_indices = set(detected_cells)
        for i in range(n_samples):
            for j in range(n_features):
                if np.isnan(X_aug[i, j]):
                    used_indices.add((i, j))

        # Injection logic is identical to inject_errors
        n_missing = int(n_samples * missing_rate)
        self._inject_missing(X_aug, n_missing, used_indices, injected)

        # Semantic = CFD + DC + FD (no DOMAIN)
        n_semantic = int(n_samples * semantic_rate)
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            count = self._inject_semantic_no_domain(X_aug, y_aug, n_semantic, used_indices, injected)
            remaining = n_semantic - count
            if remaining > 0 and self.fd_col_pairs:
                self._inject_fd_violations(X_aug, remaining, used_indices, injected)
            elif remaining > 0:
                self._inject_random_semantic(X_aug, remaining, used_indices, injected)
        elif self.fd_col_pairs:
            self._inject_fd_violations(X_aug, n_semantic, used_indices, injected)
        else:
            self._inject_random_semantic(X_aug, n_semantic, used_indices, injected)

        # Syntactic = DOMAIN + RAHA-aware
        n_syntactic = int(n_samples * syntactic_rate)
        n_domain = 0
        if self._domain_col_map:
            n_domain = int(n_syntactic * 0.3)
            self._inject_domain_violations(X_aug, n_domain, used_indices, injected)
        self._inject_raha_aware_syntactic(X_aug, n_syntactic - n_domain, used_indices, injected)

        if label_rate > 0 and label_pattern is not None:
            n_label = int(n_samples * label_rate)
            self._inject_label_noise(y_aug, n_label, label_pattern, injected)

        return X_aug, y_aug, injected

    # ====================================================================
    # 1. Missing-value injection
    # ====================================================================

    def _inject_missing(self, X_dirty: np.ndarray, n_missing: int,
                        used_indices: Set[Tuple[int, int]],
                        injected: Dict[str, List]):
        """Inject missing values (type=0)."""
        n_samples, n_features = X_dirty.shape
        for _ in range(n_missing):
            idx = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                X_dirty[idx, col] = np.nan
                injected['missing'].append((idx, col, original_val))
                used_indices.add((idx, col))

    # ====================================================================
    # 2. Semantic-error injection (rule-based: DOMAIN/CFD/DC/FD)
    # ====================================================================

    def _inject_rule_based_semantic(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                     n_semantic: int,
                                     used_indices: Set[Tuple[int, int]],
                                     injected: Dict[str, List]) -> int:
        """Rule-based reverse injection of semantic errors with DOMAIN + CFD +
        DC + FD (kept for backward compatibility).

        Note: the new inject_errors() no longer calls this method and uses
        _inject_semantic_no_domain() instead. Retained for legacy callers such
        as inject_on_dirty().

        Priority: DOMAIN (40%) > CFD (30%) > DC (30%) > FD fallback.

        Returns:
            Number of errors actually injected.
        """
        total_injected = 0

        # DOMAIN violations (40%)
        n_domain = int(n_semantic * 0.4)
        if self._domain_col_map:
            total_injected += self._inject_domain_violations(
                X_dirty, n_domain, used_indices, injected)

        # CFD violations (30%)
        n_cfd = int(n_semantic * 0.3)
        if self._cfd_col_map:
            total_injected += self._inject_cfd_violations(
                X_dirty, y_dirty, n_cfd, used_indices, injected)

        # DC violations (remainder)
        n_dc = n_semantic - total_injected
        if self._dc_rule_list:
            total_injected += self._inject_dc_violations(
                X_dirty, n_dc, used_indices, injected)

        return total_injected

    def _inject_semantic_no_domain(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                    n_semantic: int,
                                    used_indices: Set[Tuple[int, int]],
                                    injected: Dict[str, List]) -> int:
        """Semantic injection with CFD + DC only (DOMAIN has moved to syntactic).

        Aligned with AutoDetector's semantic channel:
          - AutoDetector semantic = FD + CFD + DC
          - ErrorInjector semantic = CFD + DC + FD

        CFD 50% > DC 50% > FD fallback.

        Returns:
            Number of errors actually injected.
        """
        total_injected = 0

        # CFD violations (50%)
        n_cfd = int(n_semantic * 0.5)
        if self._cfd_col_map:
            total_injected += self._inject_cfd_violations(
                X_dirty, y_dirty, n_cfd, used_indices, injected)

        # DC violations (remainder)
        n_dc = n_semantic - total_injected
        if self._dc_rule_list:
            total_injected += self._inject_dc_violations(
                X_dirty, n_dc, used_indices, injected)

        return total_injected

    def _inject_domain_violations(self, X_dirty: np.ndarray, n_target: int,
                                   used_indices: Set[Tuple[int, int]],
                                   injected: Dict[str, List]) -> int:
        """Inject DOMAIN violations (values outside the valid range);
        classified as syntactic.

        Aligned with AutoDetector: DOMAIN detection runs on the syntactic
        channel (Stage 2c), so injections also land in injected['syntactic'].

        When encoders are available, uses boundary values pre-converted into
        LE+SS space to generate out-of-range values. Otherwise (no encoders),
        falls back to the raw rule values for backward compatibility.

        INT [1, 10]     -> inject values outside [encoded_min, encoded_max] in LE+SS space
        ENUM {a, b, c}  -> inject values outside [encoded_enum_min, encoded_enum_max] in LE+SS space
        """
        n_samples = len(X_dirty)
        count = 0
        domain_cols = list(self._domain_col_map.keys())

        if not domain_cols:
            return 0

        for _ in range(n_target * 3):  # extra retries
            if count >= n_target:
                break

            col_idx = np.random.choice(domain_cols)
            idx = np.random.randint(0, n_samples)

            if (idx, col_idx) in used_indices or np.isnan(X_dirty[idx, col_idx]):
                continue

            rule = self._domain_col_map[col_idx]
            original_val = X_dirty[idx, col_idx]
            new_val = None

            if rule.get('dtype') == 'ENUM':
                if rule.get('_encoding_converted'):
                    # Use the pre-converted encoded-space range
                    max_encoded = rule['_encoded_enum_max']
                    # In LE+SS space, one raw-enum unit ~ 1/scaler_scale
                    step = rule.get('_encoded_enum_step', 1.0)
                    unit_step = 1.0 / step if step > 0 else 0.5
                    new_val = max_encoded + np.random.choice([1, 2, 3]) * unit_step
                else:
                    # No encoders or conversion failed: use the actual value
                    # range from this column of the data matrix.
                    col_vals = self.X_base[:, col_idx]
                    valid_vals = col_vals[~np.isnan(col_vals)]
                    if len(valid_vals) > 0:
                        max_data = float(np.max(valid_vals))
                        std_data = float(np.std(valid_vals)) if len(valid_vals) > 1 else 1.0
                        new_val = max_data + np.random.choice([1, 2, 3]) * max(std_data * 0.5, 0.1)
                    else:
                        new_val = np.random.choice([1, 2, 3])

            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                if rule.get('_encoding_converted'):
                    # Use the pre-converted encoded-space bounds
                    encoded_min = rule['_encoded_min']
                    encoded_max = rule['_encoded_max']
                    unit_step = rule.get('_encoded_unit_step', 0.5)
                    if np.random.random() < 0.5:
                        # Above the upper bound
                        new_val = encoded_max + np.random.randint(1, 6) * unit_step
                    else:
                        # Below the lower bound
                        new_val = encoded_min - np.random.randint(1, 6) * unit_step
                else:
                    # No encoders: use raw values (backward compatible; may be
                    # less accurate).
                    min_v = rule['min_val']
                    max_v = rule['max_val']
                    if np.random.random() < 0.5:
                        new_val = max_v + np.random.randint(1, 6)
                    else:
                        new_val = min_v - np.random.randint(1, 6)

            if new_val is not None and abs(new_val - original_val) > 1e-6:
                X_dirty[idx, col_idx] = new_val
                noise = new_val - original_val
                # DOMAIN violations are classified as syntactic (aligned with
                # AutoDetector Stage 2c).
                injected['syntactic'].append((idx, col_idx, original_val, noise))
                used_indices.add((idx, col_idx))
                count += 1

        return count

    def _inject_cfd_violations(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                n_target: int,
                                used_indices: Set[Tuple[int, int]],
                                injected: Dict[str, List]) -> int:
        """Inject CFD violations (conditional-functional-dependency violations).

        Example: class=2, n_anomaly<=2 => CT EXCESS >= 5 FROM_BASELINE 5
        -> in rows with class=2, inject CT = baseline + threshold + rand(0, 2).
        """
        n_samples = len(X_dirty)
        count = 0

        # Group row indices by class value
        class_row_map: Dict[str, List[int]] = defaultdict(list)
        for i in range(n_samples):
            if not np.isnan(y_dirty[i]):
                # Use a generic string conversion (regression labels may be floats)
                try:
                    class_key = str(int(y_dirty[i])) if y_dirty[i] == int(y_dirty[i]) else str(y_dirty[i])
                except (ValueError, OverflowError):
                    class_key = str(y_dirty[i])
                class_row_map[class_key].append(i)

        # Iterate over all CFD rules
        all_cfd_rules = []
        for class_val, rules in self._cfd_col_map.items():
            for rule in rules:
                all_cfd_rules.append((class_val, rule))

        if not all_cfd_rules:
            return 0

        per_rule = max(1, n_target // len(all_cfd_rules))

        for class_val, rule in all_cfd_rules:
            if count >= n_target:
                break

            target_col_name = rule.get('target_col', '')
            if target_col_name not in self.column_names:
                continue
            col_idx = self.column_names.index(target_col_name)

            direction = rule.get('direction', 'EXCESS')
            threshold = rule.get('threshold', 5.0)
            baseline = rule.get('baseline', 0.0)

            candidate_rows = class_row_map.get(class_val, [])
            if not candidate_rows:
                continue

            np.random.shuffle(candidate_rows)
            rule_count = 0

            for row_idx in candidate_rows:
                if rule_count >= per_rule or count >= n_target:
                    break
                if (row_idx, col_idx) in used_indices or np.isnan(X_dirty[row_idx, col_idx]):
                    continue

                original_val = X_dirty[row_idx, col_idx]

                # threshold and baseline must be mapped into LE+SS space
                if rule.get('_encoding_converted') or self._has_encoding_tools:
                    # Use values in encoded space
                    if self.scaler is not None and col_idx < len(self.scaler.mean_):
                        # Map baseline and threshold from raw space into LE+SS
                        mean_j = self.scaler.mean_[col_idx]
                        scale_j = self.scaler.scale_[col_idx]
                        encoded_baseline = (baseline - mean_j) / scale_j
                        encoded_threshold = threshold / scale_j  # threshold is a difference; divide by scale
                        encoded_delta = np.random.uniform(0, 2) / scale_j
                    else:
                        encoded_baseline = baseline
                        encoded_threshold = threshold
                        encoded_delta = np.random.uniform(0, 2)
                else:
                    encoded_baseline = baseline
                    encoded_threshold = threshold
                    encoded_delta = np.random.uniform(0, 2)

                if direction == 'EXCESS':
                    new_val = encoded_baseline + encoded_threshold + encoded_delta
                elif direction == 'DEFICIT':
                    new_val = encoded_baseline - encoded_threshold - encoded_delta
                else:
                    continue

                # DOMAIN-range clamping: keep the raw value inside DOMAIN to
                # avoid being flagged as syntactic by the DOMAIN channel
                # (this should be a semantic error).
                if col_idx in self._domain_col_map:
                    domain_rule = self._domain_col_map[col_idx]
                    enc_min = domain_rule.get('_encoded_min')
                    enc_max = domain_rule.get('_encoded_max')
                    if enc_min is not None and enc_max is not None:
                        # Leave some margin to avoid boundary-precision issues.
                        # margin = 1% of the DOMAIN range.
                        margin = abs(enc_max - enc_min) * 0.01
                        clamped = max(enc_min + margin, min(enc_max - margin, new_val))
                        # Ensure the clamped value still exceeds the CFD threshold
                        if direction == 'EXCESS':
                            if clamped - encoded_baseline >= encoded_threshold:
                                new_val = clamped
                            else:
                                continue  # cannot meet the CFD threshold inside DOMAIN
                        elif direction == 'DEFICIT':
                            if encoded_baseline - clamped >= encoded_threshold:
                                new_val = clamped
                            else:
                                continue

                if abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, col_idx] = new_val
                    injected['semantic'].append((row_idx, col_idx, original_val, new_val))
                    used_indices.add((row_idx, col_idx))
                    rule_count += 1
                    count += 1

        return count

    def _inject_dc_violations(self, X_dirty: np.ndarray, n_target: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List]) -> int:
        """Inject DC (Denial Constraint) violations.

        DC uses denial semantics: when every clause holds, the constraint is
        violated. Injection = force a violation = make all clauses hold.

        Strategy:
          1. Find rows that currently do NOT violate the constraint (at least
             one clause does not hold).
          2. For the MARK column: change its value so every clause holds.
          3. For abs_diff type (no MARK): change one of the two columns so the
             difference exceeds the threshold.

        All operations happen in encoded space (the numpy array).
        """
        n_samples = len(X_dirty)
        count = 0

        if not self._dc_rule_list:
            return 0

        # Keep only DC rules that were successfully encoded
        valid_dc_rules = [r for r in self._dc_rule_list if r.get('_encoding_converted')]

        # Debug log: DC rule injection status
        import logging
        _dc_logger = logging.getLogger('demandclean.error_injector')
        _dc_logger.debug(
            f"DC inject: total={len(self._dc_rule_list)}, "
            f"valid(encoded)={len(valid_dc_rules)}, target={n_target}")

        if not valid_dc_rules:
            return 0

        # When there are more rules than the budget, sample to avoid giving
        # every rule a quota of zero.
        if len(valid_dc_rules) > n_target:
            valid_dc_rules = list(np.random.choice(
                valid_dc_rules, size=n_target, replace=False))

        per_rule = max(1, n_target // len(valid_dc_rules))

        for dc_rule in valid_dc_rules:
            if count >= n_target:
                break

            clauses = dc_rule.get('clauses', [])
            mark_cols = dc_rule.get('mark_cols', [])

            if not clauses:
                continue

            # Dispatch to the appropriate sub-strategy
            if mark_cols:
                injected_count = self._inject_dc_mark_violation(
                    X_dirty, clauses, mark_cols,
                    min(per_rule, n_target - count),
                    used_indices, injected)
                count += injected_count
            else:
                # No MARK column: check whether abs_diff clauses are present
                abs_diff_clauses = [c for c in clauses if c.get('type') == 'abs_diff']
                if abs_diff_clauses:
                    injected_count = self._inject_dc_abs_diff_violation(
                        X_dirty, abs_diff_clauses,
                        min(per_rule, n_target - count),
                        used_indices, injected)
                    count += injected_count

        return count

    def _inject_dc_mark_violation(self, X_dirty: np.ndarray,
                                   clauses: List[Dict],
                                   mark_cols: List[str],
                                   n_target: int,
                                   used_indices: Set[Tuple[int, int]],
                                   injected: Dict[str, List]) -> int:
        """DC injection when MARK columns exist.

        Find rows that satisfy all non-MARK clauses but not the MARK clause
        (currently legal rows) and modify the MARK column so every clause
        holds (forcing a violation).

        Example: EQ(holiday, 1) & NEQ(workingday, 0) & MARK(workingday)
          - Find rows with holiday=1 and workingday=0 (legal today because
            NEQ(workingday, 0) does not hold).
          - Set workingday to a non-zero value (e.g. 1) so NEQ(workingday, 0)
            holds -> the constraint is violated.
        """
        n_samples = len(X_dirty)
        count = 0

        # Separate non-MARK and MARK clauses
        non_mark_clauses = []
        mark_clauses = []  # clauses that reference MARK columns

        for clause in clauses:
            col_name = clause.get('col', '')
            if col_name in mark_cols:
                mark_clauses.append(clause)
            else:
                non_mark_clauses.append(clause)

        # Resolve MARK column indices
        mark_col_indices = []
        for mc in mark_cols:
            if mc in self.column_names:
                mark_col_indices.append(self.column_names.index(mc))
            else:
                return 0  # MARK column is not among the features; skip

        if not mark_col_indices:
            return 0

        # Find candidate rows that satisfy every non-MARK clause
        candidate_rows = []
        for i in range(n_samples):
            all_non_mark_satisfied = True
            for clause in non_mark_clauses:
                if not self._evaluate_clause(X_dirty, i, clause):
                    all_non_mark_satisfied = False
                    break

            if all_non_mark_satisfied:
                # Check that the MARK clause does NOT hold
                # (currently legal = not violating the constraint).
                any_mark_unsatisfied = False
                for clause in mark_clauses:
                    if not self._evaluate_clause(X_dirty, i, clause):
                        any_mark_unsatisfied = True
                        break

                # If there are no mark_clauses (MARK columns without a clause),
                # still treat the row as a candidate, an injection can force
                # a violation directly.
                if any_mark_unsatisfied or not mark_clauses:
                    candidate_rows.append(i)

        if not candidate_rows:
            return 0

        np.random.shuffle(candidate_rows)

        for row_idx in candidate_rows:
            if count >= n_target:
                break

            # For each MARK column, compute a value that makes its clause hold
            for mc_idx, mark_col_name in zip(mark_col_indices, mark_cols):
                if (row_idx, mc_idx) in used_indices or np.isnan(X_dirty[row_idx, mc_idx]):
                    continue

                original_val = X_dirty[row_idx, mc_idx]
                new_val = self._compute_dc_mark_value(
                    X_dirty, row_idx, mc_idx, mark_col_name, clauses)

                if new_val is not None and abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, mc_idx] = new_val
                    injected['semantic'].append((row_idx, mc_idx, original_val, new_val))
                    used_indices.add((row_idx, mc_idx))
                    count += 1
                    break  # at most one MARK column per row

        return count

    def _inject_dc_abs_diff_violation(self, X_dirty: np.ndarray,
                                       abs_diff_clauses: List[Dict],
                                       n_target: int,
                                       used_indices: Set[Tuple[int, int]],
                                       injected: Dict[str, List]) -> int:
        """DC injection for abs_diff clauses (no MARK column).

        Example: GT(ABS(t1.col1 - t1.col2), threshold)
          - Find rows with |col1 - col2| <= threshold (currently legal).
          - Change col1 so |col1 - col2| > threshold (force a violation).
        """
        n_samples = len(X_dirty)
        count = 0

        for clause in abs_diff_clauses:
            if count >= n_target:
                break

            col1_idx = clause.get('_col1_idx')
            col2_idx = clause.get('_col2_idx')
            encoded_threshold = clause.get('_encoded_value')
            op = clause.get('op', 'GT')

            if col1_idx is None or col2_idx is None or encoded_threshold is None:
                continue

            # Find rows that currently do not violate the constraint
            # (clause does not hold = legal).
            candidate_rows = []
            for i in range(n_samples):
                if (i, col1_idx) in used_indices or (i, col2_idx) in used_indices:
                    continue
                if np.isnan(X_dirty[i, col1_idx]) or np.isnan(X_dirty[i, col2_idx]):
                    continue

                abs_diff = abs(X_dirty[i, col1_idx] - X_dirty[i, col2_idx])

                # Clause does not hold = legal (no violation)
                clause_holds = self._eval_comparison(abs_diff, op, encoded_threshold)
                if not clause_holds:
                    candidate_rows.append(i)

            if not candidate_rows:
                continue

            np.random.shuffle(candidate_rows)
            per_clause = max(1, (n_target - count) // max(1, len(abs_diff_clauses)))

            clause_count = 0
            for row_idx in candidate_rows:
                if clause_count >= per_clause or count >= n_target:
                    break

                # Pick col1 or col2 at random to modify
                target_col = col1_idx if np.random.random() < 0.5 else col2_idx
                other_col = col2_idx if target_col == col1_idx else col1_idx

                if (row_idx, target_col) in used_indices:
                    continue

                original_val = X_dirty[row_idx, target_col]
                other_val = X_dirty[row_idx, other_col]

                # Make |target - other| > threshold.
                # target = other + threshold + delta (or other - threshold - delta)
                delta = encoded_threshold * np.random.uniform(0.1, 0.5)
                if np.random.random() < 0.5:
                    new_val = other_val + encoded_threshold + delta
                else:
                    new_val = other_val - encoded_threshold - delta

                # DOMAIN-range clamping: keep the injected value inside DOMAIN
                # so the DOMAIN channel does not reclassify it as syntactic.
                if target_col in self._domain_col_map:
                    domain_rule = self._domain_col_map[target_col]
                    enc_min = domain_rule.get('_encoded_min')
                    enc_max = domain_rule.get('_encoded_max')
                    if enc_min is not None and enc_max is not None:
                        # 1% margin to avoid boundary-precision issues
                        margin = abs(enc_max - enc_min) * 0.01
                        clamped = max(enc_min + margin, min(enc_max - margin, new_val))
                        # Ensure the clamped value still violates the DC
                        if abs(clamped - other_val) > encoded_threshold:
                            new_val = clamped
                        else:
                            continue  # cannot violate the DC inside DOMAIN -> skip

                if abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, target_col] = new_val
                    injected['semantic'].append((row_idx, target_col, original_val, new_val))
                    used_indices.add((row_idx, target_col))
                    clause_count += 1
                    count += 1

        return count

    def _evaluate_clause(self, X_dirty: np.ndarray, row_idx: int,
                          clause: Dict) -> bool:
        """Evaluate a single DC clause on the given row.

        Args:
            X_dirty: data matrix (encoded space)
            row_idx: row index
            clause: DC clause dict

        Returns:
            True when the clause holds.
        """
        ctype = clause.get('type', '')

        if ctype == 'simple':
            col_idx = clause.get('_col_idx')
            encoded_val = clause.get('_encoded_value')
            if col_idx is None or encoded_val is None:
                return False
            if np.isnan(X_dirty[row_idx, col_idx]):
                return False

            cell_val = X_dirty[row_idx, col_idx]
            op = clause.get('op', 'EQ')
            return self._eval_comparison(cell_val, op, encoded_val)

        elif ctype == 'abs_diff':
            col1_idx = clause.get('_col1_idx')
            col2_idx = clause.get('_col2_idx')
            encoded_threshold = clause.get('_encoded_value')
            if col1_idx is None or col2_idx is None or encoded_threshold is None:
                return False
            if np.isnan(X_dirty[row_idx, col1_idx]) or np.isnan(X_dirty[row_idx, col2_idx]):
                return False

            abs_diff = abs(X_dirty[row_idx, col1_idx] - X_dirty[row_idx, col2_idx])
            op = clause.get('op', 'GT')
            return self._eval_comparison(abs_diff, op, encoded_threshold)

        return False

    @staticmethod
    def _eval_comparison(val: float, op: str, threshold: float,
                          tol: float = 1e-4) -> bool:
        """Evaluate a comparison operator.

        Args:
            val: left-hand value
            op: operator (EQ, NEQ, GT, GTE, LT, LTE)
            threshold: right-hand value
            tol: tolerance for EQ/NEQ (float comparison in encoded space)

        Returns:
            The comparison result.
        """
        if op == 'EQ':
            return abs(val - threshold) < tol
        elif op == 'NEQ':
            return abs(val - threshold) >= tol
        elif op == 'GT':
            return val > threshold
        elif op == 'GTE':
            return val >= threshold
        elif op == 'LT':
            return val < threshold
        elif op == 'LTE':
            return val <= threshold
        return False

    def _compute_dc_mark_value(self, X_dirty: np.ndarray, row_idx: int,
                                mark_col_idx: int, mark_col_name: str,
                                clauses: List[Dict]) -> Optional[float]:
        """Compute the value to inject in the MARK column so every clause holds.

        Locates the clause tied to the MARK column and derives a value that
        satisfies it.

        Args:
            X_dirty: data matrix
            row_idx: row index
            mark_col_idx: column index of the MARK column
            mark_col_name: name of the MARK column
            clauses: all condition clauses

        Returns:
            The encoded-space value to inject, or None.
        """
        # Find the clause that references the MARK column
        target_clause = None
        for clause in clauses:
            if clause.get('type') == 'simple' and clause.get('col') == mark_col_name:
                target_clause = clause
                break

        if target_clause is None:
            # No clause directly references this MARK column.
            # Fall back to picking a different value in the column to create
            # some kind of violation.
            col_vals = self.all_values.get(mark_col_idx, np.array([]))
            if len(col_vals) > 1:
                current_val = X_dirty[row_idx, mark_col_idx]
                # Randomly pick a different value
                new_val = np.random.choice(col_vals)
                attempts = 0
                while abs(new_val - current_val) < 1e-4 and attempts < 10:
                    new_val = np.random.choice(col_vals)
                    attempts += 1
                if abs(new_val - current_val) > 1e-4:
                    return float(new_val)
            return None

        encoded_val = target_clause.get('_encoded_value')
        op = target_clause.get('op', 'EQ')
        scaler_scale = target_clause.get('_scaler_scale', 1.0)
        # Encoded-space step corresponding to one raw unit
        unit_step = 1.0 / scaler_scale if scaler_scale > 1e-10 else 0.5

        if encoded_val is None:
            return None

        current_val = X_dirty[row_idx, mark_col_idx]

        # Compute a value that makes the clause hold, based on `op`
        if op == 'EQ':
            # Need col == val -> set to encoded_val
            return float(encoded_val)

        elif op == 'NEQ':
            # Need col != val -> encoded_val + offset
            offset = unit_step * np.random.choice([1, 2, -1, -2])
            new_val = encoded_val + offset
            # Ensure it is indeed != encoded_val
            if abs(new_val - encoded_val) < 1e-4:
                new_val = encoded_val + unit_step
            return float(new_val)

        elif op == 'GT':
            # Need col > val -> val + delta
            delta = unit_step * np.random.uniform(1, 3)
            return float(encoded_val + delta)

        elif op == 'GTE':
            # Need col >= val -> val + small delta
            delta = unit_step * np.random.uniform(0, 2)
            return float(encoded_val + delta)

        elif op == 'LT':
            # Need col < val -> val - delta
            delta = unit_step * np.random.uniform(1, 3)
            return float(encoded_val - delta)

        elif op == 'LTE':
            # Need col <= val -> val - small delta
            delta = unit_step * np.random.uniform(0, 2)
            return float(encoded_val - delta)

        return None

    def _inject_fd_violations(self, X_dirty: np.ndarray, n_semantic: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List],
                               strict: bool = False):
        """Inject semantic errors that violate FD rules by swapping RHS values
        across groups.

        Key design: at most strictly fewer than half of the rows in each FD
        group are injected, so the original value remains the majority vote
        and detect_fd_violations() can still flag the injected rows.

        Args:
            strict: strict mode, any remaining budget is not forwarded to
                _inject_random_semantic.
        """
        if not self.fd_col_pairs:
            return

        per_rule_budget = max(1, n_semantic // len(self.fd_col_pairs))
        total_injected = 0

        for lhs_indices, rhs_idx in self.fd_col_pairs:
            if total_injected >= n_semantic:
                break

            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(X_dirty)):
                if (i, rhs_idx) in used_indices or np.isnan(X_dirty[i, rhs_idx]):
                    continue
                lhs_vals = X_dirty[i, lhs_indices]
                if np.isnan(lhs_vals).any():
                    continue
                key = tuple(lhs_vals.tolist())
                groups[key].append(i)

            # Keep groups with >=3 rows so the original value stays strictly
            # in the majority after a single injection.
            group_keys = [k for k in groups if len(groups[k]) >= 3]
            if len(group_keys) < 2:
                if not strict:
                    self._inject_random_semantic_for_col(
                        X_dirty, per_rule_budget, rhs_idx, used_indices, injected)
                    total_injected += per_rule_budget
                continue

            rule_injected = 0
            np.random.shuffle(group_keys)
            for i, gk in enumerate(group_keys):
                if rule_injected >= per_rule_budget:
                    break
                rows = groups[gk]
                # At most floor((n-1)/2) rows per group so the original value
                # remains a strict majority.
                max_per_group = max(1, (len(rows) - 1) // 2)

                other_key = group_keys[(i + 1) % len(group_keys)]
                other_rows = groups[other_key]
                donor_row = np.random.choice(other_rows)
                donor_val = X_dirty[donor_row, rhs_idx]

                group_injected = 0
                for row_idx in rows:
                    if (rule_injected >= per_rule_budget
                            or total_injected >= n_semantic
                            or group_injected >= max_per_group):
                        break
                    if (row_idx, rhs_idx) in used_indices:
                        continue
                    original_val = X_dirty[row_idx, rhs_idx]
                    if abs(donor_val - original_val) > 1e-6:
                        X_dirty[row_idx, rhs_idx] = donor_val
                        injected['semantic'].append(
                            (row_idx, rhs_idx, original_val, donor_val))
                        used_indices.add((row_idx, rhs_idx))
                        rule_injected += 1
                        total_injected += 1
                        group_injected += 1

        remaining = n_semantic - total_injected
        if remaining > 0 and not strict:
            self._inject_random_semantic(X_dirty, remaining, used_indices, injected)

    def _inject_random_semantic_for_col(self, X_dirty: np.ndarray, n: int,
                                         col: int,
                                         used_indices: Set[Tuple[int, int]],
                                         injected: Dict[str, List]):
        """Random semantic injection restricted to a single column."""
        n_samples = len(X_dirty)
        count = 0
        for _ in range(n * 3):
            if count >= n:
                break
            idx = np.random.randint(0, n_samples)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                candidates = self.all_values.get(col, np.array([]))
                if len(candidates) > 1:
                    new_val = np.random.choice(candidates)
                    attempts = 0
                    while abs(new_val - original_val) < 0.01 and attempts < 10:
                        new_val = np.random.choice(candidates)
                        attempts += 1
                    X_dirty[idx, col] = new_val
                    injected['semantic'].append((idx, col, original_val, new_val))
                    used_indices.add((idx, col))
                    count += 1

    def _inject_random_semantic(self, X_dirty: np.ndarray, n_semantic: int,
                                 used_indices: Set[Tuple[int, int]],
                                 injected: Dict[str, List]):
        """Random semantic injection when no rules exist (replace with another value from the same column)."""
        n_samples, n_features = X_dirty.shape
        for _ in range(n_semantic):
            idx = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                candidates = self.all_values.get(col, np.array([]))
                if len(candidates) > 1:
                    new_val = np.random.choice(candidates)
                    attempts = 0
                    while abs(new_val - original_val) < 0.01 and attempts < 10:
                        new_val = np.random.choice(candidates)
                        attempts += 1
                    X_dirty[idx, col] = new_val
                    injected['semantic'].append((idx, col, original_val, new_val))
                    used_indices.add((idx, col))

    # ====================================================================
    # 3. Syntactic-error injection (RAHA-aware, statistics-driven; no rules)
    # ====================================================================

    def _inject_raha_aware_syntactic(self, X_dirty: np.ndarray, n_syntactic: int,
                                      used_indices: Set[Tuple[int, int]],
                                      injected: Dict[str, List]):
        """RAHA-aware syntactic-error injection.

        Three sub-strategies mirror RAHA detection in reverse:
          A (40%): detectable by OD-Gaussian      -> 2~4 sigma deviations
          B (30%): detectable by OD-Histogram     -> values outside extreme quantiles
          C (30%): detectable by PVD              -> magnitude anomalies (*10, *11, sign flip)

        Avoid FD LHS columns to prevent cascading false positives from FD detection.
        """
        n_samples, n_features = X_dirty.shape

        # Eligible syntactic columns = all columns minus FD LHS columns
        eligible_cols = [c for c in range(n_features) if c not in self._fd_lhs_cols]
        if not eligible_cols:
            eligible_cols = list(range(n_features))  # degrade when all columns are LHS

        # Allocate budget across strategies
        n_gaussian = int(n_syntactic * 0.4)
        n_histogram = int(n_syntactic * 0.3)
        n_pvd = n_syntactic - n_gaussian - n_histogram

        # Strategy A: OD-Gaussian (3~5 sigma deviations)
        self._inject_syntactic_gaussian(X_dirty, n_gaussian, used_indices, injected, eligible_cols)

        # Strategy B: OD-Histogram (0.3~1.0x IQR99 offsets)
        self._inject_syntactic_histogram(X_dirty, n_histogram, used_indices, injected, eligible_cols)

        # Strategy C: PVD (magnitude anomalies)
        self._inject_syntactic_pvd(X_dirty, n_pvd, used_indices, injected, eligible_cols)

    def _inject_syntactic_gaussian(self, X_dirty: np.ndarray, n: int,
                                    used_indices: Set[Tuple[int, int]],
                                    injected: Dict[str, List],
                                    eligible_cols: Optional[List[int]] = None):
        """Strategy A: inject 3~5 sigma deviations, reliably detectable by RAHA's OD-Gaussian.

        Rationale: 3~5 sigma has probability <0.3% under the standard normal,
        enough for a z-score detector to flag without being so extreme that
        RAHA's meta-classifier overfits labeled samples.
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # Categorical columns go through the typo path
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # skip numeric-column logic

                original_val = X_dirty[idx, col]
                std = self.col_stds[col] if not np.isnan(self.col_stds[col]) else 1.0
                if std < 1e-10:
                    std = 1.0
                # 3~5 sigma offset (moderate; avoids RAHA overfitting)
                sigma_mult = np.random.uniform(3.0, 5.0)
                direction = np.random.choice([-1, 1])
                noise = direction * sigma_mult * std
                X_dirty[idx, col] = self.col_means[col] + noise
                injected['syntactic'].append((idx, col, original_val, noise))
                used_indices.add((idx, col))

    def _inject_syntactic_histogram(self, X_dirty: np.ndarray, n: int,
                                     used_indices: Set[Tuple[int, int]],
                                     injected: Dict[str, List],
                                     eligible_cols: Optional[List[int]] = None):
        """Strategy B: inject values outside extreme quantiles, reliably detectable by OD-Histogram.

        Magnitude: 0.3~1.0x IQR99 offset, past the quantile boundary without
        being overly extreme.
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # Categorical columns go through the typo path
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # skip numeric-column logic

                original_val = X_dirty[idx, col]

                if col in self.col_percentiles:
                    p1, p99 = self.col_percentiles[col]
                    prange = max(abs(p99 - p1), 1e-6)
                    # 0.3~1.0x IQR99 offset (past the quantile but not extreme)
                    offset = prange * np.random.uniform(0.3, 1.0)

                    if np.random.random() < 0.5:
                        new_val = p99 + offset  # above the 99th percentile
                    else:
                        new_val = p1 - offset   # below the 1st percentile
                else:
                    # Fall back to Gaussian when percentile info is missing
                    std = self.col_stds[col] if not np.isnan(self.col_stds[col]) else 1.0
                    new_val = original_val + np.random.choice([-1, 1]) * 3 * std

                noise = new_val - original_val
                X_dirty[idx, col] = new_val
                injected['syntactic'].append((idx, col, original_val, noise))
                used_indices.add((idx, col))

    def _inject_syntactic_pvd(self, X_dirty: np.ndarray, n: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List],
                               eligible_cols: Optional[List[int]] = None):
        """Strategy C: magnitude-anomaly simulation (character-level anomalies
        recreated in numeric space).

        - val * 10 (extra digit)
        - round(val) * 11 (double-digit repeats like 33, 55, 88)
        - -abs(val) (sign flip)
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # Categorical columns go through the typo path
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # skip numeric-column logic

                original_val = X_dirty[idx, col]

                strategy = np.random.choice(['mul10', 'double_digit', 'sign_flip'])
                if strategy == 'mul10':
                    new_val = original_val * 10
                elif strategy == 'double_digit':
                    base = max(1, abs(int(round(original_val))))
                    new_val = base * 11.0  # e.g., 3 -> 33, 5 -> 55
                else:  # sign_flip
                    new_val = -abs(original_val) if original_val > 0 else abs(original_val) + 1

                noise = new_val - original_val
                if abs(noise) > 1e-6:  # ensure the value actually changed
                    X_dirty[idx, col] = new_val
                    injected['syntactic'].append((idx, col, original_val, noise))
                    used_indices.add((idx, col))

    # ====================================================================
    # 4. Label-error injection (conditional; mirrors the detected pattern)
    # ====================================================================

    def _inject_label_noise(self, y_dirty: np.ndarray, n_label: int,
                             label_pattern: LabelErrorPattern,
                             injected: Dict[str, List],
                             exclude_rows: Optional[Set[int]] = None):
        """Inject label noise according to the detected label-error pattern
        (rule-aware).

        Conditional: this method is only called when the detector found label
        errors.
        - Classification: flip rule-covered rows first, then follow the
          flip_matrix.
        - Regression: add Gaussian noise with std = estimated noise_std.

        Args:
            exclude_rows: row indices to skip (rows already carrying feature
                errors).
        """
        n_samples = len(y_dirty)
        if n_label <= 0:
            return

        # Regression: Gaussian-noise injection
        if label_pattern.is_regression:
            valid_indices = [i for i in range(n_samples)
                             if not np.isnan(y_dirty[i])
                             and (exclude_rows is None or i not in exclude_rows)]
            if not valid_indices:
                return
            np.random.shuffle(valid_indices)
            count = 0
            noise_std = label_pattern.noise_std if label_pattern.noise_std > 0 else label_pattern.label_std * 0.2
            for idx in valid_indices:
                if count >= n_label:
                    break
                original_val = y_dirty[idx]
                # Gaussian noise
                noise = np.random.normal(0, noise_std)
                new_val = original_val + noise
                if abs(noise) > 1e-8:
                    y_dirty[idx] = new_val
                    injected['label_noise'].append((idx, -1, original_val, new_val))
                    count += 1
            return

        # Classification: rule-aware flipping
        if not label_pattern.unique_classes:
            return

        valid_indices = [i for i in range(n_samples)
                         if not np.isnan(y_dirty[i])
                         and (exclude_rows is None or i not in exclude_rows)]
        if not valid_indices:
            return

        # Find rule-covered candidate rows (in encoded space)
        rule_aware = self._find_encoded_rule_aware_candidates(y_dirty, valid_indices)

        # Flip rule-covered rows first
        priority_order = rule_aware + [
            i for i in valid_indices if i not in set(rule_aware)]
        np.random.shuffle(valid_indices)  # randomize the non-rule tail

        count = 0
        for idx in priority_order:
            if count >= n_label:
                break

            current_label = y_dirty[idx]
            new_label = self._sample_flip(current_label, label_pattern)

            if new_label is not None and new_label != current_label:
                original_val = y_dirty[idx]
                y_dirty[idx] = new_label
                injected['label_noise'].append((idx, -1, original_val, new_label))
                count += 1

    def _find_encoded_rule_aware_candidates(
        self,
        y: np.ndarray,
        valid_indices: List[int],
    ) -> List[int]:
        """Find rule-aware label-flip candidates in encoded space.

        Walk CFD rules to find rows whose current label differs from the
        rule-expected label but whose feature conditions already hold. After
        flipping, such rows satisfy the full rule and are detectable.

        Checking every feature condition in encoded space is awkward
        (categorical columns would require inverse encoding), so only the
        label direction is checked for rule coverage, enough to ensure the
        flip direction is correct.

        Returns:
            Shuffled list of candidate row indices.
        """
        candidates = set()

        # Collect every label direction covered by some rule
        covered_directions = set()  # rule-expected label values (post-flip targets)
        for class_val_str in self._cfd_col_map.keys():
            try:
                covered_directions.add(float(class_val_str))
            except (ValueError, TypeError):
                continue

        # DC label rules
        label_names = {'class'}
        if self.label_col:
            label_names.add(self.label_col)

        for dc_rule in self._dc_rule_list:
            mark_cols = dc_rule.get('mark_cols', [])
            if not mark_cols:
                continue
            is_label_dc = any(mc in label_names for mc in mark_cols)
            if not is_label_dc:
                continue

            clauses = dc_rule.get('clauses', [])
            for clause in clauses:
                col = clause.get('col', '')
                if col in label_names and clause.get('op') == 'EQ':
                    val = clause.get('_encoded_value')
                    if val is not None:
                        covered_directions.add(float(val))
                    break

        if not covered_directions:
            return []

        # For each rule-covered direction, collect rows whose current label
        # differs from the target value (so after flipping they match).
        for idx in valid_indices:
            current_label = y[idx]
            for target_val in covered_directions:
                if abs(current_label - target_val) > 1e-6:
                    # current_label != target_val; after flipping it matches
                    # and a rule will cover it.
                    candidates.add(idx)
                    break

        result = list(candidates)
        np.random.shuffle(result)
        return result

    def _sample_flip(self, current_label: float,
                     pattern: LabelErrorPattern) -> Optional[float]:
        """Sample a target class according to the detected flip distribution."""
        if pattern.flip_matrix:
            # Gather flip targets starting from current_label
            targets = []
            weights = []
            for (from_cls, to_cls), cnt in pattern.flip_matrix.items():
                if abs(from_cls - current_label) < 1e-6:
                    targets.append(to_cls)
                    weights.append(cnt)

            if targets:
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                return np.random.choice(targets, p=weights)

        # No flip_matrix: flip to any other class at random
        others = [c for c in pattern.unique_classes if abs(c - current_label) > 1e-6]
        if others:
            return np.random.choice(others)
        return None

    # ====================================================================
    # Error-list construction
    # ====================================================================

    def build_error_list(self,
                         injected: Dict[str, List]) -> List[Dict]:
        """
        Convert injected errors into the format expected by the cleaning env.

        Args:
            injected: the error dict returned by inject_errors

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
        """
        error_list = []

        # Missing errors (type=0)
        for idx, col, original_val in injected.get('missing', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': original_val
            })

        # Semantic errors (type=1)
        for idx, col, original_val, new_val in injected.get('semantic', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': original_val
            })

        # Syntactic errors (type=2)
        for idx, col, original_val, noise in injected.get('syntactic', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': original_val
            })

        # Label noise (type=3, col=-1)
        for idx, col, original_val, new_val in injected.get('label_noise', []):
            error_list.append({
                'idx': idx,
                'col': -1,
                'type': 3,
                'repair_value': original_val
            })

        return error_list

    # ====================================================================
    # Statistics
    # ====================================================================

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        return {
            'n_samples': len(self.X_base),
            'n_features': self.X_base.shape[1],
            'col_means': self.col_means,
            'col_stds': self.col_stds,
            'n_fd_rules': len(self.fd_col_pairs),
            'has_rich_rules': bool(self.rich_rules and self.rich_rules.get('has_rich_rules')),
            'n_domain_rules': len(self._domain_col_map),
            'n_cfd_rules': sum(len(v) for v in self._cfd_col_map.values()),
            'n_dc_rules': len(self._dc_rule_list),
        }

    # ====================================================================
    # CSV-space injection (operate directly on DataFrame strings)
    # ====================================================================

    def inject_csv_space(
        self,
        clean_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        categorical_cols: Set[str],
        missing_rate: float = 0.05,
        semantic_rate: float = 0.1,
        syntactic_rate: float = 0.15,
        protected_cols: Optional[Set[str]] = None,
        label_pattern: Optional['LabelErrorPattern'] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """Inject errors directly into the raw CSV string space.

        Aligned with inject_errors() on encoded space, but operates on
        DataFrame strings to eliminate the float precision and format drift
        introduced by LE+SS encoding/decoding.

        Args:
            clean_df: clean CSV DataFrame (raw format)
            feature_cols: feature column names
            label_col: label column name
            categorical_cols: set of categorical column names
            missing_rate / semantic_rate / syntactic_rate: per-type injection rates
            protected_cols: protected columns (excluded from syntactic
                injection, e.g. high-frequency FD LHS columns)
            label_pattern: label-error pattern

        Returns:
            (dirty_df, injected), `injected` has the format:
            {
                'missing': [(row_idx, col_name, original_str), ...],
                'semantic': [(row_idx, col_name, original_str, new_str), ...],
                'syntactic': [(row_idx, col_name, original_str, new_str), ...],
                'label_noise': [(row_idx, label_col, original_str, new_str), ...],
            }
        """
        dirty_df = clean_df.copy()
        n_samples = len(dirty_df)
        protected_cols = protected_cols or set()

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        # Positions already injected: (row_idx, col_name)
        used: Set[Tuple[int, str]] = set()

        # 1. Missing-value injection
        n_missing = int(n_samples * missing_rate)
        self._csv_inject_missing(dirty_df, n_missing, feature_cols, used, injected)

        # 2. Semantic-error injection (FD/CFD violations)
        n_semantic_total = int(n_samples * semantic_rate)

        # Label-budget strategy (matches the encoded-space path)
        has_label_rules = self._has_cfd_for_label()
        n_label = 0
        label_from_semantic = False
        if has_label_rules and label_pattern is not None:
            n_label = max(1, int(n_semantic_total * 0.2))
            label_from_semantic = True
        n_semantic = n_semantic_total - n_label if label_from_semantic else n_semantic_total

        self._csv_inject_semantic(
            dirty_df, n_semantic, feature_cols, label_col,
            categorical_cols, used, injected)

        # 3. Syntactic-error injection
        n_syntactic = int(n_samples * syntactic_rate)
        self._csv_inject_syntactic(
            dirty_df, n_syntactic, feature_cols,
            categorical_cols, protected_cols, used, injected)

        # 4. Label-error injection
        if n_label > 0 and label_pattern is not None:
            self._csv_inject_label(dirty_df, n_label, label_col, label_pattern, injected)

        return dirty_df, injected

    def _csv_inject_missing(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV-space missing-value injection: set cells to empty strings."""
        n_samples = len(df)
        n_cols = len(feature_cols)
        for _ in range(n * 3):
            if len(injected['missing']) >= n:
                break
            idx = np.random.randint(0, n_samples)
            col_name = feature_cols[np.random.randint(0, n_cols)]
            if (idx, col_name) in used:
                continue
            original = str(df.at[df.index[idx], col_name])
            if original == '' or original.lower() in ('nan', 'none', ''):
                continue
            df.at[df.index[idx], col_name] = ''
            injected['missing'].append((idx, col_name, original))
            used.add((idx, col_name))

    def _csv_inject_semantic(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        label_col: str,
        categorical_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV-space semantic-error injection.

        Three sources tried in priority order:
          1. FD rules: swap RHS values across groups.
          2. DC abs_diff rules: modify one column so |col1-col2| > threshold.
          3. CFD feature rules: modify feature values so they drift from the
             in-class baseline.
        """
        count = 0

        # --- 1. FD-rule injection ---
        if self.fd_rules and self.column_names:
            count += self._csv_inject_semantic_fd(df, n, feature_cols, used, injected)

        # --- 2. DC abs_diff injection ---
        remaining = n - count
        if remaining > 0 and self.rich_rules and self.rich_rules.get('dc_rules'):
            count += self._csv_inject_semantic_dc(
                df, remaining, feature_cols, used, injected)

        return count

    def _csv_inject_semantic_fd(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ) -> int:
        """FD rules: swap RHS string values across groups."""
        count = 0
        if not self.fd_rules:
            return 0

        per_rule_budget = max(1, n // max(len(self.fd_rules), 1))

        for lhs_str, rhs_str in self.fd_rules:
            if count >= n:
                break
            lhs_cols = [c.strip() for c in str(lhs_str).split(',')]
            rhs_col = str(rhs_str).strip()

            if rhs_col not in feature_cols:
                continue
            if not all(c in df.columns for c in lhs_cols):
                continue
            if rhs_col not in df.columns:
                continue

            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(df)):
                if (i, rhs_col) in used:
                    continue
                lhs_vals = tuple(str(df.at[df.index[i], c]) for c in lhs_cols)
                if any(v == '' or v.lower() in ('nan', 'none') for v in lhs_vals):
                    continue
                rhs_val = str(df.at[df.index[i], rhs_col])
                if rhs_val == '' or rhs_val.lower() in ('nan', 'none'):
                    continue
                groups[lhs_vals].append(i)

            group_keys = [k for k in groups if len(groups[k]) >= 3]
            if len(group_keys) < 2:
                continue

            rule_injected = 0
            np.random.shuffle(group_keys)

            for gi, gk in enumerate(group_keys):
                if rule_injected >= per_rule_budget or count >= n:
                    break
                rows = groups[gk]
                max_per_group = max(1, (len(rows) - 1) // 2)

                other_key = group_keys[(gi + 1) % len(group_keys)]
                other_rows = groups[other_key]
                donor_row = np.random.choice(other_rows)
                donor_val = str(df.at[df.index[donor_row], rhs_col])

                group_injected = 0
                for row_idx in rows:
                    if (rule_injected >= per_rule_budget
                            or count >= n
                            or group_injected >= max_per_group):
                        break
                    if (row_idx, rhs_col) in used:
                        continue
                    original = str(df.at[df.index[row_idx], rhs_col])
                    if original != donor_val:
                        df.at[df.index[row_idx], rhs_col] = donor_val
                        injected['semantic'].append((row_idx, rhs_col, original, donor_val))
                        used.add((row_idx, rhs_col))
                        rule_injected += 1
                        count += 1
                        group_injected += 1

        return count

    def _csv_inject_semantic_dc(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ) -> int:
        """Inject DC abs_diff semantic violations in CSV space.

        For GT(ABS(t1.col1 - t1.col2), threshold) rules:
          - Find legal rows with |col1 - col2| <= threshold.
          - Change col1 or col2 so the difference exceeds the threshold.
          - Respect DOMAIN constraints (e.g. INT [1, 10]).
        """
        dc_rules = self.rich_rules.get('dc_rules', [])
        if not dc_rules:
            return 0

        # Pre-build DOMAIN bounds (col_name -> (min, max)).
        # domain_rules inside rich_rules are dicts produced by rules_to_dict.
        domain_bounds: Dict[str, Tuple[float, float]] = {}
        if self.rich_rules.get('domain_rules'):
            for dr in self.rich_rules['domain_rules']:
                col_name = dr['column'] if isinstance(dr, dict) else dr.column
                min_v = dr.get('min_val') if isinstance(dr, dict) else getattr(dr, 'min_val', None)
                max_v = dr.get('max_val') if isinstance(dr, dict) else getattr(dr, 'max_val', None)
                if min_v is not None and max_v is not None:
                    domain_bounds[col_name] = (min_v, max_v)

        count = 0
        per_rule_budget = max(1, n // max(len(dc_rules), 1))

        for dc_rule in dc_rules:
            if count >= n:
                break

            # dc_rules inside rich_rules are dicts produced by rules_to_dict
            clauses = dc_rule['clauses'] if isinstance(dc_rule, dict) else dc_rule.clauses
            # Only handle pure abs_diff rules without MARK
            if len(clauses) != 1 or clauses[0].get('type') != 'abs_diff':
                continue
            mark_cols = dc_rule.get('mark_cols', []) if isinstance(dc_rule, dict) else dc_rule.mark_cols
            if mark_cols:
                continue

            clause = clauses[0]
            col1, col2 = clause['col1'], clause['col2']
            threshold = clause['value']

            if col1 not in df.columns or col2 not in df.columns:
                continue

            # Collect currently legal rows
            candidates = []
            for i in range(len(df)):
                if (i, col1) in used or (i, col2) in used:
                    continue
                try:
                    v1 = float(str(df.at[df.index[i], col1]).strip())
                    v2 = float(str(df.at[df.index[i], col2]).strip())
                except (ValueError, TypeError):
                    continue
                if abs(v1 - v2) <= threshold:
                    candidates.append((i, v1, v2))

            np.random.shuffle(candidates)
            rule_count = 0

            for idx, v1, v2 in candidates:
                if rule_count >= per_rule_budget or count >= n:
                    break

                # Pick col1 or col2 at random to modify
                target_col = col1 if np.random.random() < 0.5 else col2
                other_val = v2 if target_col == col1 else v1

                # New value: make |new - other| > threshold
                delta = threshold * np.random.uniform(0.2, 0.8)
                if np.random.random() < 0.5:
                    new_val = other_val + threshold + delta
                else:
                    new_val = other_val - threshold - delta

                # DOMAIN clamping
                if target_col in domain_bounds:
                    lo, hi = domain_bounds[target_col]
                    new_val = max(lo, min(hi, new_val))
                    # Does the clamped value still violate the rule?
                    if abs(new_val - other_val) <= threshold:
                        # Try the other direction
                        new_val = other_val + threshold + delta
                        new_val = max(lo, min(hi, new_val))
                        if abs(new_val - other_val) <= threshold:
                            new_val = other_val - threshold - delta
                            new_val = max(lo, min(hi, new_val))
                            if abs(new_val - other_val) <= threshold:
                                continue  # cannot produce a DOMAIN-valid violation

                # Detect integer-typed columns via the original string format
                original_str = str(df.at[df.index[idx], target_col]).strip()
                try:
                    int(original_str)
                    new_val = int(round(new_val))
                except (ValueError, TypeError):
                    new_val = round(new_val, 6)

                new_str = str(new_val)
                if new_str != original_str:
                    df.at[df.index[idx], target_col] = new_str
                    injected['semantic'].append((idx, target_col, original_str, new_str))
                    used.add((idx, target_col))
                    rule_count += 1
                    count += 1

        return count

    def _csv_inject_syntactic(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        categorical_cols: Set[str],
        protected_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV-space syntactic-error injection.

        Numeric columns: 3-5 sigma deviation / x10 / sign flip.
        Categorical columns: generate_typo().
        DOMAIN violations: values outside the rule-defined range.
        """
        n_samples = len(df)
        eligible_cols = [c for c in feature_cols if c not in protected_cols]
        if not eligible_cols:
            eligible_cols = list(feature_cols)

        # Gather per-column CSV-space statistics (numeric columns)
        col_stats: Dict[str, Dict] = {}
        for col_name in eligible_cols:
            if col_name in categorical_cols:
                continue
            vals = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(vals) > 1:
                col_stats[col_name] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'p1': float(vals.quantile(0.01)),
                    'p99': float(vals.quantile(0.99)),
                }

        # DOMAIN violations (30% of the budget)
        n_domain = 0
        domain_cols_csv = {}  # col_name -> domain_rule
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            for rule in self.rich_rules.get('domain_rules', []):
                col_name = rule.get('column', '')
                if col_name in eligible_cols:
                    domain_cols_csv[col_name] = rule
            if domain_cols_csv:
                n_domain = int(n * 0.3)
                self._csv_inject_domain(df, n_domain, domain_cols_csv, categorical_cols,
                                        used, injected)

        # Statistical anomalies (remaining budget)
        n_stat = n - n_domain
        count = 0

        # Per-strategy allocation: 40% gaussian, 30% histogram, 30% pvd
        n_gaussian = int(n_stat * 0.4)
        n_histogram = int(n_stat * 0.3)
        n_pvd = n_stat - n_gaussian - n_histogram

        for sub_n, strategy in [(n_gaussian, 'gaussian'),
                                 (n_histogram, 'histogram'),
                                 (n_pvd, 'pvd')]:
            for _ in range(sub_n * 3):
                if count >= n_stat:
                    break
                idx = np.random.randint(0, n_samples)
                col_name = eligible_cols[np.random.randint(0, len(eligible_cols))]
                if (idx, col_name) in used:
                    continue
                original = str(df.at[df.index[idx], col_name])
                if original == '' or original.lower() in ('nan', 'none'):
                    continue

                # Categorical columns: format-anomaly injection for short
                # strings, or typo for long strings.
                if col_name in categorical_cols:
                    new_val = self._generate_csv_categorical_anomaly(original, col_name)
                    if new_val != original:
                        df.at[df.index[idx], col_name] = new_val
                        injected['syntactic'].append((idx, col_name, original, new_val))
                        used.add((idx, col_name))
                        count += 1
                    continue

                # Numeric columns
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue

                stats = col_stats.get(col_name)
                if stats is None:
                    continue

                new_float = None

                if strategy == 'gaussian':
                    std = stats['std'] if stats['std'] > 1e-10 else 1.0
                    sigma_mult = np.random.uniform(3.0, 5.0)
                    direction = np.random.choice([-1, 1])
                    new_float = stats['mean'] + direction * sigma_mult * std

                elif strategy == 'histogram':
                    p1, p99 = stats['p1'], stats['p99']
                    prange = max(abs(p99 - p1), 1e-6)
                    offset = prange * np.random.uniform(0.3, 1.0)
                    if np.random.random() < 0.5:
                        new_float = p99 + offset
                    else:
                        new_float = p1 - offset

                elif strategy == 'pvd':
                    pvd_strat = np.random.choice(['mul10', 'double_digit', 'sign_flip'])
                    if pvd_strat == 'mul10':
                        new_float = orig_float * 10
                    elif pvd_strat == 'double_digit':
                        base = max(1, abs(int(round(orig_float))))
                        new_float = float(base * 11)
                    else:
                        new_float = -abs(orig_float) if orig_float > 0 else abs(orig_float) + 1

                if new_float is not None and abs(new_float - orig_float) > 1e-6:
                    # Preserve the column's integer / float format
                    if '.' not in original and original.lstrip('-').isdigit():
                        new_str = str(int(round(new_float)))
                    else:
                        new_str = f"{new_float:.6g}"
                    df.at[df.index[idx], col_name] = new_str
                    injected['syntactic'].append((idx, col_name, original, new_str))
                    used.add((idx, col_name))
                    count += 1

    def _generate_csv_categorical_anomaly(
        self, original: str, col_name: str
    ) -> str:
        """Generate a format-anomaly value for a categorical column (CSV space).

        Short strings (len<=2): use format-anomaly injection so both RAHA and
        DOMAIN can detect it:
          - digit injection: "a" -> "a1", "az" -> "a2z"
          - special character: "a" -> "a_", "az" -> "a-z"
          - repeated character: "a" -> "aa", "az" -> "azz"

        Long strings (len>=3): standard generate_typo(), verifying the result
        is not in the ENUM. If the typo still produces a legal value, fall
        back to format-anomaly injection.

        Args:
            original: raw string value
            col_name: column name (used to look up the DOMAIN ENUM list)

        Returns:
            Anomalous string (guaranteed to differ from original).
        """
        # Fetch the column's valid ENUM values (used to confirm the injected
        # value really is out of range).
        enum_vals = set()
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            for rule in self.rich_rules.get('domain_rules', []):
                if rule.get('column') == col_name and rule.get('dtype') == 'ENUM':
                    enum_vals = set(str(v) for v in rule.get('enum_vals', []))
                    break

        # Short-string strategy: direct format-anomaly injection
        if len(original) <= 2:
            return self._format_anomaly(original, enum_vals)

        # Long-string strategy: try generate_typo first
        new_val = generate_typo(original)
        # Validate: the typo must not be in the ENUM (otherwise DOMAIN cannot detect it)
        if new_val != original and (not enum_vals or new_val not in enum_vals):
            return new_val

        # Fallback: format-anomaly injection
        return self._format_anomaly(original, enum_vals)

    @staticmethod
    def _format_anomaly(original: str, enum_vals: set) -> str:
        """Generate a format-anomaly value.

        Pick a strategy at random, ensuring the result is not in the ENUM:
          1. digit: insert a digit at a random position.
          2. special character: append an underscore / hyphen.
          3. repeat: repeat the final character.
          4. space: insert a space somewhere in the middle.

        Args:
            original: original string
            enum_vals: set of valid ENUM values

        Returns:
            An anomalous value (guaranteed != original, and preferably not in enum_vals).
        """
        import random as _rng

        strategies = ['digit', 'special', 'repeat']
        if len(original) >= 2:
            strategies.append('space')

        _rng.shuffle(strategies)

        for strategy in strategies:
            if strategy == 'digit':
                # Append a digit at the end
                digit = str(_rng.randint(1, 9))
                candidate = original + digit
            elif strategy == 'special':
                # Append an underscore
                candidate = original + '_'
            elif strategy == 'repeat':
                # Repeat the last character
                candidate = original + original[-1]
            else:  # space
                # Insert a space in the middle
                pos = _rng.randint(1, len(original) - 1)
                candidate = original[:pos] + ' ' + original[pos:]

            if candidate != original and candidate not in enum_vals:
                return candidate

        # Final fallback: original value + digit suffix
        return original + '1'

    def _csv_inject_domain(
        self,
        df: pd.DataFrame,
        n: int,
        domain_cols: Dict[str, Dict],
        categorical_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV-space DOMAIN-violation injection (values outside the rule-defined range)."""
        n_samples = len(df)
        col_names = list(domain_cols.keys())
        count = 0

        for _ in range(n * 3):
            if count >= n:
                break
            col_name = np.random.choice(col_names)
            idx = np.random.randint(0, n_samples)
            if (idx, col_name) in used:
                continue
            original = str(df.at[df.index[idx], col_name])
            if original == '' or original.lower() in ('nan', 'none'):
                continue

            rule = domain_cols[col_name]
            new_str = None

            if rule.get('dtype') == 'ENUM':
                enum_vals = rule.get('enum_vals', [])
                if enum_vals and col_name in categorical_cols:
                    # Generate a typo that is not in the enum
                    new_str = generate_typo(original)
                    if new_str in enum_vals:
                        new_str = original + '_invalid'
            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                min_v = rule['min_val']
                max_v = rule['max_val']
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue
                if np.random.random() < 0.5:
                    new_float = max_v + np.random.randint(1, 6)
                else:
                    new_float = min_v - np.random.randint(1, 6)
                # Preserve integer / float format
                if '.' not in original and original.lstrip('-').isdigit():
                    new_str = str(int(round(new_float)))
                else:
                    new_str = f"{new_float:.6g}"

            if new_str is not None and new_str != original:
                df.at[df.index[idx], col_name] = new_str
                injected['syntactic'].append((idx, col_name, original, new_str))
                used.add((idx, col_name))
                count += 1

    def _csv_inject_label(
        self,
        df: pd.DataFrame,
        n: int,
        label_col: str,
        label_pattern: 'LabelErrorPattern',
        injected: Dict[str, List],
    ):
        """CSV-space label-error injection (rule-aware).

        Classification: flip rows that satisfy CFD/DC label-rule conditions first
            (so the rule detects the flip post hoc).
        Regression: add Gaussian noise (unchanged).
        """
        if label_col not in df.columns:
            return

        n_samples = len(df)
        valid_indices = [i for i in range(n_samples)
                         if str(df.at[df.index[i], label_col]).strip() not in
                         ('', 'nan', 'none', 'None', 'NaN')]
        if not valid_indices:
            return

        np.random.shuffle(valid_indices)
        count = 0

        if label_pattern.is_regression:
            noise_std = label_pattern.noise_std if label_pattern.noise_std > 0 else label_pattern.label_std * 0.2
            for idx in valid_indices:
                if count >= n:
                    break
                original = str(df.at[df.index[idx], label_col])
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue
                noise = np.random.normal(0, noise_std)
                new_float = orig_float + noise
                if abs(noise) > 1e-8:
                    if '.' not in original and original.lstrip('-').isdigit():
                        new_str = str(int(round(new_float)))
                    else:
                        new_str = f"{new_float:.6g}"
                    df.at[df.index[idx], label_col] = new_str
                    injected['label_noise'].append((idx, label_col, original, new_str))
                    count += 1
        else:
            # Classification: rule-aware label injection
            label_values = [str(df.at[df.index[i], label_col])
                            for i in valid_indices]
            unique_labels = list(set(label_values))
            if len(unique_labels) < 2:
                return

            # Collect rule-covered rows (post-flip, at least one rule detects them)
            rule_aware_indices = self._find_rule_aware_label_candidates(
                df, label_col, valid_indices, unique_labels)

            # Flip rule-covered rows first, then randomly flip the rest
            priority_order = rule_aware_indices + [
                i for i in valid_indices if i not in set(rule_aware_indices)]

            for idx in priority_order:
                if count >= n:
                    break
                original = str(df.at[df.index[idx], label_col])
                others = [l for l in unique_labels if l != original]
                if not others:
                    continue
                new_label = np.random.choice(others)
                df.at[df.index[idx], label_col] = new_label
                injected['label_noise'].append((idx, label_col, original, new_label))
                count += 1

    def _find_rule_aware_label_candidates(
        self,
        df: pd.DataFrame,
        label_col: str,
        valid_indices: List[int],
        unique_labels: List[str],
    ) -> List[int]:
        """Find rows whose label flip will trigger at least one CFD/DC label rule.

        For each CFD label rule (conditions include the label column):
          - Find rows satisfying the non-label conditions.
          - After flipping the label the row meets every condition, and the
            rule detects it.

        Example:
          rule: income=1, capital_gain>=5000 => income EXCESS >= 1
          current row: income=0 (high income), capital_gain=8000
          -> after flipping income=1 the conditions hold and the rule fires.

        Returns:
            Shuffled list of candidate row indices covered by rules.
        """
        candidates = set()

        if not self.rich_rules or not self.rich_rules.get('has_rich_rules'):
            return []

        # Walk CFD rules
        for rule in self.rich_rules.get('cfd_rules', []):
            conditions = rule.get('conditions', [])
            # Split conditions into label vs. non-label
            label_conds = []
            feature_conds = []
            for col, op, val in conditions:
                if col == label_col or col == 'class':
                    label_conds.append((col, op, val))
                else:
                    feature_conds.append((col, op, val))

            if not label_conds:
                continue  # rules that do not touch the label column are skipped

            # Identify the expected label value
            rule_label_val = None
            for col, op, val in label_conds:
                if op == '=':
                    rule_label_val = val
                    break

            if rule_label_val is None:
                continue

            # Find rows whose current label != rule value (so it equals the
            # rule value after flipping) and whose feature conditions hold.
            for idx in valid_indices:
                current_label = str(df.at[df.index[idx], label_col])
                if current_label == rule_label_val:
                    continue  # already satisfies the label condition; flipping would break coverage

                # Check non-label feature conditions
                if self._csv_check_feature_conditions(df, idx, feature_conds):
                    candidates.add(idx)

        # DC label rules (MARK column is the label column)
        for dc_rule in self.rich_rules.get('dc_rules', []):
            mark_cols = dc_rule.get('mark_cols', [])
            if label_col not in mark_cols and 'class' not in mark_cols:
                continue

            clauses = dc_rule.get('clauses', [])
            # Find the label EQ condition
            label_eq_val = None
            non_label_clauses = []
            for clause in clauses:
                col = clause.get('col', '')
                if (col == label_col or col == 'class') and clause.get('op') == 'EQ':
                    label_eq_val = clause.get('value')
                elif col not in mark_cols:
                    non_label_clauses.append(clause)

            if label_eq_val is None:
                continue

            label_eq_str = str(int(label_eq_val)) if isinstance(label_eq_val, (int, float)) else str(label_eq_val)

            for idx in valid_indices:
                current_label = str(df.at[df.index[idx], label_col])
                if current_label == label_eq_str:
                    continue

                # Check non-label clauses
                all_ok = True
                for clause in non_label_clauses:
                    col = clause.get('col', '')
                    if col not in df.columns:
                        all_ok = False
                        break
                    try:
                        cell_val = float(df.at[df.index[idx], col])
                    except (ValueError, TypeError):
                        all_ok = False
                        break
                    op = clause.get('op', 'EQ')
                    val = clause.get('value', 0)
                    if not self._eval_comparison(cell_val, op, float(val)):
                        all_ok = False
                        break

                if all_ok:
                    candidates.add(idx)

        result = list(candidates)
        np.random.shuffle(result)
        return result

    @staticmethod
    def _csv_check_feature_conditions(
        df: pd.DataFrame, idx: int, conditions: List[Tuple]
    ) -> bool:
        """Check whether a row satisfies every feature condition (CSV space).

        Args:
            df: DataFrame
            idx: row index
            conditions: [(col, op, val), ...]

        Returns:
            True when every condition holds.
        """
        for col, op, val in conditions:
            if col == 'n_anomaly':
                continue  # n_anomaly is a dynamically computed pseudo-column; skip
            if col not in df.columns:
                return False
            try:
                cell_val = float(df.at[df.index[idx], col])
                threshold = float(val)
            except (ValueError, TypeError):
                # Fall back to string comparison
                cell_str = str(df.at[df.index[idx], col])
                if op == '=':
                    if cell_str != val:
                        return False
                elif op == '!=':
                    if cell_str == val:
                        return False
                continue

            if op == '=' and abs(cell_val - threshold) > 1e-6:
                return False
            elif op == '!=' and abs(cell_val - threshold) < 1e-6:
                return False
            elif op == '<=' and cell_val > threshold + 1e-6:
                return False
            elif op == '>=' and cell_val < threshold - 1e-6:
                return False
            elif op == '<' and cell_val >= threshold:
                return False
            elif op == '>' and cell_val <= threshold:
                return False

        return True
