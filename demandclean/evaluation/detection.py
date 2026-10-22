"""
Detection precision, recall and F1
==================================

Compares the cells reported by the detector with the cells where the dirty and the clean
training split differ (P_d, R_d and F1 of the paper).
"""

from typing import Any, Dict, List, Optional

import numpy as np


def evaluate_detector_accuracy(
    detected_errors: Dict[str, List],
    X_dirty: np.ndarray,
    X_clean: np.ndarray,
    verbose: bool = True,
    y_dirty: Optional[np.ndarray] = None,
    y_clean: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Precision, recall and F1 of the detector, overall and per error type.

    Ground truth: a cell is an error when the dirty value is NaN and the clean value is not
    (missing), or when both are present and differ (value error). A label is an error when
    y_dirty and y_clean differ.

    Args:
        detected_errors: detector output
            {'missing': [(idx, col, ...), ...], 'syntactic': [...], 'semantic': [...],
             'label_noise': [(idx, -1, ...), ...]}
        X_dirty: dirty feature matrix, shape (n_samples, n_features)
        X_clean: clean feature matrix of the same shape
        verbose: print the result
        y_dirty, y_clean: dirty and clean labels (label errors are evaluated when both are given)

    Returns:
        {'overall', 'missing', 'syntactic', 'semantic', 'value_error_combined', 'label_noise':
         {'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'gt_count', 'det_count'},
         'ground_truth_total', 'detected_total'}
    """
    assert X_dirty.shape == X_clean.shape, (
        f"X_dirty and X_clean differ in shape: {X_dirty.shape} vs {X_clean.shape}"
    )

    n_rows, n_cols = X_dirty.shape

    gt_missing: set = set()
    gt_value_error: set = set()

    for i in range(n_rows):
        for j in range(n_cols):
            d_val = X_dirty[i, j]
            c_val = X_clean[i, j]

            d_is_nan = (np.isnan(d_val) if isinstance(d_val, float) else False)
            c_is_nan = (np.isnan(c_val) if isinstance(c_val, float) else False)

            if d_is_nan and not c_is_nan:
                gt_missing.add((i, j))
            elif not d_is_nan and not c_is_nan:
                try:
                    if abs(float(d_val) - float(c_val)) > 1e-9:
                        gt_value_error.add((i, j))
                except (TypeError, ValueError):
                    if str(d_val).strip() != str(c_val).strip():
                        gt_value_error.add((i, j))

    gt_all = gt_missing | gt_value_error

    gt_label_noise: set = set()
    if y_dirty is not None and y_clean is not None:
        for i in range(len(y_dirty)):
            try:
                if abs(float(y_dirty[i]) - float(y_clean[i])) > 1e-9:
                    gt_label_noise.add((i, -1))
            except (TypeError, ValueError):
                if str(y_dirty[i]).strip() != str(y_clean[i]).strip():
                    gt_label_noise.add((i, -1))

    gt_all_with_label = gt_all | gt_label_noise

    det_missing: set = set()
    det_syntactic: set = set()
    det_semantic: set = set()
    det_label_noise: set = set()

    for item in detected_errors.get('missing', []):
        det_missing.add((item[0], item[1]))
    for item in detected_errors.get('syntactic', []):
        det_syntactic.add((item[0], item[1]))
    for item in detected_errors.get('semantic', []):
        det_semantic.add((item[0], item[1]))
    for item in detected_errors.get('label_noise', []):
        det_label_noise.add((item[0], item[1]))

    det_all = det_missing | det_syntactic | det_semantic
    det_all_with_label = det_all | det_label_noise

    def _calc_prf(gt_set: set, det_set: set) -> Dict[str, Any]:
        tp = len(gt_set & det_set)
        fp = len(det_set - gt_set)
        fn = len(gt_set - det_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': len(gt_set),
            'det_count': len(det_set),
        }

    # overall includes label errors
    overall = _calc_prf(gt_all_with_label, det_all_with_label)
    missing_eval = _calc_prf(gt_missing, det_missing)
    # The ground truth does not separate syntactic from semantic errors, so both are
    # compared with the union of value errors
    syntactic_eval = _calc_prf(gt_value_error, det_syntactic)
    semantic_eval = _calc_prf(gt_value_error, det_semantic)
    value_combined_eval = _calc_prf(gt_value_error, det_syntactic | det_semantic)
    label_noise_eval = _calc_prf(gt_label_noise, det_label_noise)

    result = {
        'overall': overall,
        'missing': missing_eval,
        'syntactic': syntactic_eval,
        'semantic': semantic_eval,
        'value_error_combined': value_combined_eval,
        'label_noise': label_noise_eval,
        'ground_truth_total': len(gt_all_with_label),
        'detected_total': len(det_all_with_label),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Detection accuracy")
        print("=" * 60)
        print(f"Ground-truth errors: {len(gt_all)} (missing: {len(gt_missing)}, "
              f"value errors: {len(gt_value_error)})")
        print(f"Detected: {len(det_all)} (missing: {len(det_missing)}, "
              f"syntactic: {len(det_syntactic)}, semantic: {len(det_semantic)})")
        for category in ('overall', 'missing', 'syntactic', 'semantic',
                         'value_error_combined', 'label_noise'):
            m = result[category]
            print(f"  [{category}] P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")

    return result
