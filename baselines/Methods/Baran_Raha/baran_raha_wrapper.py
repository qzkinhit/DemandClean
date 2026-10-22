"""
Raha and Baran wrapper: error detection and correction from a small number of labeled tuples.

Raha (SIGMOD 2019) detects errors with an ensemble of detection strategies.
Baran (SIGMOD 2020) corrects them by active learning.

Papers:
- Raha: A Configuration-Free Error Detection System (SIGMOD 2019)
- Baran: Effective Error Correction via a Unified Context Representation (SIGMOD 2020)

Ground truth used: the labeled tuples (LABELING_BUDGET, 20 by default).
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# Put this directory on the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import the official implementation
try:
    import raha
    from raha.dataset import Dataset
    from detection import Detection
    from correction import Correction
    HAS_BARAN_RAHA = True
except ImportError as e:
    HAS_BARAN_RAHA = False
    IMPORT_ERROR = str(e)


class BaranRahaWrapper:
    """
    Wrapper of Raha (detection) and Baran (correction).

    Both learn from labeled tuples. The ground truth used is LABELING_BUDGET tuples
    (20 by default).
    """

    def __init__(self,
                 labeling_budget: int = 20,
                 classification_model: str = "ABC",
                 min_correction_candidate_probability: float = 0.0,
                 min_correction_occurrence: int = 2,
                 max_value_length: int = 50,
                 verbose: bool = True,
                 save_results: bool = False):
        """
        Args:
            labeling_budget: number of labeled tuples
            classification_model: classifier ["ABC", "DTC", "GBC", "GNB", "KNC", "SGDC", "SVC"]
            min_correction_candidate_probability: minimum probability of a correction candidate
            min_correction_occurrence: minimum occurrence of a correction
            max_value_length: maximum value length
            verbose: print progress information
            save_results: keep intermediate results
        """
        self.labeling_budget = labeling_budget
        self.classification_model = classification_model
        self.min_correction_candidate_probability = min_correction_candidate_probability
        self.min_correction_occurrence = min_correction_occurrence
        self.max_value_length = max_value_length
        self.verbose = True
        self.save_results = save_results

        # Ground truth used = labeling budget
        self.ground_truth_used = labeling_budget

    def _check_dependencies(self):
        """Check the dependencies."""
        if not HAS_BARAN_RAHA:
            raise ImportError(
                f"Baran/Rahamodule import failed: {IMPORT_ERROR}\n"
                "Make sure methods/Baran_Raha holds the complete official code "
                "and that its dependencies (raha, mwparserfromhell, py7zr) are installed"
            )

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              output_path: str = None,
              task_name: str = "baran_task",
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        Run Raha and Baran.

        Args:
            dirty_path: path of the dirty table
            clean_path: path of the clean table (labels of the sampled tuples, optional)
            output_path: output path
            task_name: task name
            index_attribute: name of the index column

        Returns:
            repaired table and run information
        """
        self._check_dependencies()

        import time
        import tempfile
        start_time = time.perf_counter()

        # "empty" becomes an empty value, which Baran handles through fillna("nan")
        temp_dirty_path = None
        temp_clean_path = None
        try:
            # Handle dirty data
            dirty_df_raw = pd.read_csv(dirty_path)
            dirty_df_raw = dirty_df_raw.replace('empty', np.nan)
            temp_dirty_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
            dirty_df_raw.to_csv(temp_dirty_path, index=False)

            # Clean data (if given)
            if clean_path:
                clean_df_raw = pd.read_csv(clean_path)
                clean_df_raw = clean_df_raw.replace('empty', np.nan)
                temp_clean_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
                clean_df_raw.to_csv(temp_clean_path, index=False)

            # Dataset dictionary (through a temporary file)
            dataset_dictionary = {
                "name": task_name,
                "path": temp_dirty_path,
            }
            if temp_clean_path:
                dataset_dictionary["clean_path"] = temp_clean_path
            # 1. error detection (Raha)
            if self.verbose:
                print("=" * 60)
                print("Stage1: Rahaerror detection")
                print("=" * 60)

            detector = Detection()
            detected_cells = detector.run(dataset_dictionary)
            detection_p, detection_r, detection_f = detector.d.get_data_cleaning_evaluation(detected_cells)[:3]

            if self.verbose:
                print(f"detected error cells: {len(detected_cells)}")
                print(f"detection Precision: {detection_p:.4f}")
                print(f"detection Recall: {detection_r:.4f}")
                print(f"detection F1: {detection_f:.4f}")

            # 2. error repair (Baran)
            if self.verbose:
                print("=" * 60)
                print("Stage2: Baranerror repair")
                print("=" * 60)

            corrector = Correction()
            corrector.LABELING_BUDGET = self.labeling_budget
            corrector.CLASSIFICATION_MODEL = self.classification_model
            corrector.MIN_CORRECTION_CANDIDATE_PROBABILITY = self.min_correction_candidate_probability
            corrector.MIN_CORRECTION_OCCURRENCE = self.min_correction_occurrence
            corrector.MAX_VALUE_LENGTH = self.max_value_length
            corrector.VERBOSE = self.verbose
            corrector.SAVE_RESULTS = self.save_results

            correction_dictionary = corrector.run(detector.d)

            if self.verbose:
                print(f"repaired cell count: {len(correction_dictionary)}")

            # 3. Apply the repairs to the raw dirty data, which keeps the original format
            repaired_df = pd.read_csv(dirty_path)
            for cell, value in correction_dictionary.items():
                repaired_df.iloc[cell[0], cell[1]] = value

            # Post-process: map Baran's "nan" empty marker to the standard "empty"
            repaired_df = repaired_df.replace('nan', 'empty')

            elapsed_time = time.perf_counter() - start_time

            # Save results
            if output_path:
                # Ensure the output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                repaired_df.to_csv(output_path, index=False)
                if self.verbose:
                    print(f"Repaired data saved: {output_path}")

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'Baran_Raha',
                'type': 'data-oriented',
                'auto_level': 3,  # iterative interactive
                'labeling_budget': self.labeling_budget,
                'detected_cells': len(detected_cells),
                'corrected_cells': len(correction_dictionary),
                'detection_precision': detection_p,
                'detection_recall': detection_r,
                'detection_f1': detection_f,
                'elapsed_time': elapsed_time
            }

            return repaired_df, info

        except Exception as e:
            raise RuntimeError(f"Baran/Rahacleaningfailure: {e}")

        finally:
            # Clean up temp files
            if temp_dirty_path and os.path.exists(temp_dirty_path):
                os.remove(temp_dirty_path)
            if temp_clean_path and os.path.exists(temp_clean_path):
                os.remove(temp_clean_path)

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used (= LABELING_BUDGET)."""
        return self.ground_truth_used


def baran_raha_clean(dirty_path: str,
                     clean_path: str = None,
                     output_path: str = None,
                     labeling_budget: int = 20,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the Raha and Baran cleaner."""
    wrapper = BaranRahaWrapper(labeling_budget=labeling_budget, **kwargs)
    return wrapper.clean(dirty_path, clean_path, output_path)
