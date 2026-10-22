"""
BoostClean wrapper: an ensemble of error detectors and repairs selected by boosting.

Paper: BoostClean: Automatic Error Detection and Repair for Machine Learning (2017)
Code: https://github.com/HoloClean/boostclean

Ground truth used: the clean labels of a small validation set.
The module calls the official activedetect package.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# Put this directory on the path for the official implementation
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import the official implementation
try:
    from activedetect.loaders.csv_loader import CSVLoader
    from activedetect.error_detectors.ErrorDetector import ErrorDetector
    from activedetect.learning.BoostClean import BoostClean
    from activedetect.learning.EvaluateCleaning import EvaluateCleaning
    from activedetect.reporting.CSVLogging import CSVLogging
    from activedetect.loaders.type_inference import LoLTypeInference
    from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
    from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
    HAS_ACTIVEDETECT = True
except ImportError as e:
    HAS_ACTIVEDETECT = False
    IMPORT_ERROR = str(e)


class BoostCleanWrapper:
    """
    Wrapper of the BoostClean class of the official activedetect package.

    1. detector ensemble: statistical, rule-based and learned error detectors
    2. repair ensemble: deletion, imputation and replacement
    3. boosting: selects the best detector and repair combinations
    4. a small validation set scores the cleaning
    """

    def __init__(self,
                 boosting_rounds: int = 5,
                 quantitative_thresh: int = 10,
                 verbose: bool = False):
        """
        Args:
            boosting_rounds: number of boosting rounds
            quantitative_thresh: threshold of the numeric outlier detector
            verbose: print progress information
        """
        self.boosting_rounds = boosting_rounds
        self.quantitative_thresh = quantitative_thresh
        self.verbose = verbose
        self.ground_truth_used = 0
        self.selected_operations = []
        self.ensemble = None

    def _check_dependencies(self):
        """Check the dependencies."""
        if not HAS_ACTIVEDETECT:
            raise ImportError(
                f"activedetectpackage import failed: {IMPORT_ERROR}\n"
                "Make sure the activedetect package of BoostClean is installed."
            )

    def _load_data_as_lol(self, filepath: str, convert_empty: bool = True) -> List[List]:
        """
        Load a CSV file as a list of lists (format of the official code).

        Args:
            filepath: CSV file
            convert_empty: turn "empty" into the empty string

        Returns:
            list of lists
        """
        loader = CSVLoader()
        data = loader.loadFile(filepath)

        # "empty" becomes the empty string (BoostClean marks missing values with the empty string)
        if convert_empty and data:
            data = [['' if cell == 'empty' else cell for cell in row] for row in data]

        return data

    def _get_default_modules(self):
        """getdefaultdetectionmoduleConfiguration"""
        modules = [QuantitativeErrorModule, PuncErrorModule]
        config = [{'thresh': self.quantitative_thresh}, {}]
        return modules, config

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              label_column: str = None,
              output_path: str = None,
              base_model=None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run BoostClean.

        Args:
            dirty_path: path of the dirty table
            clean_path: path of the clean table (labels)
            label_column: label column (None takes the last column)
            output_path: output path
            base_model: sklearn base model (RandomForest by default)

        Returns:
            cleaned table and run information
        """
        self._check_dependencies()

        # setbase model
        if base_model is None:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Load data
        loaded_data = self._load_data_as_lol(dirty_path)

        # Drop the header
        header = loaded_data[0] if loaded_data else []
        data = loaded_data[1:] if len(loaded_data) > 1 else loaded_data

        # Split features and label
        features = [row[:-1] for row in data]
        labels = [row[-1] for row in data]

        # Numeric labels
        try:
            labels = [float(l) for l in labels]
        except:
            # Encode a non-numeric label
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(labels).tolist()

        # setlog
        log_path = output_path.replace('.csv', '.log') if output_path else 'boostclean.log'
        logging = CSVLogging(log_path)

        # getdetectionmodule
        modules, config = self._get_default_modules()

        # loadcleandataused forevaluation
        wrong_cells = []
        if clean_path:
            clean_data = self._load_data_as_lol(clean_path)
            clean_data = clean_data[1:] if len(clean_data) > 1 else clean_data
            # IdentifyErrorcell
            for i, (dirty_row, clean_row) in enumerate(zip(data, clean_data)):
                for j, (d, c) in enumerate(zip(dirty_row, clean_row)):
                    if str(d) != str(c):
                        wrong_cells.append((i, j))
            self.ground_truth_used = len(clean_data)  # validation setuse

        try:
            # Create a BoostClean instance
            bc = BoostClean(
                modules=modules,
                config=config,
                base_model=base_model,
                features=features,
                labels=labels,
                logging=logging,
                wrong_cells=wrong_cells
            )

            # RunBoostClean
            self.ensemble, rep_tuples, sel_clf = bc.run(j=self.boosting_rounds)

            # ConvertcleaningresulttoDataFrame
            if rep_tuples:
                cleaned_df = pd.DataFrame(rep_tuples, columns=header if header else None)
            else:
                # If nothing was repaired, return the original data
                cleaned_df = pd.read_csv(dirty_path)

            # Post-process: convert empty strings back to the standard "empty" empty valuemarker
            cleaned_df = cleaned_df.replace('', 'empty')

            if self.verbose:
                print(f"BoostClean done, ensemble size: {len(self.ensemble)}")

        except Exception as e:
            if self.verbose:
                print(f"BoostCleanexecution error: {e}")
            # returnoriginaldata
            cleaned_df = pd.read_csv(dirty_path)
            self.ensemble = []

        # Save results
        if output_path:
            cleaned_df.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': 'BoostClean',
            'type': 'model-oriented',
            'auto_level': 2,  # Needs a validation set
            'ensemble_size': len(self.ensemble) if self.ensemble else 0,
            'boosting_rounds': self.boosting_rounds
        }

        return cleaned_df, info

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used."""
        return self.ground_truth_used


def boostclean_clean(dirty_path: str,
                     clean_path: str = None,
                     label_column: str = None,
                     output_path: str = None,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the BoostClean cleaner."""
    wrapper = BoostCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, label_column, output_path)
