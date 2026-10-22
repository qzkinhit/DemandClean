"""
HoloClean wrapper: data repair that combines integrity constraints, statistics and external
knowledge in a probabilistic model.

Paper: HoloClean: Holistic Data Repairs with Probabilistic Inference (VLDB 2017)
Code: https://github.com/HoloClean/holoclean

Ground truth used: none (fully automatic).
HoloClean needs a PostgreSQL database.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# Put this directory on the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import the official implementation
try:
    from holoclean import HoloClean, Session
    from detect import NullDetector, ViolationDetector
    from repair.featurize import (
        InitAttrFeaturizer, OccurAttrFeaturizer,
        FreqFeaturizer, ConstraintFeaturizer
    )
    HAS_HOLOCLEAN = True
except ImportError as e:
    HAS_HOLOCLEAN = False
    IMPORT_ERROR = str(e)


class HoloCleanWrapper:
    """
    Wrapper of HoloClean with the common cleaning interface.

    HoloClean needs a PostgreSQL database: install PostgreSQL, create a user and a database,
    and pass the connection through environment variables or arguments.
    """

    def __init__(self,
                 db_user: str = "holocleanuser",
                 db_pwd: str = "abcd1234",
                 db_host: str = "localhost",
                 db_name: str = "holo",
                 domain_thresh_1: float = 0.1,
                 domain_thresh_2: float = 0.0,
                 weak_label_thresh: float = 0.90,
                 max_domain: int = 1000000,
                 cor_strength: float = 0.05,
                 nb_cor_strength: float = 0.3,
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.001,
                 threads: int = 20,
                 batch_size: int = 1,
                 epochs: int = 20,
                 verbose: bool = False,
                 timeout: int = 60000):
        """
        Args:
            db_user: database user
            db_pwd: database password
            db_host: database host
            db_name: database name
            domain_thresh_1: domain threshold of the first stage
            domain_thresh_2: domain threshold of the second stage
            weak_label_thresh: weak-label threshold
            max_domain: maximum domain size
            cor_strength: correlation strength
            nb_cor_strength: neighbour correlation strength
            weight_decay: weight decay
            learning_rate: learning rate
            threads: number of threads
            batch_size: batch size
            epochs: training epochs
            verbose: print progress information
            timeout: timeout in milliseconds
        """
        self.db_user = db_user
        self.db_pwd = db_pwd
        self.db_host = db_host
        self.db_name = db_name
        self.domain_thresh_1 = domain_thresh_1
        self.domain_thresh_2 = domain_thresh_2
        self.weak_label_thresh = weak_label_thresh
        self.max_domain = max_domain
        self.cor_strength = cor_strength
        self.nb_cor_strength = nb_cor_strength
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.threads = threads
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.timeout = timeout

        self.hc = None
        self.session = None
        self.ground_truth_used = 0

    def _check_dependencies(self):
        """Check the dependencies."""
        if not HAS_HOLOCLEAN:
            raise ImportError(
                f"HoloCleanmodule import failed: {IMPORT_ERROR}\n"
                "Make sure methods/HoloClean holds the complete official code "
                "and that its dependencies (torch, psycopg2) are installed"
            )

    def setup(self) -> bool:
        """
        setHoloCleanenvironment

        Returns:
            whethersuccessinitialize
        """
        self._check_dependencies()

        try:
            self.hc = HoloClean(
                db_user=self.db_user,
                db_pwd=self.db_pwd,
                db_host=self.db_host,
                db_name=self.db_name,
                domain_thresh_1=self.domain_thresh_1,
                domain_thresh_2=self.domain_thresh_2,
                weak_label_thresh=self.weak_label_thresh,
                max_domain=self.max_domain,
                cor_strength=self.cor_strength,
                nb_cor_strength=self.nb_cor_strength,
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                threads=self.threads,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                timeout=self.timeout
            )
            self.session = self.hc.session
            return True
        except Exception as e:
            if self.verbose:
                print(f"HoloCleaninitialization failed: {e}")
            return False

    def clean(self,
              dirty_path: str,
              dc_path: str = None,
              output_path: str = None,
              dataset_name: str = "data") -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete cleaning.

        Args:
            dirty_path: path of the dirty table
            dc_path: constraint file (optional)
            output_path: output path (optional)
            dataset_name: dataset name

        Returns:
            repaired table and run information
        """
        self._check_dependencies()

        # initialize
        if not self.setup():
            raise RuntimeError("HoloClean initialization failed, check the database configuration")

        try:
            # Load data
            self.session.load_data(dataset_name, dirty_path)

            # Load the constraints (if given)
            if dc_path and os.path.exists(dc_path):
                self.session.load_dcs(dc_path)

            # detectionError
            detectors = [NullDetector()]
            if dc_path:
                detectors.append(ViolationDetector())
            self.session.detect_errors(detectors)

            # Set domain
            self.session.setup_domain()

            # feature extractor
            featurizers = [
                InitAttrFeaturizer(),
                OccurAttrFeaturizer(),
                FreqFeaturizer()
            ]
            if dc_path:
                featurizers.append(ConstraintFeaturizer())

            # repair
            self.session.repair_errors(featurizers)

            # getrepairafterdata
            repaired_df = self.session.ds.get_repaired_dataset()

            # Save results
            if output_path:
                repaired_df.to_csv(output_path, index=False)

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'HoloClean',
                'type': 'data-oriented',
                'auto_level': 1,  # fully automatic
                'detectors': len(detectors),
                'featurizers': len(featurizers)
            }

            return repaired_df, info

        except Exception as e:
            raise RuntimeError(f"HoloCleancleaningfailure: {e}")

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used (always 0, HoloClean is fully automatic)."""
        return self.ground_truth_used


def holoclean_clean(dirty_path: str,
                    dc_path: str = None,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the HoloClean cleaner."""
    wrapper = HoloCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, dc_path, output_path)
