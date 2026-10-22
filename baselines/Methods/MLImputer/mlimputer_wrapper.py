"""
MLImputer wrapper: imputation of missing values with learned models.

Ground truth used: none (fully automatic).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Common missing-value tokens (case-insensitive)
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class MLImputerWrapper:
    """
    Wrapper of learned imputers.

    1. MICE (Multiple Imputation by Chained Equations)
    2. KNN imputation
    3. random-forest imputation
    """

    def __init__(self,
                 method: str = 'mice',
                 max_iter: int = 10,
                 n_neighbors: int = 5,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Args:
            method: 'mice', 'knn' or 'rf'
            max_iter: maximum MICE iterations
            n_neighbors: neighbours of the KNN imputer
            random_state: random seed
            verbose: print progress information
        """
        self.method = method
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose

        self.imputer = None
        self.label_encoders = {}
        self.ground_truth_used = 0

    def setup(self):
        """Create the imputer."""
        if self.method == 'mice':
            self.imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=2 if self.verbose else 0
            )
        elif self.method == 'knn':
            self.imputer = KNNImputer(
                n_neighbors=self.n_neighbors
            )
        elif self.method == 'rf':
            self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=10,
                    random_state=self.random_state
                ),
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=2 if self.verbose else 0
            )
        return True

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load a table and turn missing-value tokens into NaN."""
        self.data = pd.read_csv(data_path)
        # Missing-value tokens become NaN
        self.data = self._convert_placeholders_to_nan(self.data)
        return self.data

    def _convert_placeholders_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Turn common missing-value tokens into NaN.

        Tokens: empty, null, none, na, n/a, nan, missing and the empty string
        """
        result = df.copy()

        for col in result.columns:
            if result[col].dtype == object:  # String columns only
                # Create a lowercase copy for comparison
                col_lower = result[col].astype(str).str.lower().str.strip()

                # Mark cells that need to be replaced
                for placeholder in MISSING_VALUE_PLACEHOLDERS:
                    mask = col_lower == placeholder.lower()
                    result.loc[mask, col] = np.nan

        return result

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Encode the categorical features.

        Args:
            data: input DataFrame

        Returns:
            encoded array and column names
        """
        processed = data.copy()
        columns = data.columns.tolist()

        # Encode the categorical features
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # Handle missing values
            mask = processed[col].notna()
            if mask.sum() > 0:
                processed.loc[mask, col] = le.fit_transform(processed.loc[mask, col].astype(str))
                self.label_encoders[col] = le
            processed[col] = pd.to_numeric(processed[col], errors='coerce')

        return processed.values, columns

    def postprocess(self, X: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """
        Decode the categorical features.

        Args:
            X: imputed array
            columns: column names

        Returns:
            DataFrame
        """
        df = pd.DataFrame(X, columns=columns)

        # Decode the categorical features
        for col, le in self.label_encoders.items():
            df[col] = df[col].round().astype(int)
            df[col] = df[col].clip(0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(df[col])

        return df

    def impute(self, X: np.ndarray) -> np.ndarray:
        """
        Impute.

        Args:
            X: input array

        Returns:
            imputed array
        """
        return self.imputer.fit_transform(X)

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used."""
        return self.ground_truth_used

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the imputation.

        Args:
            dirty_path: path of the dirty table
            output_path: output path

        Returns:
            imputed table and run information
        """
        # initialize
        self.setup()

        # Load data
        data = self.load_data(dirty_path)

        # Count the missing values
        missing_before = data.isnull().sum().sum()

        # Encode
        X, columns = self.preprocess(data)

        # Impute
        X_imputed = self.impute(X)

        # Decode
        repaired_df = self.postprocess(X_imputed, columns)

        # Count the imputed cells
        missing_after = repaired_df.isnull().sum().sum()

        # Empty values become "empty"
        repaired_df = repaired_df.fillna('empty')
        repaired_df = repaired_df.replace('', 'empty')

        # Save results
        if output_path:
            repaired_df.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.get_ground_truth_cost(),
            'method': f'MLImputer-{self.method}',
            'type': 'data-preparation',
            'auto_level': 1,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'imputed_cells': int(missing_before - missing_after)
        }

        return repaired_df, info


def mlimputer_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the MLImputer cleaner."""
    wrapper = MLImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
