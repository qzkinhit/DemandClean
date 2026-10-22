"""
SimpleImputer wrapper: imputation with column statistics (mean, median or mode).

Ground truth used: none (fully automatic).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.preprocessing import LabelEncoder

# Common missing-value tokens (case-insensitive)
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class SimpleImputerWrapper:
    """
    Wrapper of simple imputers.

    1. mean
    2. median
    3. most_frequent
    4. constant
    """

    def __init__(self,
                 strategy: str = 'mean',
                 fill_value: Optional[str] = None,
                 verbose: bool = False):
        """
        Args:
            strategy: 'mean', 'median', 'most_frequent' or 'constant'
            fill_value: fill value of the 'constant' strategy
            verbose: print progress information
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose

        self.numeric_imputer = None
        self.categorical_imputer = None
        self.ground_truth_used = 0

    def setup(self):
        """Create the imputers."""
        # Imputer of the numeric columns
        if self.strategy in ['mean', 'median']:
            self.numeric_imputer = SklearnSimpleImputer(strategy=self.strategy)
        elif self.strategy == 'constant':
            self.numeric_imputer = SklearnSimpleImputer(
                strategy='constant',
                fill_value=self.fill_value if self.fill_value else 0
            )
        else:
            self.numeric_imputer = SklearnSimpleImputer(strategy='mean')

        # Imputer of the categorical columns
        if self.strategy == 'most_frequent':
            self.categorical_imputer = SklearnSimpleImputer(strategy='most_frequent')
        elif self.strategy == 'constant':
            self.categorical_imputer = SklearnSimpleImputer(
                strategy='constant',
                fill_value=self.fill_value if self.fill_value else 'missing'
            )
        else:
            self.categorical_imputer = SklearnSimpleImputer(strategy='most_frequent')

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

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute.

        Args:
            data: input DataFrame

        Returns:
            imputed DataFrame
        """
        result = data.copy()

        # Numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        # Impute the numeric columns
        if numeric_cols:
            result[numeric_cols] = self.numeric_imputer.fit_transform(data[numeric_cols])

        # Impute the categorical columns
        if categorical_cols:
            # Categorical columns as strings, None handled
            cat_data = data[categorical_cols].astype(str).replace('nan', np.nan)
            result[categorical_cols] = self.categorical_imputer.fit_transform(cat_data)

        return result

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

        # Impute
        repaired_df = self.impute(data)

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
            'method': f'SimpleImputer-{self.strategy}',
            'type': 'data-preparation',
            'auto_level': 1,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'imputed_cells': int(missing_before - missing_after)
        }

        return repaired_df, info


def simpleimputer_clean(dirty_path: str,
                        output_path: str = None,
                        **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the SimpleImputer cleaner."""
    wrapper = SimpleImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
