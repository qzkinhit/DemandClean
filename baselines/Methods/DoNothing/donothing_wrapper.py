"""
DoNothing wrapper: the no-cleaning reference strategy (NoFix).

The dirty table is returned unchanged. Ground truth used: none.
"""

import os
import pandas as pd
from typing import Dict, Tuple


class DoNothingWrapper:
    """
    DoNothing cleaner.

    Returns the input table without any change, the lower reference of the comparison.
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: print progress information
        """
        self.verbose = verbose
        self.ground_truth_used = 0

    def setup(self):
        """No-op."""
        return True

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        "Clean" the table (no operation).

        Args:
            dirty_path: path of the dirty table
            output_path: output path

        Returns:
            the input table and run information
        """
        # Read the dirty table
        data = pd.read_csv(dirty_path)

        if self.verbose:
            print(f"DoNothing: Read data {len(data)} row, {len(data.columns)} column")
            print("DoNothing: no cleaning")

        # Save the result (identical to the input)
        if output_path:
            data.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': 0,
            'method': 'DoNothing',
            'type': 'baseline',
            'auto_level': 1,
            'rows': len(data),
            'columns': len(data.columns),
            'changes_made': 0
        }

        return data, info

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used."""
        return self.ground_truth_used


def donothing_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the DoNothing cleaner."""
    wrapper = DoNothingWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
