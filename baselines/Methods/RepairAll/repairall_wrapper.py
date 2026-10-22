"""
RepairAll wrapper: the clean-table reference strategy.

The dirty table is replaced by the clean table, which repairs every error and gives the upper
reference of the comparison. It reads ground truth for every cell.
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple


class RepairAllWrapper:
    """
    RepairAll cleaner.

    Replaces the dirty table by the clean table, a repair of every error.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.method_name = "RepairAll"
        self.method_type = 2  # Needs ground truth

    def clean(self,
              dirty_path: str,
              clean_path: str,
              output_path: str,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        Return the clean table as the repair.

        Args:
            dirty_path: path of the dirty table
            clean_path: path of the clean table
            output_path: output file
            index_attribute: name of the index column

        Returns:
            (repaired DataFrame, run information)
        """
        # Read data
        dirty_data = pd.read_csv(dirty_path)
        clean_data = pd.read_csv(clean_path)

        if self.verbose:
            print(f"RepairAll: dirty table with {len(dirty_data)} rows and {len(dirty_data.columns)} columns")
            print(f"RepairAll: Readcleandata {len(clean_data)} row, {len(clean_data.columns)} column")

        # Number of repaired cells
        total_cells = len(dirty_data) * len(dirty_data.columns)
        repaired_cells = 0

        # Compare dirty and clean table cell by cell
        common_indices = dirty_data.index.intersection(clean_data.index)
        common_columns = [c for c in dirty_data.columns if c in clean_data.columns]

        for col in common_columns:
            if col == index_attribute:
                continue
            for idx in common_indices:
                try:
                    dirty_val = dirty_data.loc[idx, col]
                    clean_val = clean_data.loc[idx, col]
                    if str(dirty_val) != str(clean_val):
                        repaired_cells += 1
                except:
                    pass

        # The clean table is the output
        result = clean_data.copy()

        # Save results
        result.to_csv(output_path, index=False)

        if self.verbose:
            print(f"RepairAll: {repaired_cells} cells repaired")
            print(f"RepairAll: outputsaveto {output_path}")

        clean_info = {
            'original_rows': len(dirty_data),
            'output_rows': len(result),
            'repaired_cells': repaired_cells,
            'total_cells': total_cells,
            'repair_ratio': repaired_cells / total_cells if total_cells > 0 else 0,
            'ground_truth_cost': repaired_cells,  # Every ground-truth value is used
            'method_type': self.method_type
        }

        return result, clean_info


if __name__ == '__main__':
    # Simple test
    import sys
    if len(sys.argv) >= 3:
        wrapper = RepairAllWrapper(verbose=True)
        result, info = wrapper.clean(
            dirty_path=sys.argv[1],
            clean_path=sys.argv[2],
            output_path='test_repairall_output.csv'
        )
        print(f"result: {info}")
