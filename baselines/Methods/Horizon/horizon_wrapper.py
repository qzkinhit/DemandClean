"""
Horizon wrapper: data repair guided by functional dependencies.

Paper: Horizon: Scalable Dependency-driven Data Cleaning (VLDB 2021)

Ground truth used: none (fully automatic, needs functional-dependency rules).
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


def load_fds_from_rules(rules_path: str) -> List[Tuple[str, str]]:
    """
    Load the functional dependencies of Horizon from a rule file.

    Args:
        rules_path: rule file (for example data/beers/rules_baselines.txt)

    Returns:
        list of (lhs, rhs) tuples
    """
    fds = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"rule file not found: {rules_path}")

    in_horizon_section = False
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('[') and line.endswith(']'):
                in_horizon_section = (line == '[HORIZON_FD]')
                continue

            if in_horizon_section:
                # Accept both => and ⇒
                if '=>' in line:
                    parts = line.split('=>')
                elif '⇒' in line:
                    parts = line.split('⇒')
                else:
                    continue

                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    fds.append((lhs, rhs))

    return fds


def write_horizon_rule_file(rules_path: str, output_path: str) -> str:
    """
    Extract the FDs of a rule file and write them in the Horizon format.

    Args:
        rules_path: rule file
        output_path: output FD file

    Returns:
        output path
    """
    fds = load_fds_from_rules(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} ⇒ {rhs}\n")

    return output_path


class HorizonWrapper:
    """
    Wrapper of Horizon.

    Horizon repairs data from functional dependencies given in a file with lines
    "A ⇒ B". Ground truth used: none.
    """

    def __init__(self,
                 verbose: bool = False):
        """
        Args:
            verbose: print progress information
        """
        self.verbose = verbose
        self.ground_truth_used = 0  # Horizonisfully automatic

    def clean(self,
              dirty_path: str,
              clean_path: str,
              rule_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run Horizon.

        Args:
            dirty_path: path of the dirty table
            clean_path: path of the clean table
            rule_path: functional-dependency file
            output_path: output path

        Returns:
            repaired table and run information
        """
        # Import Horizon
        try:
            from .horizon import Horizon, dirty_cells
        except ImportError:
            from horizon import Horizon, dirty_cells

        if self.verbose:
            print("=" * 60)
            print("Horizon datacleaning")
            print("=" * 60)
            print(f"dirty table: {dirty_path}")
            print(f"rule file: {rule_path}")

        # Run Horizon
        pattern_expressions, dirty_c, elapsed_time = Horizon(dirty_path, rule_path, clean_path)

        # Apply pattern_expressions to the table
        repaired_df = pd.read_csv(dirty_path)
        # "empty" and the empty string become NaN
        repaired_df = repaired_df.replace({'empty': '', 'Empty': '', 'EMPTY': ''})
        repaired_df = repaired_df.fillna("")

        corrected_cells = 0
        for i, expr in enumerate(pattern_expressions):
            for col, val in expr.items():
                if col in repaired_df.columns:
                    old_val = str(repaired_df.iloc[i][col])
                    if old_val != str(val) and val != 'empty' and val != '':
                        repaired_df.iloc[i, repaired_df.columns.get_loc(col)] = val
                        corrected_cells += 1

        # Empty strings become "empty" again
        repaired_df = repaired_df.replace('', 'empty')

        if self.verbose:
            print(f"dirty cells identified: {len(dirty_c)}")
            print(f"repaircell: {corrected_cells}")
            print(f"time: {elapsed_time:.2f}s")

        # Save results
        if output_path:
            repaired_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"Repaired data saved: {output_path}")

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': 'Horizon',
            'type': 'data-oriented',
            'auto_level': 1,
            'dirty_cells': len(dirty_c),
            'corrected_cells': corrected_cells,
            'elapsed_time': elapsed_time
        }

        return repaired_df, info

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used."""
        return self.ground_truth_used


def horizon_clean(dirty_path: str,
                  clean_path: str,
                  rule_path: str,
                  output_path: str = None,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function of the Horizon cleaner."""
    wrapper = HorizonWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, rule_path, output_path)
