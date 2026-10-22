"""
Rules Parser - unified rule-file parser

Parses Data/{dataset}/rules.txt and extracts the relevant rules per baseline.

File format:
[HOLOCLEAN_DC]
t1&t2&EQ(t1.attr1,t2.attr2)&IQ(t1.attr3,t2.attr3)

[UNICLEAN]
Number("attr")
AttrRelation(["lhs"], ["rhs"], "name")

[HORIZON_FD]
lhs => rhs

[MSE_ATTRIBUTES]
attr1
attr2
"""

import os
import re
from typing import List, Dict, Optional, Tuple


def parse_rules_file(rules_path: str) -> Dict[str, List[str]]:
    """
    Parse a unified rules file.

    Args:
        rules_path: path to the rules.txt file

    Returns:
        dict mapping section names to their rule lists
    """
    if not os.path.exists(rules_path):
        return {}

    sections = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments.
            if not line or line.startswith('#'):
                continue

            # Detect section headers.
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    return sections


def get_holoclean_dcs(rules_path: str) -> List[str]:
    """Return the HoloClean Denial Constraints."""
    sections = parse_rules_file(rules_path)
    return sections.get('HOLOCLEAN_DC', [])


def get_uniclean_cleaners(rules_path: str) -> List[str]:
    """Return the UniClean cleaner definition strings."""
    sections = parse_rules_file(rules_path)
    return sections.get('UNICLEAN', [])


def get_horizon_fds(rules_path: str) -> List[Tuple[str, str]]:
    """
    Return the Horizon functional dependency rules.

    Returns:
        list of (lhs, rhs) tuples
    """
    sections = parse_rules_file(rules_path)
    fds = []

    for line in sections.get('HORIZON_FD', []):
        # Accept both '=>' and the Unicode double-arrow as separators.
        if '=>' in line:
            parts = line.split('=>')
        elif '\u21d2' in line:
            parts = line.split('\u21d2')
        else:
            continue

        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            fds.append((lhs, rhs))

    return fds


def get_mse_attributes(rules_path: str) -> List[str]:
    """Return the MSE evaluation attribute list."""
    sections = parse_rules_file(rules_path)
    return sections.get('MSE_ATTRIBUTES', [])


def parse_uniclean_cleaner(cleaner_str: str):
    """
    Parse a UniClean cleaner string into an executable object.

    Args:
        cleaner_str: e.g. 'Number("age")' or 'AttrRelation(["a"], ["b"], "0")'

    Returns:
        UniClean cleaner object
    """
    # Import UniClean cleaners lazily.
    try:
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Methods', 'UniClean'))
        from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
        from SampleScrubber.cleaner.multiple import AttrRelation

    # Evaluate the cleaner definition in a restricted namespace.
    local_vars = {
        'Number': Number,
        'Pattern': Pattern,
        'Outlier': Outlier,
        'Date': Date,
        'AttrRelation': AttrRelation
    }

    try:
        return eval(cleaner_str, {"__builtins__": {}}, local_vars)
    except Exception as e:
        print(f"Warning: Failed to parse cleaner '{cleaner_str}': {e}")
        return None


def get_uniclean_cleaner_objects(rules_path: str) -> List:
    """
    Return the list of UniClean cleaner objects.

    Args:
        rules_path: path to the rules.txt file

    Returns:
        list of cleaner objects
    """
    cleaner_strs = get_uniclean_cleaners(rules_path)
    cleaners = []

    for cleaner_str in cleaner_strs:
        cleaner = parse_uniclean_cleaner(cleaner_str)
        if cleaner:
            cleaners.append(cleaner)

    return cleaners


def write_holoclean_dc_file(rules_path: str, output_path: str) -> str:
    """
    Extract HoloClean DCs from the unified rules file and write them to a standalone file.

    Args:
        rules_path: path to the unified rules file
        output_path: output DC file path

    Returns:
        output file path
    """
    dcs = get_holoclean_dcs(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for dc in dcs:
            f.write(dc + '\n')

    return output_path


def write_horizon_fd_file(rules_path: str, output_path: str) -> str:
    """
    Extract Horizon FDs from the unified rules file and write them to a standalone file.

    Args:
        rules_path: path to the unified rules file
        output_path: output FD file path

    Returns:
        output file path
    """
    fds = get_horizon_fds(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} => {rhs}\n")

    return output_path


# Convenience helper
def get_dataset_rules(dataset_name: str, data_dir: str = 'Data') -> Dict[str, List[str]]:
    """
    Return all rules for a given dataset.

    Args:
        dataset_name: dataset name
        data_dir: data root directory

    Returns:
        rules dict
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return parse_rules_file(rules_path)


if __name__ == '__main__':
    # Quick test.
    import sys
    if len(sys.argv) > 1:
        rules_path = sys.argv[1]
    else:
        rules_path = 'Data/beers/rules.txt'

    print(f"Parsing: {rules_path}")
    sections = parse_rules_file(rules_path)

    for section, rules in sections.items():
        print(f"\n[{section}]")
        for rule in rules:
            print(f"  {rule}")
