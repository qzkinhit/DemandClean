"""Data protocol of the DemandClean runs for the runners of the baselines.

load_protocol returns the dataset configuration, the encoders and the 60/20/20 split that the
DemandClean runs and the scorer use: preprocess_data and encode_subset of
scripts/run_demandclean.py and split_idx of evaluation/paper_score.py. Categories absent from the
dirty training split are mapped to the nearest known category by edit distance
(demandclean/utils/edit_distance.py), which is the encoding the reported Comet runs used.

read_table and paper_split serve the scripts that need the split and the raw tables only.
"""
import importlib.util
import os
import sys

import pandas as pd
import yaml


def read_table(path):
    """CSV file as strings without missing-value parsing, column names stripped of spaces and of a
    byte order mark."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('\ufeff') for c in df.columns]
    return df


def paper_split(root):
    """split_idx of evaluation/paper_score.py (60/20/20, seed 42)."""
    path = os.path.join(os.path.abspath(root), 'evaluation')
    if path not in sys.path:
        sys.path.insert(0, path)
    import paper_score
    return paper_score.split_idx


def load_protocol(root, dataset):
    """(DATASETS, preprocess_data, encode_subset, find_nearest_known, split_idx) for one dataset."""
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        'run_demandclean', os.path.join(root, 'scripts', 'run_demandclean.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with open(os.path.join(root, 'configs', 'datasets', dataset + '.yaml')) as f:
        cfg = yaml.safe_load(f)
    runner.DATASETS[dataset] = {
        'task_type': cfg['task'],
        'model_type': cfg['reward_model'],
        'label_col': cfg['label_col'],
        'categorical_cols': set(cfg.get('categorical_cols') or []),
    }
    from demandclean.utils.edit_distance import find_nearest_known
    runner._find_nearest_known = find_nearest_known
    sys.path.insert(0, os.path.join(root, 'evaluation'))
    import paper_score
    return (runner.DATASETS, runner.preprocess_data, runner.encode_subset, find_nearest_known,
            paper_score.split_idx)
