#!/usr/bin/env python
"""Size and error statistics of the datasets (Table tab:datasets).

    python evaluation/dataset_stats.py
    python evaluation/dataset_stats.py --datasets beers,nasa

For every dataset the oracle error mask of evaluation/build_replaceall.py compares the dirty and
the clean table cell by cell (numeric tolerance 1e-6). Printed per dataset
  rows        data rows of dirty_index.csv
  columns     columns of the file, the index column included
  row_pct     rows with at least one error cell, in percent of the rows
  cell_pct    error cells in percent of the cells of the data columns (index column excluded)
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402
from build_replaceall import error_mask  # noqa: E402

DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har', 'mercedes', 'nasa', 'smartfactory', 'soilmoisture',
            'flights', 'soccer', 'hospitals']


def stats(root, ds):
    d = pd.read_csv(os.path.join(root, ds, 'dirty_index.csv'), low_memory=False)
    c = pd.read_csv(os.path.join(root, ds, 'clean_index.csv'), low_memory=False)
    cols = [k for k in d.columns if k != 'index']
    M = error_mask(d, c, cols)
    return dict(dataset=ds, rows=len(d), columns=len(d.columns), error_cells=int(M.sum()),
                row_pct=round(100.0 * M.any(axis=1).mean(), 2),
                cell_pct=round(100.0 * M.sum() / M.size, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args()
    root = P.data_root(a.data_root)
    for ds in a.datasets.split(','):
        if os.path.exists(os.path.join(root, ds, 'dirty_index.csv')):
            print(json.dumps(stats(root, ds)), flush=True)


if __name__ == '__main__':
    main()
