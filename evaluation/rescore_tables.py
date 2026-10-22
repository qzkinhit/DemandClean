#!/usr/bin/env python
"""Score the released reference/baseline outputs with their paper-model protocol.

    python evaluation/rescore_tables.py                       # main_table and budget outputs
    python evaluation/rescore_tables.py --datasets beers,nasa
    python evaluation/rescore_tables.py --table beers:nofix:results/main_table/beers/references/nofix/cleaned.csv.gz
    python evaluation/rescore_tables.py --datasets flights,soccer,hospitals --nearest
    python evaluation/rescore_tables.py --datasets beers --model gb
    python evaluation/rescore_tables.py --table har:diana:results/main_table/har/baselines/diana/cleaned_encoded.npz
    python evaluation/rescore_tables.py --table har:dagpipe:results/main_table/har/baselines/dagpipe/transformed.npz

Every table is aligned to the 60% training split by its `index` column (rows at training
positions only, in training-split order, rows a method deleted stay absent), encoded with the
encoders fitted on the dirty training split and scored with the main-table model on the clean
test split, which is how evaluation/paper_score.py scores a DemandClean run. One JSON object per
table is printed (results/scores/downstream.jsonl holds the output for the shipped tables).
DIANA, Learn2Clean and DiffPrep output their cleaned training rows in this encoding
(cleaned_encoded.npz), which is scored as saved. DAGPipe outputs the training and test features of
its learned graph (transformed.npz), and the model is trained and tested on these features. These
two cases follow score_output of baselines/runners/common.py, and no transformation is re-trained.
The other methods use their indexed CSV output. The default scan visits only the canonical
main_table and budget directories, without scanning historical copies of the real-world datasets.
The dataset's default categorical encoding is used unless --nearest explicitly overrides it.
--model selects another downstream model without changing the saved cleaning output.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_score as P  # noqa: E402

DS = ['adult', 'beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har']


def score_aligned(ds, df, ctx, model=None):
    task = P.DATASETS[ds]['task']
    main_model, metric = P.PAPER_MODEL[ds]
    model = model or main_model
    if model not in P.MODELS[task]:
        raise ValueError('model ' + model + ' is not supported for ' + task)
    tr_index = ctx['dirty']['index'].values[ctx['tr']]
    order = {v: i for i, v in enumerate(tr_index)}
    df = df[df['index'].isin(order)].drop_duplicates('index')
    df = df.iloc[np.argsort([order[v] for v in df['index'].values], kind='stable')].reset_index(drop=True)
    X, y = ctx['enc'].encode(df)
    res, n_tr, _ = P.score_encoded(task, X, y, {'test': ctx['test'], 'val': ctx['val']}, [model], metric)
    return res[model]['test'], n_tr


def score_saved_arrays(ds, path, ctx, model=None):
    """Score cleaned_encoded.npz or transformed.npz as score_output of baselines/runners/common.py."""
    task = P.DATASETS[ds]['task']
    main_model, metric = P.PAPER_MODEL[ds]
    model = model or main_model
    if model not in P.MODELS[task]:
        raise ValueError('model ' + model + ' is not supported for ' + task)
    with np.load(path, allow_pickle=False) as z:
        if Path(path).name == 'cleaned_encoded.npz':
            X, y, test = z['X_result'], z['y_result'], ctx['test']
        else:
            X = np.nan_to_num(z['X_train'], nan=0., posinf=0., neginf=0.)
            y = z['y_train']
            test = (np.nan_to_num(z['X_test'], nan=0., posinf=0., neginf=0.), ctx['test'][1])
    res, n_tr, _ = P.score_encoded(task, X, y, {'test': test}, [model], metric)
    return res[model]['test'], n_tr


# organically dirty datasets of Table tab:real
REAL_WORLD = ['flights', 'soccer', 'hospitals']


def shipped_tables(ds):
    """(name, path) of each canonical CSV or array output, exactly once."""
    out = []
    main = Path(P._ROOT) / 'results/main_table' / ds
    directories = [p for role in ('references', 'baselines') for p in (main / role).glob('*') if p.is_dir()]
    directories += [p for p in (Path(P._ROOT) / 'results/budget' / ds).glob('*') if p.is_dir()]
    for directory in sorted(directories):
        for filename in ('transformed.npz', 'cleaned_encoded.npz', 'cleaned.csv.gz', 'cleaned.csv'):
            path = directory / filename
            if path.is_file():
                out.append((directory.name, str(path)))
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', default=','.join(DS + REAL_WORLD))
    ap.add_argument('--table', action='append', default=[], help='dataset:name:path')
    ap.add_argument('--model', default=None, help='downstream model override, e.g. rf, gb, dt, lr or kmeans')
    ap.add_argument('--nearest', action='store_true', default=None,
                    help='explicitly use nearest-category encoding instead of the dataset default')
    a = ap.parse_args()
    P.USE_NEAREST = a.nearest
    root = P.data_root(a.data_root)
    items = [tuple(spec.split(':', 2)) for spec in a.table]
    if not items:
        for ds in a.datasets.split(','):
            items += [(ds, name, path) for name, path in shipped_tables(ds)]
    for ds, name, path in items:
        main_model, metric = P.PAPER_MODEL[ds]
        model = a.model or main_model
        relative = os.path.relpath(os.path.abspath(path), P._ROOT)
        parts = Path(relative).parts
        experiment = parts[1] if len(parts) > 1 and parts[0] == 'results' else 'custom'
        role = 'budget' if experiment == 'budget' else ('references' if 'references' in parts else 'baselines')
        rec = dict(dataset=ds, method=name, table=os.path.relpath(os.path.abspath(path), P._ROOT),
                   role=role, experiment=experiment, model=model, metric=metric)
        try:
            if Path(path).suffix == '.npz':
                rec['nearest'] = P._use_nearest(ds)
                ctx = P.dataset_context(root, ds)
                rec['test'], rec['n_train_rows'] = score_saved_arrays(ds, path, ctx, model=model)
            else:
                rec['nearest'] = P._use_nearest(ds)
                ctx = P.dataset_context(root, ds)
                df = pd.read_csv(path, low_memory=False)
                if 'index' not in df.columns:
                    raise ValueError('no index column')
                rec['rows'] = len(df)
                rec['test'], rec['n_train_rows'] = score_aligned(ds, df, ctx, model=model)
        except Exception as e:  # keep going
            rec['error'] = f'{type(e).__name__}: {e}'
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
