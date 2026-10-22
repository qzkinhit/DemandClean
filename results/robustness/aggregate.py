#!/usr/bin/env python
"""Rebuild the rule-library table from archived scores; no cleaning or scoring is run.

    python results/robustness/aggregate.py
    python results/robustness/aggregate.py --write
"""
import argparse
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_table():
    index = json.loads((HERE / 'index.json').read_text())
    groups = {}
    for item in index['runs']:
        key = (item['dataset'], item['condition'], item['draw'])
        if key in groups:
            raise ValueError(f'Duplicate rule-library observation: {key}')
        run = ROOT / item['run']
        score = json.loads((run / 'scores.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        if not manifest['completion']['console_finished'] or manifest['completion']['training_seconds'] != 0:
            raise ValueError(f'{item["run"]}: expected completed checkpoint inference')
        nf, ra = score['references']['NoFix'], score['references']['RepairAll']
        tol = max(-1.0, min(2.5, (score['test'] - nf) / (ra - nf)))
        if abs(tol - score['tol_rel']) > 1e-12:
            raise ValueError(f'{item["run"]}: tolerance does not match its stored score and references')
        groups[key] = (score['detection_f1'], tol)
    rows = []
    for condition in index['conditions']:
        draws = [None] if condition in ('full', 'dirty_derived') else [0, 1, 2]
        per_dataset = []
        for ds in index['datasets']:
            values = [groups[(ds, condition, draw)] for draw in draws]
            per_dataset.append((mean(x[0] for x in values), mean(x[1] for x in values)))
        rows.append({'condition': condition, 'datasets': len(per_dataset),
                     'draws_per_dataset': len(draws),
                     'detection_f1': mean(x[0] for x in per_dataset),
                     'tol_rel': mean(x[1] for x in per_dataset)})
    if len(groups) != 126:
        raise ValueError(f'Expected 126 reported rule-library observations, found {len(groups)}')
    return {'table': 'rule_library_robustness', 'observations': len(groups), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true', help='also save the summary as table.json')
    args = parser.parse_args()
    text = json.dumps(build_table(), indent=2) + '\n'
    if args.write:
        (HERE / 'table.json').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
