#!/usr/bin/env python
"""Rebuild the transfer matrices from archived scores; no model is run or trained.

    python results/transfer/aggregate.py
    python results/transfer/aggregate.py --write
"""
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_table():
    index = json.loads((HERE / 'index.json').read_text())
    observations = {}
    for item in index['runs']:
        key = (item['source_dataset'], item['target_dataset'])
        if key in observations:
            raise ValueError(f'Duplicate transfer observation: {key}')
        run = ROOT / item['run']
        score = json.loads((run / 'scores.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        if not manifest['completion']['console_finished'] or manifest['completion']['training_seconds'] != 0:
            raise ValueError(f'{item["run"]}: expected completed checkpoint inference')
        nf, ra = score['references']['NoFix'], score['references']['RepairAll']
        tol = max(-1.0, min(2.5, (score['test'] - nf) / (ra - nf)))
        if abs(tol - score['tol_rel']) > 1e-12:
            raise ValueError(f'{item["run"]}: tolerance does not match its stored score and references')
        observations[key] = tol
    blocks = {}
    for task, datasets in index['blocks'].items():
        blocks[task] = {'sources': datasets, 'targets': datasets,
                        'tol_rel': [[observations[(source, target)] for target in datasets]
                                    for source in datasets]}
    if len(observations) != 32:
        raise ValueError(f'Expected 32 reported transfer observations, found {len(observations)}')
    return {'table': 'cross_dataset_transfer', 'observations': len(observations), 'blocks': blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true', help='also save the matrices as table.json')
    args = parser.parse_args()
    text = json.dumps(build_table(), indent=2) + '\n'
    if args.write:
        (HERE / 'table.json').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
