#!/usr/bin/env python3
"""Export Figure 4 observations after checking their released score records.

    python figures/fig4_final.py --check
    python figures/fig4_final.py output/fig4_final.json

Detection and REDR are the saved measurements of the released outputs. DAGPipe outputs
transformed features, so it has neither and its observation leaves both empty. This
command reads and checks those records without reevaluating a cleaning method.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from build_figure_data import DATA, UNDEFINED, load_rows


ROOT = Path(__file__).resolve().parents[1]


def build():
    load_rows()
    result = copy.deepcopy(json.loads((DATA / 'fig4_final.json').read_text(encoding='utf-8')))
    for dataset, record in result['datasets'].items():
        for method, observation in record['methods'].items():
            score_path = ROOT / observation['result']
            score = json.loads(score_path.read_text(encoding='utf-8'))
            manifest = json.loads((ROOT / observation['manifest']).read_text(encoding='utf-8'))
            if score['model'] != record['model'] or score['dataset'] != dataset:
                raise ValueError('Wrong score record: ' + observation['result'])
            if not math.isclose(score['test'], observation['score'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Figure score differs from released output: ' + observation['result'])
            redr_path = score_path.parent / 'redr.json'
            redr = (json.loads(redr_path.read_text(encoding='utf-8')) if redr_path.exists()
                    else score.get('redr'))
            if 'redr' in UNDEFINED.get(method, ()):
                if redr is not None or observation['redr'] is not None:
                    raise ValueError('REDR is undefined for this output: ' + observation['result'])
            elif redr is None or not math.isclose(redr['redr'], observation['redr'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Figure REDR differs from released output: ' + observation['result'])
            gt_cells = manifest['ground_truth_cells']
            gt_pct = score.get('gt_pct', manifest.get('gt_pct'))
            if gt_pct is None or not math.isclose(gt_pct, observation['gt_pct'], rel_tol=0, abs_tol=5.1e-5):
                raise ValueError('Figure GT differs from released output: ' + observation['result'])
            if method == 'llm' and gt_cells != 0:
                raise ValueError('The FM cleaner does not receive clean values')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    value = build()
    if args.check:
        print('Validated %d Figure 4 observations against their released records.'
              % sum(len(record['methods']) for record in value['datasets'].values()))
    elif args.output is None:
        parser.error('provide an output JSON path, or use --check')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('Wrote ' + str(args.output))


if __name__ == '__main__':
    main()
