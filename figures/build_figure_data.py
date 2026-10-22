#!/usr/bin/env python
"""Export the final paper inputs for Figures 4 and 5 without running an experiment.

    python figures/build_figure_data.py
    python figures/build_figure_data.py --check

fig4_final.json and abl_final.json in figures/data contain the reported observations, their first-choice
downstream models, and links to the corresponding result and run manifest. This command
validates those inputs and exports the CSV files consumed by make_figures.py. It does not
train a model, rescore a table, choose a run, or change an observation.

Figure 4 contains eight datasets and eighteen baselines plus DemandClean. REDR counts
erroneous records. GT usage is a percentage of dataset cells. The FM baseline uses zero
GT, and its API cost is reported separately in results/llm/llm_cost.json. DAGPipe outputs
transformed features in place of cleaned cells, so its REDR and detection F1 are undefined
and stay empty.
Figure 5 contains one reported observation for each of five configurations on those
eight datasets. An observation's manifest records the exact configuration of its run.
"""
import argparse
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DATASETS = ['beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa',
            'soilmoisture', 'har']
METHODS = ['holoclean', 'raha_baran', 'uniclean', 'lopster', 'horizon', 'activeclean',
           'comet', 'boostclean', 'saga', 'diffprep', 'ctxpipe', 'simpleimputer',
           'mlimputer', 'llm', 'diana', 'learn2clean', 'retclean', 'dagpipe', 'DemandClean']
# metrics a method's output leaves undefined: DAGPipe outputs transformed features, not cleaned cells
UNDEFINED = {'dagpipe': ('redr', 'det_f1')}
VARIANTS = ['full', 'ngt', 'nd', 'arch', 'stage']
MODELS = {'beers': 'rf', 'breast_cancer': 'lr', 'smartfactory': 'rf', 'bike': 'lr',
          'mercedes': 'rf', 'nasa': 'lr', 'soilmoisture': 'gb', 'har': 'agglomerative'}
METRICS = {'beers': 'accuracy', 'breast_cancer': 'accuracy', 'smartfactory': 'accuracy',
           'bike': 'r2', 'mercedes': 'r2', 'nasa': 'r2', 'soilmoisture': 'r2',
           'har': 'silhouette'}
BASELINE_FIELDS = ['dataset', 'method', 'model', 'metric', 'score', 'nofix', 'repairall',
                   'tol_rel', 'redr', 'gt_pct', 'det_f1', 'result', 'manifest']
ABLATION_FIELDS = ['dataset', 'variant', 'model', 'metric', 'score', 'nofix', 'repairall', 'tol_rel', 'seed',
                   'dataset_config', 'variant_config', 'result', 'manifest']


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field}: expected a finite numeric observation, got {value!r}')
    return value


def result_link(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'results':
        raise ValueError(f'Expected a repository-relative result path: {value!r}')
    return value


def load_rows():
    baseline = json.loads((DATA / 'fig4_final.json').read_text(encoding='utf-8'))
    ablation = json.loads((DATA / 'abl_final.json').read_text(encoding='utf-8'))
    for source in (baseline, ablation):
        if source['dataset_order'] != DATASETS or set(source['datasets']) != set(DATASETS):
            raise ValueError('Figure inputs must contain exactly the eight paper datasets in order')
    if baseline['method_order'] != METHODS or ablation['variant_order'] != VARIANTS:
        raise ValueError('The figure methods or configurations do not match the paper')

    baseline_rows, ablation_rows = [], []
    for ds in DATASETS:
        b = baseline['datasets'][ds]
        a = ablation['datasets'][ds]
        for group in (a, b):
            if group['model'] != MODELS[ds] or group['metric'] != METRICS[ds]:
                raise ValueError(f'{ds}: the observation must use the first-choice paper model')
        nf = number(b['references']['NoFix'], f'{ds}/NoFix')
        ra = number(b['references']['RepairAll'], f'{ds}/RepairAll')
        if abs(ra - nf) < 1e-12:
            raise ValueError(f'{ds}: relative tolerance has no nonzero reference gain')
        if set(b['methods']) != set(METHODS) or set(a['variants']) != set(VARIANTS):
            raise ValueError(f'{ds}: a method or configuration is missing or duplicated')

        for method in METHODS:
            observation = b['methods'][method]
            undefined = UNDEFINED.get(method, ())
            fields = {key: number(observation[key], f'{ds}/{method}/{key}')
                      for key in ('score', 'tol_rel', 'redr', 'gt_pct', 'det_f1') if key not in undefined}
            for key in undefined:
                if observation[key] is not None:
                    raise ValueError(f'{ds}/{method}: {key} is undefined for this method and must be empty')
                fields[key] = ''
            if not math.isclose(fields['tol_rel'], (fields['score'] - nf) / (ra - nf),
                                rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f'{ds}/{method}: relative tolerance disagrees with its references')
            if ('redr' not in undefined and fields['redr'] > 1) or fields['gt_pct'] < 0 or \
                    ('det_f1' not in undefined and not 0 <= fields['det_f1'] <= 1):
                raise ValueError(f'{ds}/{method}: an observation is outside its metric range')
            if method == 'llm' and fields['gt_pct'] != 0:
                raise ValueError(f'{ds}: API dollars must not be entered as GT cells')
            baseline_rows.append(dict(dataset=ds, method=method, model=b['model'],
                                      metric=b['metric'], nofix=nf, repairall=ra, **fields,
                                      result=result_link(observation['result']),
                                      manifest=result_link(observation['manifest'])))

        for variant in VARIANTS:
            observation = a['variants'][variant]
            score = number(observation['score'], f'{ds}/{variant}/score')
            tol = number(observation['tol_rel'], f'{ds}/{variant}/tol_rel')
            references = observation.get('references', b['references'])
            tol_nf = number(references['NoFix'], f'{ds}/{variant}/NoFix')
            tol_ra = number(references['RepairAll'], f'{ds}/{variant}/RepairAll')
            if not math.isclose(tol, (score - tol_nf) / (tol_ra - tol_nf), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f'{ds}/{variant}: relative tolerance disagrees with its references')
            if variant == 'full' and score != b['methods']['DemandClean']['score']:
                raise ValueError(f'{ds}: Full must be the same deployed observation as Figure 4')
            ablation_rows.append(dict(dataset=ds, variant=observation['label'], model=a['model'],
                                      metric=a['metric'], score=score, nofix=tol_nf, repairall=tol_ra, tol_rel=tol,
                                      seed=observation.get('seed', ''),
                                      dataset_config=observation.get('config', a['config']),
                                      variant_config=observation.get('variant_config', ''),
                                      result=result_link(observation['result']),
                                      manifest=result_link(observation['manifest'])))
    return baseline_rows, ablation_rows


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DATA, help='directory for the two CSV exports')
    parser.add_argument('--check', action='store_true', help='validate the inputs without writing CSV files')
    args = parser.parse_args()
    baseline_rows, ablation_rows = load_rows()
    if args.check:
        print(f'Validated {len(baseline_rows)} Figure 4 and {len(ablation_rows)} Figure 5 observations.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'fig_baselines.csv', BASELINE_FIELDS, baseline_rows)
    write_csv(args.out / 'fig_ablation.csv', ABLATION_FIELDS, ablation_rows)
    print(f'Exported {len(baseline_rows)} Figure 4 and {len(ablation_rows)} Figure 5 observations to {args.out}.')


if __name__ == '__main__':
    main()
