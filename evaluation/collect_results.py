#!/usr/bin/env python3
"""Collect the released observations without fitting or evaluating a model.

Run from any directory:
    python evaluation/collect_results.py
    python evaluation/collect_results.py --check

Only the fixed result directories listed below are read. Per-run score and
manifest files remain the source of the consolidated tables and indexes.
"""
import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['adult', 'beers', 'breast_cancer', 'smartfactory', 'bike',
            'mercedes', 'nasa', 'soilmoisture', 'har', 'flights', 'soccer',
            'hospitals']
MODELS = {'adult': ('rf', 'gb'), 'beers': ('rf', 'gb'),
          'breast_cancer': ('lr', 'gb'), 'smartfactory': ('rf', 'gb'),
          'bike': ('lr', 'lasso'), 'mercedes': ('rf', 'ridge'),
          'nasa': ('lr', 'rf'), 'soilmoisture': ('gb', 'rf'),
          'har': ('agglomerative', 'kmeans'), 'flights': ('dt',),
          'soccer': ('dt',), 'hospitals': ('rf',)}
DENOMINATORS = {'adult': 678330, 'beers': 24100, 'breast_cancer': 6990,
                'smartfactory': 449255, 'bike': 278064, 'mercedes': 1586793,
                'nasa': 9018, 'soilmoisture': 86912, 'har': 280000,
                'flights': 16632, 'soccer': 106100, 'hospitals': 19000}
REFERENCES = [('NoFix', 'nofix'), ('DeleteAll', 'deleteall'),
              ('ReplaceAll', 'replaceall'), ('RepairAll', 'repairall')]
BASELINES = [('HoloClean', 'holoclean'), ('Horizon', 'horizon'),
             ('Raha+Baran', 'raha_baran'), ('UniClean', 'uniclean'),
             ('Lopster', 'lopster'), ('ActiveClean', 'activeclean'),
             ('BoostClean', 'boostclean'), ('SAGA', 'saga'),
             ('DiffPrep', 'diffprep'), ('Comet', 'comet'),
             ('CtxPipe', 'ctxpipe'), ('MLImputer', 'mlimputer'),
             ('SimpleImputer', 'simpleimputer'), ('LLM', 'llm'),
             ('DIANA', 'diana'), ('Learn2Clean', 'learn2clean'),
             ('RetClean', 'retclean'), ('DAGPipe', 'dagpipe')]
# methods whose records hold the first-choice model only, the model the paper reports for them
FIRST_MODEL_ONLY = {'DiffPrep', 'DIANA', 'Learn2Clean', 'RetClean', 'DAGPipe'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('{} must be a finite number: {!r}'.format(name, value))
    return value


def main_runs():
    for dataset in DATASETS:
        base = ROOT / 'results/main_table' / dataset
        for name, slug in REFERENCES:
            yield dataset, name, 'references', base / 'references' / slug
        for name, slug in BASELINES:
            yield dataset, name, 'baselines', base / 'baselines' / slug
        yield dataset, 'DC', 'demandclean', base / 'demandclean'


def budget_runs():
    # ActiveClean, Comet and Lopster on the nine benchmark datasets, RetClean on all twelve
    for dataset in DATASETS:
        methods = ('activeclean', 'comet', 'lopster', 'retclean') if dataset in DATASETS[:9] else ('retclean',)
        for method in methods:
            for setting in ('dcbudget', 'ample'):
                path = ROOT / 'results/budget' / dataset / (method + '_' + setting)
                yield dataset, method, setting, path


def load_run(dataset, method, role, path, main_table):
    scores = read_json(path / 'scores.json')
    manifest = read_json(path / 'manifest.json')
    model = scores['model']
    if model != MODELS[dataset][0] or scores['dataset'] != dataset:
        raise ValueError('Dataset or primary model mismatch: {}'.format(path))
    values = dict(scores.get('model_scores', scores.get('all_models', {})))
    values[model] = finite(scores['test'], relative(path) + '/test')
    required = MODELS[dataset] if main_table and method not in FIRST_MODEL_ONLY else (model,)
    values = {key: finite(values[key], relative(path) + '/' + key) for key in required}
    gt_cells = manifest.get('ground_truth_cells', scores.get('gt_cells'))
    finite(gt_cells, relative(path) + '/GT cells')
    if gt_cells < 0 or int(gt_cells) != gt_cells:
        raise ValueError('Invalid GT cell count: {}'.format(path))
    if scores.get('gt_cells', gt_cells) != gt_cells:
        raise ValueError('GT count differs between score and manifest: {}'.format(path))
    artifact_names = ['cleaned_encoded.npz'] if role == 'demandclean' else []
    artifact_names += [manifest.get('artifact', ''), 'cleaned.csv.gz', 'cleaned.csv', 'transformed.npz']
    artifact = next((path / name for name in artifact_names if name and (path / name).is_file()), None)
    if artifact is None:
        raise ValueError('Missing result artifact: {}'.format(path))
    for name in manifest.get('files', []):
        if not isinstance(name, str) or not (path / name).is_file():
            raise ValueError('Missing manifest artifact: {}/{}'.format(path, name))
    entry = {'dataset': dataset, 'method': method, 'role': role,
             'run': relative(path), 'manifest': relative(path / 'manifest.json'),
             'scores': relative(path / 'scores.json'), 'artifact': relative(artifact),
             'model': model, 'metric': scores['metric'], 'score': values[model],
             'model_scores': values, 'gt_cells': int(gt_cells),
             'gt_denominator': DENOMINATORS[dataset],
             'gt_pct': 100.0 * gt_cells / DENOMINATORS[dataset]}
    if 'seed' in manifest:
        entry['seed'] = manifest['seed']
    if not main_table:
        entry['setting'] = path.name.rsplit('_', 1)[1]
    return entry, scores


def build_outputs():
    main, budget, downstream, redr = [], [], [], []
    attribution, operation = [], []
    cells = {}
    gt = {'rule': 'Clean cells read on the training rows, including detector calibration and repairs; '
                   'the denominator is the full-table cell count, excluding the row identifier.',
          'denominators': DENOMINATORS, 'records': []}
    for dataset in DATASETS:
        gt[dataset] = {}
    specifications = [(True, dataset, method, role, path)
                      for dataset, method, role, path in main_runs()]
    specifications += [(False, dataset, method, 'budget', path)
                       for dataset, method, setting, path in budget_runs()]
    for is_main, dataset, method, role, path in specifications:
        entry, scores = load_run(dataset, method, role, path, is_main)
        (main if is_main else budget).append(entry)
        for model, value in entry['model_scores'].items():
            row = {key: entry[key] for key in ('dataset', 'method', 'role', 'run', 'artifact', 'metric')}
            row.update(model=model, test=value, gt_cells=entry['gt_cells'], gt_pct=entry['gt_pct'])
            if 'setting' in entry:
                row['setting'] = entry['setting']
            downstream.append(row)
        redr_path = path / 'redr.json'
        redr_value = read_json(redr_path) if redr_path.is_file() else scores.get('redr')
        if redr_value is not None:
            finite(redr_value['redr'], relative(path) + '/REDR')
            row = {'dataset': dataset, 'method': method, 'role': role, 'run': relative(path)}
            row.update({key: redr_value[key] for key in
                        ('r_before', 'r_after', 'redr', 'scope_rows', 'kept_rows') if key in redr_value})
            redr.append(row)
        if is_main:
            cells[dataset + '|' + method] = {'scores': entry['model_scores'], 'source': entry['run']}
            gt[dataset][method] = entry['gt_pct']
            gt['records'].append({'dataset': dataset, 'method': method, 'run': entry['run'],
                                  'gt_cells': entry['gt_cells'], 'gt_pct': entry['gt_pct'],
                                  'denominator': DENOMINATORS[dataset],
                                  'manifest': entry['manifest']})
        if role == 'demandclean':
            for name, destination in [('attribution.json', attribution),
                                      ('attribution_operation.json', operation)]:
                row = read_json(path / name)
                if row['dataset'] != dataset or not math.isclose(row['v_full'], entry['score'], abs_tol=1e-12):
                    raise ValueError('Attribution is not for the released output: {}/{}'.format(path, name))
                row['run'] = entry['run']
                destination.append(row)
    if len(main) != 276 or len(budget) != 78:
        raise ValueError('Incomplete result grid')
    ablation = []
    for dataset in ('beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har'):
        for variant in ('ngt', 'nd', 'arch', 'stage'):
            path = ROOT / 'results/ablation' / dataset / variant
            for name in ('manifest.json', 'scores.json'):
                if not (path / name).is_file():
                    raise ValueError('Missing ablation record: {}/{}'.format(path, name))
            ablation.append({'dataset': dataset, 'variant': variant, 'run': relative(path),
                             'manifest': relative(path / 'manifest.json'),
                             'scores': relative(path / 'scores.json')})
    index = {'datasets': DATASETS, 'main_table': main, 'budget': budget,
             'ablation': ablation,
             'robustness': 'results/robustness/index.json',
             'transfer': 'results/transfer/index.json',
             'figures': {'4': 'figures/data/fig4_final.json', '5': 'figures/data/abl_final.json'},
             'tables': {'cells': 'results/tables/cells.json', 'gt': 'results/tables/gt_final.json',
                        'organic': 'results/real_world/table.json'},
             'supporting_records': {
                 'operation_attribution': 'results/attribution_operation/manifest.json',
                 'har': 'results/main_table/har/demandclean/manifest.json'},
             'llm': 'results/llm/llm_cost.json',
             'aggregation_command': 'python evaluation/collect_results.py --check'}
    organic = {'table': 'Organically dirty datasets', 'datasets': {}}
    for dataset in ('beers', 'flights', 'soccer', 'hospitals'):
        organic['datasets'][dataset] = {
            'model': MODELS[dataset][0],
            'results': {entry['method']: entry['scores'] for entry in main if entry['dataset'] == dataset},
            'llm_cost': 'results/llm/llm_cost.json'}
    return {'results/index.json': index, 'results/tables/cells.json': cells,
            'results/tables/gt_final.json': gt, 'results/real_world/table.json': organic,
            'results/scores/downstream.jsonl': downstream,
            'results/scores/redr.jsonl': redr,
            'results/scores/attribution.jsonl': attribution,
            'results/scores/attribution_operation.jsonl': operation}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='check the existing indexes without writing')
    args = parser.parse_args(argv)
    outputs = build_outputs()
    for name, value in outputs.items():
        path = ROOT / name
        jsonl = path.suffix == '.jsonl'
        if args.check:
            saved = ([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                     if jsonl else read_json(path))
            if value != saved:
                raise ValueError('Consolidated output differs from its source records: {}'.format(name))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            text = ('\n'.join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in value) + '\n'
                    if jsonl else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            path.write_text(text, encoding='utf-8')
    action = 'Validated' if args.check else 'Collected'
    print('{} 276 main-table runs, 78 budget runs, and 24 attribution records.'.format(action))


if __name__ == '__main__':
    main()
