"""Build the final SAGA record file, one line per (dataset, method).

Reads the per-run records appended by run_saga.py (the runs of all machines in
one file), keeps the last record of every (dataset, method), and completes it:

* search time and search statistics parsed from the archived systemds.log
  (run_saga.parse_systemds_log). Records written before that parser existed
  carry a searchSecs.csv value of about 0.002 s, which is replaced.
* the deviation list rebuilt with run_saga.deviation_list (records written
  before the list existed have none). For those runs unit parsing did not
  exist yet, it would have changed no cell (only beers has unit strings, and
  beers was run with the current code).
* the contract check output of saga_check_outputs.py, the GT accounting and
  notes on the search (error-filter sample, saturated objective, top-1 score
  below the dirty score).

usage: python saga_finalize_records.py <code_root> <run_records.jsonl> <logs_dir> <check.jsonl> <out.jsonl>
"""
import json
import os
import sys

code_root, src, logs_dir, check_path, dst = sys.argv[1:6]
sys.path.insert(0, os.path.join(code_root, 'baselines', 'runners'))
import run_saga  # noqa: E402


last = {}
with open(src) as fh:
    for line in fh:
        r = json.loads(line)
        last[(r['dataset'], r['method'])] = r
checks = {}
with open(check_path) as fh:
    for line in fh:
        c = json.loads(line)
        checks[(c['dataset'], c.get('method', 'saga'))] = c

out = []
for (ds, mn), r in sorted(last.items()):
    cfg = dict(r['config'])
    cfg.setdefault('sil_n', 3000)
    task = r.get('task') or ('clustering' if ds == 'har' else None)
    search_task = r.get('search_task') or task
    if task is None:
        task = search_task = 'classification' if cfg['eval_func'] == 'evalClassification' else 'regression'
    log_path = os.path.join(logs_dir, f'{ds}_{mn}.systemds.log')
    with open(log_path, errors='replace') as fh:
        stats = run_saga.parse_systemds_log(fh.read())
    with open(os.path.join(logs_dir, f'{ds}_{mn}.cmd.json')) as fh:
        cmd = json.load(fh)
    assert ' '.join(cmd) == r['cmd'], (ds, mn, 'archived cmd.json differs from the record')
    dropped = int(r.get('n_train', 0)) - int(r.get('n_train_used', r.get('n_train', 0)))
    # the label name only enters the har label-surrogate item; har's label column is gt
    dev = run_saga.deviation_list(task, search_task, 'gt', r['input_stats'], cfg, cfg.get('cluster_k', 2), dropped)
    # stored items that the rebuilt list does not reproduce are kept, except the hyper-parameter item
    # that the earlier runner also wrote for the silhouette objective (which tunes nothing)
    dev += [d for d in r.get('deviations', []) if d not in dev
            and not (search_task == 'clustering' and d.startswith('evaluation model hyper-parameters'))]
    notes = []
    if 'search_rows' in stats and stats['search_rows'] < stats['search_rows_of']:
        notes.append(f'topk_cleaning error-filter sampling (builtin, n >= 3500 and more than half of the columns '
                     f'categorical): the search fitted pipelines on {stats["search_rows"]} of '
                     f'{stats["search_rows_of"]} train rows; the export replays the top-1 pipeline on all rows')
    sup = search_task == 'classification'
    if sup and r['dirty_val_score'] >= 100.0 - 1e-9:
        n = 'search objective saturated: the dirty train split already reaches 100 % validation accuracy, so no ' \
            'pipeline can score above the dirty data'
        if r['topk_val_scores'] and min(r['topk_val_scores']) >= 100.0 - 1e-9:
            n += '; every top-k pipeline ties at 100 % and the top-1 is the first tie found'
        notes.append(n)
    if r['topk_val_scores'] and r['topk_val_scores'][0] < r['dirty_val_score'] - 1e-12:
        notes.append(f'top-1 search score {r["topk_val_scores"][0]:.4g} is below the dirty score '
                     f'{r["dirty_val_score"]:.4g}; SAGA returns its top-1 pipeline regardless, applied as specified')
    rec = dict(r)
    rec.update({
        'task': task, 'search_task': search_task,
        'gt_cells': 0,
        'gt_note': 'SAGA reads no ground-truth cell of the train split; its only clean data is the 20 % validation '
                   'split used as the search objective (allowed for every task-driven method)',
        'budget_matched_variant': 'not applicable: SAGA has no ground-truth budget parameter and consumes 0 cells, '
                                  'so the default run is also the budget-matched run',
        'search_secs': stats.get('search_secs'), 'search_log': stats,
        'deviations': dev, 'notes': notes,
        'contract_check': checks.get((ds, mn)),
        'systemds_log': log_path,
    })
    out.append(rec)
with open(dst, 'w') as fh:
    for rec in out:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
for rec in out:
    c = rec['contract_check'] or {}
    print(json.dumps({'dataset': rec['dataset'], 'method': rec['method'], 'secs': rec['secs'],
                      'search_secs': rec['search_secs'], 'gt': rec['gt_cells'], 'ok': c.get('ok'),
                      'n_dev': len(rec['deviations']), 'notes': len(rec['notes'])}))
