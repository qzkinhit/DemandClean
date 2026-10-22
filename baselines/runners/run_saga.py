"""Runner for SAGA (SystemDS 3.2.0 topk_cleaning) under the 60/20/20 protocol.

IO, argparse and process control live here; the method itself is
baselines/Methods/SAGA/{saga_topk.dml,saga_core.py}.

Example (baselines/scripts/run_saga.sh runs the nine datasets of the main table). SystemDS runs
in its own directory, so the paths are absolute:
  python baselines/runners/run_saga.py --dataset beers --repo $PWD \
      --sysds_home $SYSTEMDS_HOME --java $JAVA_HOME/bin/java --work_dir $PWD/runs/saga/work \
      --result_root $PWD/runs/saga/results --records $PWD/runs/saga/records.jsonl
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.abspath(os.path.join(HERE, '..', 'Methods', 'SAGA'))
sys.path.insert(0, METHOD_DIR)
import saga_core  # noqa: E402

CONFIG_XML = """<root>
  <sysds.localtmpdir>{tmp}</sysds.localtmpdir>
  <sysds.scratch>{scratch}</sysds.scratch>
  <sysds.optlevel>2</sysds.optlevel>
  <sysds.defaultblocksize>1000</sysds.defaultblocksize>
  <sysds.cp.parallel.ops>{par}</sysds.cp.parallel.ops>
  <sysds.cp.parallel.io>false</sysds.cp.parallel.io>
  <sysds.parallel.encode>false</sysds.parallel.encode>
  <sysds.native.blas>none</sysds.native.blas>
</root>
"""


def read_sysds_csv(path, header=False, dtype=float):
    """SystemDS writes a file or a directory of part files."""
    if os.path.isdir(path):
        parts = sorted(p for p in os.listdir(path) if not p.startswith('.') and not p.endswith('.crc'))
        frames = [pd.read_csv(os.path.join(path, p), header=None, dtype=str, keep_default_na=False)
                  for p in parts if os.path.getsize(os.path.join(path, p)) > 0]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if dtype is float:
        return df.replace({'NaN': 'nan', '': 'nan'}).astype(float).values
    return df


def parse_systemds_log(text):
    """Phase timings and search statistics printed by topk_cleaning.

    The DML time() pair around the topk_cleaning call is not ordered against the
    call by the SystemDS compiler (searchSecs.csv holds about 0.002 s for every
    run), so the search time is taken from the phase timings the builtin prints."""
    out = {}
    phase = None
    names = {'-- Cleaning - Get Dirty Score': 'dirty_score_secs',
             '-- Cleaning - Data Preparation': 'data_prep_secs',
             '-- Cleaning - Enum Logical Pipelines': 'enum_logical_secs'}
    for line in text.splitlines():
        s = line.strip()
        for key, name in names.items():
            if s.startswith(key):
                phase = name
        m = re.match(r'-- Cleaning - (Prepare Metadata|Prepare Labels|Enum Physical Pipelines): ([0-9.eE+-]+)s', s)
        if m:
            out[{'Prepare Metadata': 'prepare_meta_secs', 'Prepare Labels': 'prepare_labels_secs',
                 'Enum Physical Pipelines': 'enum_physical_secs'}[m.group(1)]] = float(m.group(2))
            continue
        m = re.match(r'---- finalized in: ([0-9.eE+-]+)s', s)
        if m and phase and phase not in out:
            out[phase] = float(m.group(1))
            continue
        m = re.match(r'sampled rows (\d+) out of (\d+)', s)
        if m:
            out['search_rows'] = int(m.group(1))
            out['search_rows_of'] = int(m.group(2))
        m = re.match(r'---- EnumLP\s+(converged|did not converge) after (\d+) / (\d+) iterations', s)
        if m:
            out['enum_logical_iterations'] = int(m.group(2))
            out['enum_logical_converged'] = m.group(1) == 'converged'
        m = re.match(r'actual executed pipelines: (\d+)', s)
        if m:
            out['physical_pipelines_executed'] = int(m.group(1))
        m = re.match(r'Total elapsed time:\s+([0-9.]+) sec', s)
        if m:
            out['systemds_elapsed_secs'] = float(m.group(1))
    keys = ('prepare_meta_secs', 'prepare_labels_secs', 'dirty_score_secs', 'data_prep_secs',
            'enum_logical_secs', 'enum_physical_secs')
    if all(k in out for k in keys):
        out['search_secs'] = round(sum(out[k] for k in keys), 1)
    return out


def deviation_list(task, search_task, label, input_stats, cfg, cluster_k, dropped_unlabeled):
    """Every departure from the SAGA paper and package set-up for one run.

    cfg holds eval_maxi, eval_maxii, max_iter, rv, topk and sil_n (argparse names)."""
    st = input_stats['columns']
    dev = [
        'hold-out search (cv=FALSE): pipelines fitted on the dirty train split, scored on the clean validation split',
        'top-1 pipeline replayed on the full dirty train split primitive by primitive; encoders skipped and affine '
        'rescaling inverted so the export stays in the original attribute space; SMOTE rows dropped',
        'non-numeric cells in numeric columns blanked before the search (what dropInvalidType does)',
        'numeric literals written in scientific notation so that the SystemDS 3.2.0 dropInvalidType primitive does '
        'not blank integral values (input format only, values unchanged)',
        'column types and categorical mask taken from the benchmark schema instead of SystemDS detectSchema',
    ]
    if search_task != 'clustering':
        dev.insert(1, 'evaluation model hyper-parameters selected once on the hold-out pair (package gridSearch cv '
                      'path fails in 3.2.0)')
    if input_stats.get('fd_mask_on'):
        dev.append('FD mask set to the categorical mask because rules.txt lists an FD between categorical '
                   'attributes (executePipeline::imputeByFd only aligns when the two masks are equal)')
    n_unit = sum(v.get('unit_parsed', 0) for v in st.values())
    if n_unit:
        dev.append(f'unit strings in numeric columns parsed to their number before the search '
                   f'({n_unit} train cells: ' + ', '.join(f'{c} {v["unit_parsed"]}' for c, v in st.items()
                                                           if v.get('unit_parsed')) + ')')
    if cfg['eval_maxi'] != 1000 or cfg['eval_maxii'] != 0:
        dev.append(f'multiLogReg iteration limits maxi={cfg["eval_maxi"]} maxii={cfg["eval_maxii"]} '
                   f'(package 1000 and 0) to fit the runtime budget')
    if cfg['max_iter'] != 10 or cfg['rv'] != 20 or cfg['topk'] != 3:
        dev.append(f'search budget max_iter={cfg["max_iter"]} rv={cfg["rv"]} topk={cfg["topk"]}')
    if task == 'clustering' and search_task == 'clustering':
        dev.append(f'clustering task: evaluation function replaced by k-means (k={cluster_k}) on the cleaned '
                   f'train matrix, silhouette of the nearest-centroid assignment on {cfg["sil_n"]} clean '
                   f'validation rows; row-number placeholder label so the label-free search branch is used')
    if task == 'clustering' and search_task == 'classification':
        dev.append(f'clustering task: supervised surrogate, label column {label} used as the class label '
                   f'with the package multiLogReg accuracy; {dropped_unlabeled} train rows with a missing '
                   f'label removed (absent from the export)')
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--repo', required=True, help='root of this repository (data/, configs/, scripts/)')
    ap.add_argument('--sysds_home', required=True)
    ap.add_argument('--java', default='java')
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--result_root', required=True, help='<root>/saga/<ds>_saga/<ds>_saga_cleaned.csv')
    ap.add_argument('--records', default=None, help='jsonl file for the run record')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--rv', type=int, default=20)
    ap.add_argument('--max_iter', type=int, default=10)
    ap.add_argument('--sample', type=float, default=1.0)
    ap.add_argument('--expected_increase', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval_maxi', type=int, default=1000, help='multiLogReg outer iteration limit')
    ap.add_argument('--eval_maxii', type=int, default=0, help='multiLogReg inner iteration limit, 0 = none')
    ap.add_argument('--xmx', default='12g')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=6 * 3600)
    ap.add_argument('--method_name', default='saga', help='result folder <root>/<name>/<ds>_<name>')
    ap.add_argument('--cluster_objective', choices=('silhouette', 'label'), default='silhouette',
                    help='clustering tasks: k-means silhouette on the clean validation rows, or the '
                         'supervised surrogate (multiLogReg accuracy on the label column)')
    ap.add_argument('--sil_n', type=int, default=3000, help='validation rows used for the silhouette')
    ap.add_argument('--no_parse_units', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from protocol import load_protocol, read_table as _read  # noqa: E402
    DATASETS, _, _, _, split_idx = load_protocol(args.repo, args.dataset)

    ds = args.dataset
    cfg = DATASETS[ds]
    task = cfg['task_type']
    if task not in ('classification', 'regression', 'clustering'):
        raise SystemExit(f'{ds}: task {task} has no SAGA evaluation function')
    search_task = task
    if task == 'clustering' and args.cluster_objective == 'label':
        search_task = 'classification'
    label = cfg['label_col']
    cats = set(cfg['categorical_cols'])
    ddir = os.path.join(args.repo, 'data', ds)
    dirty = _read(os.path.join(ddir, 'dirty_index.csv'))
    clean = _read(os.path.join(ddir, 'clean_index.csv'))
    assert len(dirty) == len(clean) and (dirty['index'].values == clean['index'].values).all()
    tr, va, te = split_idx(len(clean))
    dirty_tr = dirty.iloc[tr].reset_index(drop=True)
    clean_va = clean.iloc[va].reset_index(drop=True)
    n_train_split = len(dirty_tr)
    dropped_unlabeled = 0
    if search_task == 'classification' and task == 'clustering':
        keep = ~dirty_tr[label].isin(saga_core.MISS)
        dropped_unlabeled = int((~keep).sum())
        dirty_tr = dirty_tr[keep].reset_index(drop=True)
        clean_va = clean_va[~clean_va[label].isin(saga_core.MISS)].reset_index(drop=True)
    feature_cols = [c for c in dirty.columns if c not in ('index', label)]
    with open(os.path.join(ddir, 'rules.txt'), encoding='utf-8') as fh:
        fd_cols = saga_core.parse_fd_columns(fh.read())

    run_dir = os.path.join(args.work_dir, ds)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    out_dir = os.path.join(run_dir, 'out')
    for d in (run_dir, out_dir, os.path.join(run_dir, 'tmp'), os.path.join(run_dir, 'scratch')):
        os.makedirs(d, exist_ok=True)

    inp = saga_core.prepare_inputs(dirty_tr, clean_va, feature_cols, label, cats, search_task, fd_cols,
                                   parse_units=not args.no_parse_units)
    sep = inp['sep']
    import csv
    for name in ('train', 'val'):
        inp[name].to_csv(os.path.join(run_dir, f'{name}.csv'), index=False, sep=sep,
                         quoting=csv.QUOTE_NONE, quotechar='\x07', escapechar=None)
    with open(os.path.join(run_dir, 'meta.csv'), 'w', encoding='utf-8') as fh:
        for row in inp['meta']:
            fh.write(sep.join(row) + '\n')
    with open(os.path.join(run_dir, 'config.xml'), 'w') as fh:
        fh.write(CONFIG_XML.format(tmp=os.path.join(run_dir, 'tmp'), scratch=os.path.join(run_dir, 'scratch'),
                                   par='true' if args.threads > 1 else 'false'))
    props = os.path.join(args.sysds_home, 'scripts', 'pipelines', 'properties')
    func = {'classification': 'evalClassification', 'regression': 'evalRegression',
            'clustering': 'evalClustering'}[search_task]
    # number of clusters = number of distinct labels, as in the common evaluator; taken from the
    # clean validation split (the objective data), so no clean value of a train row is read
    cluster_k = int(clean.iloc[va][label].replace(list(saga_core.MISS), np.nan).dropna().nunique()) \
        if task == 'clustering' else 2
    jar = [f for f in os.listdir(args.sysds_home) if f.startswith('systemds') and f.endswith('.jar')][0]
    cp = os.path.join(args.sysds_home, jar) + ':' + os.path.join(args.sysds_home, 'lib', '*')
    cmd = [args.java, f'-Xmx{args.xmx}', f'-Xms{args.xmx}', f'-XX:ActiveProcessorCount={args.threads}',
           '-XX:ParallelGCThreads=1', '-XX:CICompilerCount=2',
           f'-Djava.io.tmpdir={os.path.join(run_dir, "tmp")}',
           f'-Dlog4j.configuration=file:{os.path.join(args.sysds_home, "conf", "log4j.properties")}',
           '-cp', cp, 'org.apache.sysds.api.DMLScript', '-exec', 'singlenode',
           '-config', os.path.join(run_dir, 'config.xml'),
           '-f', os.path.join(METHOD_DIR, 'saga_topk.dml'), '-stats', '-nvargs',
           f'sep={sep}', f'train={os.path.join(run_dir, "train.csv")}', f'val={os.path.join(run_dir, "val.csv")}',
           f'meta={os.path.join(run_dir, "meta.csv")}', f'primitives={os.path.join(props, "primitives.csv")}',
           f'parameters={os.path.join(props, "param.csv")}', f'out={out_dir}', f'func={func}',
           f'topk={args.topk}', f'rv={args.rv}', f'max_iter={args.max_iter}', f'sample={args.sample}',
           f'expectedIncrease={args.expected_increase}', f'seed={args.seed}',
           f'evalMaxi={args.eval_maxi}', f'evalMaxii={args.eval_maxii}',
           f'clusterK={cluster_k}', f'silN={args.sil_n}']
    with open(os.path.join(run_dir, 'cmd.json'), 'w') as fh:
        json.dump(cmd, fh, indent=1)
    t0 = time.time()
    with open(os.path.join(run_dir, 'systemds.log'), 'w') as log:
        rc = subprocess.call(cmd, cwd=args.sysds_home, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    secs = time.time() - t0
    if rc != 0 or not os.path.exists(os.path.join(out_dir, 'exp_pipeline.csv')):
        raise SystemExit(f'{ds}: SystemDS exited with {rc}, see {run_dir}/systemds.log')

    pip = read_sysds_csv(os.path.join(out_dir, 'exp_pipeline.csv'), dtype=str).iloc[0].tolist()
    ops = [p for p in pip if p not in ('', '0', 'NaN')]
    X0 = read_sysds_csv(os.path.join(out_dir, 'exp_X0.csv'))
    steps = [read_sysds_csv(os.path.join(out_dir, f'exp_X{i}.csv')) for i in range(1, len(ops) + 1)]
    cleaned, info = saga_core.decode_export(ops, steps, X0, dirty_tr, feature_cols, label, cats)
    ordered = [c for c in dirty.columns if c in cleaned.columns]
    cleaned = cleaned[ordered]
    mn = args.method_name
    res_dir = os.path.join(args.result_root, mn, f'{ds}_{mn}')
    os.makedirs(res_dir, exist_ok=True)
    res_csv = os.path.join(res_dir, f'{ds}_{mn}_cleaned.csv')
    cleaned.to_csv(res_csv, index=False)

    topk_pip = read_sysds_csv(os.path.join(out_dir, 'pip.csv'), dtype=str).values.tolist()
    with open(os.path.join(run_dir, 'systemds.log'), errors='replace') as fh:
        log_stats = parse_systemds_log(fh.read())
    deviations = deviation_list(task, search_task, label, inp['stats'], vars(args), cluster_k, dropped_unlabeled)
    rec = {
        'dataset': ds, 'method': mn, 'task': task, 'search_task': search_task,
        'n_train': int(n_train_split), 'n_train_used': int(len(dirty_tr)), 'n_val': int(len(clean_va)),
        'n_test': int(len(te)), 'gt_cells': 0, 'secs': round(secs, 1),
        'search_secs': log_stats.get('search_secs'), 'search_log': log_stats,
        'top1_pipeline': ops,
        'topk_pipelines': [[p for p in r if p not in ('', '0', 'NaN')] for r in topk_pip],
        'topk_val_scores': read_sysds_csv(os.path.join(out_dir, 'bestAcc.csv'))[:, 0].tolist(),
        'dirty_val_score': float(read_sysds_csv(os.path.join(out_dir, 'dirtyScore.csv'))[0, 0]),
        'top1_full_train_val_score': float(read_sysds_csv(os.path.join(out_dir, 'exp_fullValScore.csv'))[0, 0]),
        'eval_hp': read_sysds_csv(os.path.join(out_dir, 'evalHp.csv'))[0].tolist(),
        'export': info, 'changed_cells': saga_core.changed_cells(cleaned, dirty_tr, feature_cols, cats),
        'input_stats': inp['stats'], 'cleaned_csv': res_csv, 'deviations': deviations,
        'gt_note': 'no ground-truth cell of the train split is read; the clean validation split is the search objective',
        'config': {'topk': args.topk, 'rv': args.rv, 'max_iter': args.max_iter, 'sample': args.sample,
                   'expectedIncrease': args.expected_increase, 'seed': args.seed, 'cv': False,
                   'eval_maxi': args.eval_maxi, 'eval_maxii': args.eval_maxii,
                   'enablePruning': True, 'eval_func': func, 'xmx': args.xmx, 'threads': args.threads,
                   'cluster_k': cluster_k, 'sil_n': args.sil_n, 'parse_units': not args.no_parse_units},
        'cmd': ' '.join(cmd),
    }
    if args.records:
        with open(args.records, 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({k: rec[k] for k in ('dataset', 'secs', 'top1_pipeline', 'dirty_val_score',
                                          'topk_val_scores', 'top1_full_train_val_score', 'changed_cells')}))


if __name__ == '__main__':
    main()
