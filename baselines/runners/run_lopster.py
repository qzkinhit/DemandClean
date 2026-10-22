#!/usr/bin/env python
"""Run the fixed Lopster protocol, prepare inputs, or apply its output transformations.

The final configuration comes from --dataset/--budget-level or --result-config. Postprocessing
and preparation need no upstream code, TensorFlow, model training, or downstream evaluation.
"""
import argparse
import functools
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from lopster_protocol import (SCHEMAS, apply_postprocessing, clean_training_copy,
                              load_configuration, read_table, training_rows)

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_URL = 'https://github.com/DataManagementLab/data_cleaning_with_latent_operators'
MODULES = ('datasets.py', 'latent_operators.py', 'utils.py', 'error_detection.py', 'preprocess.py',
           'latent_operator_train_loop.py', 'transformation_in_x.py', 'plotter.py', 'cleaners.py')


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_table(path, frame):
    raw = frame.to_csv(index=False).encode()
    if str(path).endswith('.gz'):
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode='wb', mtime=0) as stream:
            stream.write(raw)
        write_bytes(path, compressed.getvalue())
    else:
        write_bytes(path, raw)
    return raw


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_backend(args, cfg, source_root, output_dir):
    upstream = Path(args.upstream_root).resolve()
    required = MODULES[:4]
    missing = [name for name in required if not (upstream / name).is_file()]
    if missing:
        raise FileNotFoundError('Lopster upstream files are required: ' + ', '.join(missing)
                                + '. Obtain them from ' + UPSTREAM_URL
                                + ' and follow baselines/Methods/Lopster/LICENSE_NOTE.md.')
    backend = output_dir / 'backend' / 'Methods' / 'Lopster'
    if backend.exists():
        raise FileExistsError('the backend directory already exists; choose a new task name for a fresh run')
    backend.mkdir(parents=True)
    for name in MODULES:
        if (upstream / name).is_file():
            shutil.copy2(str(upstream / name), str(backend / name))
    write_bytes(backend / 'dataset_configuration.json',
                (json.dumps({cfg['dataset']: SCHEMAS[cfg['dataset']]}, indent=2) + '\n').encode())
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[name] = str(cfg['threads'])
    legacy = load_module('_lopster_input_adapter', Path(__file__).parent / 'run_lopster' / 'run_lopster_base.py')
    legacy.PROJECT_ROOT = str(backend.parents[1])
    info = legacy.prepare_lopster_dataset(str(source_root), cfg['dataset'], str(output_dir))
    wrapper = load_module('_lopster_release_wrapper', ROOT / 'baselines/Methods/Lopster/lopster_wrapper.py')
    wrapper._current_dir = str(backend)
    previous = Path.cwd()
    sys.path.insert(0, str(backend))
    try:
        os.chdir(str(backend))
        import datasets
        import utils
        for function in (datasets.load_regression, datasets.load_features_and_data):
            if 'path_to_dataset' not in inspect.signature(function).parameters:
                raise RuntimeError('the upstream data loader needs the path_to_dataset adapter described in LICENSE_NOTE.md')
        # The legacy wrapper accepts the learning-rate argument but calls the upstream default.
        # Bind it explicitly so the recorded value (0.001 for the paper) is honored.
        original_train = utils.create_and_train_LOP
        utils.create_and_train_LOP = functools.partial(original_train, lr=cfg['learning_rate'])
        try:
            cleaner = wrapper.LopsterWrapper(latent_dim=cfg['latent_dim'], learning_rate=cfg['learning_rate'],
                batch_size=cfg['batch_size'], epochs=cfg['epochs'], K=cfg['K'],
                clean_ratio=cfg['clean_ratio'], verbose=args.verbose)
            raw_path = output_dir / 'raw_cleaned.csv'
            cleaner.clean(cfg['dataset'], str(Path(info['lopster_data_dir']).parent), str(raw_path))
        finally:
            utils.create_and_train_LOP = original_train
    finally:
        os.chdir(str(previous))
        sys.path.remove(str(backend))
    # The legacy adapter restores index, identifier/date columns and column ordering.
    legacy.postprocess_lopster_output(str(raw_path), info, str(raw_path))
    return read_table(raw_path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', choices=sorted(SCHEMAS))
    ap.add_argument('--result-config', help='a final run.json or manifest.json')
    ap.add_argument('--budget-level', choices=['main', 'dcbudget', 'ample'], default='main')
    ap.add_argument('--data_path', '--data-root', dest='data_path', default=str(ROOT / 'data'))
    ap.add_argument('--clean_path', help='optional indexed clean reference file')
    ap.add_argument('--dirty-input', help='explicit indexed training input for postprocessing/preparation')
    ap.add_argument('--output_path', '--output-root', dest='output_path', default=str(ROOT / 'results/runs/lopster'))
    ap.add_argument('--task_name', '--run-name', dest='task_name')
    ap.add_argument('--out', help='output CSV for --postprocess-only or an input-passthrough run')
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--postprocess-only', metavar='CSV', help='apply final transformations to an indexed raw export')
    modes.add_argument('--prepare-only', action='store_true', help='write only the sampled training inputs')
    ap.add_argument('--check', metavar='CSV', help='compare the resulting CSV bytes with a saved CSV or CSV.gz')
    ap.add_argument('--dry-run', action='store_true', help='print the fixed configuration and exit without writing')
    ap.add_argument('--upstream-root', default=str(ROOT / 'baselines/Methods/Lopster'))
    ap.add_argument('--latent_dim', type=int)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--learning_rate', type=float)
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--K', type=int)
    ap.add_argument('--clean_ratio', type=float)
    ap.add_argument('--threads', type=int)
    ap.add_argument('--index_attribute', default='index')
    ap.add_argument('--label_column')
    ap.add_argument('--use_split', action='store_true', help='accepted for compatibility; the shared split is always used')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.index_attribute != 'index':
        ap.error('the released protocol aligns rows using the index column')
    if args.check and not args.postprocess_only:
        ap.error('--check requires --postprocess-only')
    return args


def main():
    args = parse_args()
    try:
        cfg = load_configuration(args.dataset, args.budget_level, args.result_config)
        for field in ('latent_dim', 'epochs', 'learning_rate', 'K', 'clean_ratio', 'threads'):
            if getattr(args, field) is not None:
                cfg[field] = getattr(args, field)
        if args.label_column and args.label_column != cfg['label_column']:
            raise ValueError('--label_column differs from the dataset configuration')
        cfg['batch_size'] = (args.batch_size if args.batch_size is not None else
                             max(1, min(256, int(0.7 * cfg['clean_records']) - 1)))
        if cfg['clean_ratio'] != 1.0:
            raise ValueError('the fixed protocol uses the entire sampled clean copy; clean_ratio must be 1.0')
        if min(cfg['epochs'], cfg['latent_dim'], cfg['K'], cfg['batch_size'], cfg['threads']) < 1:
            raise ValueError('training dimensions, counts, and threads must be positive')
        if args.dry_run:
            print(json.dumps(cfg, indent=2))
            return
        ds = cfg['dataset']
        full = read_table(Path(args.data_path) / ds / 'dirty_index.csv') if not args.dirty_input else None
        dirty = read_table(args.dirty_input) if args.dirty_input else training_rows(full)
        if args.postprocess_only:
            if cfg['clean_records'] == 0:
                result = full if full is not None else dirty
                collapsed = []
            else:
                result, collapsed = apply_postprocessing(read_table(args.postprocess_only), dirty,
                    cfg['label_column'], cfg['preserve_label'], cfg['constant_column_fallback'])
            raw = result.to_csv(index=False).encode()
            summary = {'dataset': ds, 'rows': len(result), 'restored_columns': collapsed,
                       'sha256': hashlib.sha256(raw).hexdigest(), 'configuration': cfg}
            if args.out:
                write_table(args.out, result)
            if args.check:
                expected = Path(args.check).read_bytes()
                if str(args.check).endswith('.gz'):
                    expected = gzip.decompress(expected)
                summary['matches_saved_output'] = expected == raw
                if expected != raw:
                    raise ValueError('postprocessed CSV differs from the saved output')
            if not args.out and not args.check:
                raise ValueError('supply --out or --check with --postprocess-only')
            print(json.dumps(summary, indent=2))
            return
        output_dir = Path(args.output_path).resolve() / (args.task_name or (ds + '_' + cfg['budget_level']))
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if cfg['clean_records'] == 0:
            result = full if full is not None else dirty
            write_table(args.out or (output_dir / 'cleaned.csv'), result)
            cfg['mode'] = 'input_passthrough'
        else:
            clean = read_table(args.clean_path or (Path(args.data_path) / ds / 'clean_index.csv'))
            chosen = clean_training_copy(clean, dirty, cfg)
            inputs = output_dir / 'inputs' / ds
            write_table(inputs / 'dirty_index.csv', dirty)
            write_table(inputs / 'clean_index.csv', chosen)
            cfg['ground_truth_cells'] = len(chosen) * (len(chosen.columns) - 1)
            if args.prepare_only:
                cfg['status'] = 'inputs_prepared'
            else:
                result = run_backend(args, cfg, inputs.parent, output_dir)
                result, collapsed = apply_postprocessing(result, dirty, cfg['label_column'],
                    cfg['preserve_label'], cfg['constant_column_fallback'])
                write_table(args.out or (output_dir / 'cleaned.csv'), result)
                cfg['restored_columns'] = collapsed
                cfg['status'] = 'completed'
        cfg['elapsed_seconds'] = time.time() - started
        write_bytes(output_dir / 'run.json', (json.dumps(cfg, indent=2) + '\n').encode())
        print(json.dumps(cfg, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main()
