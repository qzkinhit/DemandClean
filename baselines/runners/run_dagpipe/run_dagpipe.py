#!/usr/bin/env python3
"""Author HierarchicalDAGPipe on the fixed main-table data protocol.

The author's preprocessing graph, operations, Adam, loss, sparsity, and DAG
penalties are reused.  Train-only operation statistics, held-out validation,
and a fixed main-table evaluation model are explicit protocol adaptations.

    DAGPIPE_SOURCE=<artifact directory of the DAGPipe Zenodo record> SCORER_PYTHON=<python with scikit-learn 1.3.2> \
        python baselines/runners/run_dagpipe/run_dagpipe.py --dataset beers --out results/runs/dagpipe/beers/seed0

Training on a GPU is not bitwise deterministic. The defaults train on one CPU thread with
single-threaded BLAS, the setting of the released outputs, which reproduces them byte for byte.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'  # single-threaded BLAS, set before numpy and torch are imported

ROOT = Path(__file__).resolve().parents[3]  # repository root
SOURCE = Path(os.environ.get('DAGPIPE_SOURCE', str(ROOT / 'baselines/Methods/DAGPipe/artifact')))
SOURCE_RECORD = dict(method='DAGPipe', source='https://zenodo.org/records/20492518',
                     source_file='https://zenodo.org/records/20492518/files/dagpipe-artifact.zip?download=1',
                     doi='10.5281/zenodo.20492518',
                     sha256='14d8f10f057cc2322328707888b7cf67cac11cdbddc43c8207d882efd418ad72')
SUPPORTED = ['breast_cancer', 'beers', 'flights', 'soccer', 'hospitals', 'smartfactory', 'adult']


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_protocol(path):
    spec = importlib.util.spec_from_file_location('baseline_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(model, array, device, chunk=256):
    import numpy as np
    import torch
    results = []
    dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        for start in range(0, len(array), chunk):
            value = model(torch.as_tensor(array[start:start+chunk], dtype=dtype, device=device))
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite frozen DAGPipe output')
            results.append(value.cpu().numpy())
    return np.concatenate(results, axis=0)


def train_regression(model, predictor, dataloader, optimizer, device):
    """Only the task head/loss differs from the author's classification train."""
    import torch
    from torch.nn import functional as F
    model.to(device).train()
    predictor.to(device).train()
    totals = [0., 0., 0., 0.]
    for x, y in dataloader:
        if len(x) == 1:
            continue
        predictions = predictor(model(x.to(device))).reshape(-1)
        mse = F.mse_loss(predictions, y.to(device=device, dtype=predictions.dtype))
        dag = .1 * model.dag_penalty() / (model.num_layers + 1)
        sparse = torch.sigmoid(model.layer_adj_logits).sum()
        for logits in model.intra_adj_logits:
            sparse = sparse + torch.sigmoid(logits).sum()
        sparse = .0001 * sparse
        loss = mse + dag + sparse
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
        optimizer.step()
        for index, value in enumerate([loss, mse, dag, sparse]):
            totals[index] += value.item()
    return tuple(value / len(dataloader) for value in totals)


def evaluate_regression(model, predictor, dataloader, device):
    import numpy as np
    import torch
    from sklearn.metrics import r2_score
    model.eval()
    predictor.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            predictions.append(predictor(model(x.to(device))).reshape(-1).cpu().numpy())
            targets.append(y.numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    mse = float(np.mean((predictions.astype(float) - targets) ** 2))
    return -float(r2_score(targets, predictions)), float(r2_score(targets, predictions)), mse


def run(args):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PROJ_ROOT'] = str(SOURCE)
    os.environ['PROJ_DIR'] = str(SOURCE)
    sys.path.insert(0, str(SOURCE))
    from dagpipe.models.dagpipe_hie import HierarchicalDAGPipe
    from models.operations import (MeanImputer, MedianImputer, KNNImputer,
                                   TargetEncoder, FrequencyEncoder, StandardScaler,
                                   RobustScaler, MinMaxScaler, QuantileScaler,
                                   LogScaler, PolynomialTransform)
    from dagpipe.training.trainer import train, evaluate
    from train_stats import TrainingStatisticsOperation, freeze_on_training_data, inference_statistics_manifest

    protocol = load_protocol(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter_text = Path(__file__).read_text()
    statistics_text = (Path(__file__).parent / 'train_stats.py').read_text()
    (out / 'adapter_used.py').write_text(adapter_text)
    (out / 'train_stats_used.py').write_text(statistics_text)
    started = time.time()
    bundle = protocol.load_bundle(args.dataset)
    initial = dict(dataset=args.dataset, method='DAGPipe', status='running', pid=os.getpid(),
                   seed=args.seed, started_at=started, split_seed=42,
                   gt_cells=0, gt_cap_cells=int(bundle['gt_cap_cells']),
                   adapter_sha256=hashlib.sha256(adapter_text.encode()).hexdigest(),
                   train_statistics_adapter_sha256=hashlib.sha256(statistics_text.encode()).hexdigest(),
                   protocol_sha256=hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
                   source=dict(SOURCE_RECORD))
    json_write(out / 'run_status.json', initial)
    json_write(out / 'gt_access.json', [])
    task = bundle['task']
    regression = task == 'regression'
    clustering_surrogate = task == 'clustering'
    if task not in ('classification', 'regression', 'clustering'):
        raise ValueError(f'Unknown task {task}')

    # Repeated author polynomial operations can make float32 clip-norm overflow
    # under MSE. Float64 is a fixed arithmetic repair for all regression tasks.
    numpy_dtype = np.float64 if regression else np.float32
    torch_dtype = torch.float64 if regression else torch.float32
    X = np.asarray(bundle['X_train'], dtype=numpy_dtype)
    original_y = np.asarray(bundle['y_train'], dtype=np.float64)
    valid_training = np.isfinite(original_y)
    X = X[valid_training]
    raw_y = original_y[valid_training]
    Xv = np.asarray(bundle['X_val'], dtype=numpy_dtype)
    raw_yv = np.asarray(bundle['y_val'], dtype=np.float64)
    label_scaler = None
    if regression:
        # Only dirty training targets determine these normalization statistics.
        labels = None
        target_mean = float(raw_y.mean())
        target_std = max(float(raw_y.std()), 1e-8)
        label_scaler = dict(mean=target_mean, std=target_std, fitted_on='dirty_training_targets')
        y = ((raw_y - target_mean) / target_std).astype(numpy_dtype)
        valid_validation = np.isfinite(raw_yv)
        yv = ((raw_yv[valid_validation] - target_mean) / target_std).astype(numpy_dtype)
    else:
        labels = np.unique(raw_y)
        if len(labels) < 2:
            raise ValueError('Dirty training split has fewer than two finite label classes')
        # HAR uses its dirty activity labels as an explicitly declared proxy.
        mapping = {float(value): index for index, value in enumerate(labels)}
        y = np.asarray([mapping[float(value)] for value in raw_y], dtype=np.int64)
        valid_validation = np.isfinite(raw_yv) & np.isin(raw_yv, labels)
        yv = np.asarray([mapping[float(value)] for value in raw_yv[valid_validation]], dtype=np.int64)
    Xv_selected = Xv[valid_validation]
    if len(Xv_selected) < 2:
        raise ValueError('Insufficient validation targets represented by dirty training labels')
    if not np.isfinite(X).all() or not np.isfinite(Xv).all():
        raise FloatingPointError('Shared protocol must supply finite train-fitted encoded/scaled inputs')

    device = torch.device(args.device)
    original_layers = [
        [MeanImputer(), MedianImputer(), KNNImputer()],
        [TargetEncoder(), FrequencyEncoder()],
        [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileScaler(), LogScaler(), PolynomialTransform()],
        [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileScaler(), LogScaler(), PolynomialTransform()],
    ]
    layers = [[TrainingStatisticsOperation(op) for op in layer] for layer in original_layers]
    preprocessor = HierarchicalDAGPipe(X.shape[1], layers).to(device=device, dtype=torch_dtype)
    predictor = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1 if regression else len(labels))).to(device=device, dtype=torch_dtype)
    optimizer = torch.optim.Adam([
        {'params': preprocessor.parameters(), 'lr': .01},
        {'params': predictor.parameters(), 'lr': .01},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xv_selected), torch.from_numpy(yv)),
                            batch_size=256, shuffle=False, num_workers=0)
    X_tensor = torch.as_tensor(X, dtype=torch_dtype, device=device)
    best_loss = float('inf')
    best_epoch = None
    best_model = None
    best_predictor = None
    stale = 0
    epoch_records = []
    print(json.dumps({'event':'training_started','dataset':args.dataset,'device':str(device),
                      'n_train':len(X),'n_val':len(Xv_selected),'features':X.shape[1],
                      'dirty_label_classes':None if regression else len(labels),
                      'task':task, 'clustering_supervised_surrogate':clustering_surrogate,
                      'label_scaler':label_scaler,'config':vars(args)}), flush=True)
    for epoch in range(args.epochs):
        for layer in preprocessor.op_layers:
            for op in layer:
                op.frozen = False
                op.statistics = {}
        epoch_start = time.time()
        if regression:
            loss, cls, dag, sparse = train_regression(preprocessor, predictor, loader, optimizer, device)
        else:
            loss, cls, dag, sparse = train(preprocessor, predictor, loader, optimizer,
                                          lambda_dag=.1, lambda_sparse=.0001,
                                          clip_grad_norm=1., device=device, with_acyc=True)
        if not np.isfinite([loss, cls, dag, sparse]).all():
            raise FloatingPointError(f'Non-finite official training loss at epoch {epoch}')
        # Clean validation is scored only after caches are fitted on dirty train.
        freeze_on_training_data(preprocessor, X_tensor)
        if regression:
            val_loss, val_accuracy, val_mse = evaluate_regression(preprocessor, predictor, val_loader, device)
        else:
            val_loss, val_accuracy = evaluate(preprocessor, predictor, val_loader,
                                              lambda_dag=.1, lambda_sparse=.0001, device=device)
        if not np.isfinite([val_loss, val_accuracy]).all():
            raise FloatingPointError(f'Non-finite validation result at epoch {epoch}')
        scheduler.step(val_loss)
        record = dict(epoch=epoch, train_loss=loss, task_loss=cls,
                      dag_loss=dag, sparsity_loss=sparse, validation_loss=val_loss,
                      seconds=time.time()-epoch_start)
        if regression:
            record.update(training_objective='MSE_on_standardized_dirty_targets', validation_r2=val_accuracy,
                          validation_mse=val_mse, selection_value='negative_validation_r2')
        else:
            record.update(classification_loss=cls, validation_accuracy=val_accuracy)
        epoch_records.append(record)
        with (out/'epochs.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(preprocessor.state_dict())
            best_predictor = copy.deepcopy(predictor.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= args.patience or epoch == args.epochs-1:
            print(json.dumps({'event':'epoch','dataset':args.dataset,**record,'best_epoch':best_epoch}), flush=True)
        if stale >= args.patience:
            break

    preprocessor.load_state_dict(best_model)
    predictor.load_state_dict(best_predictor)
    frozen_training = freeze_on_training_data(preprocessor, X_tensor).cpu().numpy()
    frozen_validation = transform(preprocessor, Xv, device)
    frozen_at = time.time()
    torch.save({'preprocessor': preprocessor.state_dict(), 'predictor': predictor.state_dict(),
                'statistics': [[op.statistics for op in layer] for layer in preprocessor.op_layers],
                'labels':labels, 'label_scaler':label_scaler, 'best_epoch':best_epoch, 'seed':args.seed}, out/'frozen_pipeline.pt')
    json_write(out/'frozen.json',dict(frozen_at=frozen_at,best_epoch=best_epoch,best_validation_loss=best_loss,
                                     inference_statistics=inference_statistics_manifest(preprocessor),
                                     test_loaded=False, gt_cells=0))
    # Test feature access happens strictly after checkpoint/selection are frozen.
    test = protocol.load_frozen_test(args.dataset)
    frozen_test = transform(preprocessor, np.asarray(test['X_test'], dtype=numpy_dtype), device)
    metadata = dict(author_implementation='dagpipe.models.dagpipe_hie.HierarchicalDAGPipe',
                    author_training='dagpipe.training.trainer.train/evaluate',
                    author_source_sha256=initial['source']['sha256'],
                    adaptations=[
                        'Fixed canonical 60/20/20 split, train-fitted shared encoders/scaler',
                        'Dirty training labels only, no ground-truth training cells',
                        'Separate clean validation selects epoch; the author runner builds val_loader from X_train',
                        'Each official operator uses original forward during training; dirty-train-only cached statistics at validation/test',
                        'KNN inference reference is dirty train, empirical CDF/frequency and log-domain clipping fixed from train',
                        'Frozen DAG representations are scored with the common main-table model rather than the author MLP head',
                    ], epochs_run=len(epoch_records), best_epoch=best_epoch, seed=args.seed,
                    search_objective=('MSE_with_original_DAG_and_sparsity_penalty; checkpoint_selected_by_validation_R2'
                                      if regression else 'official_MLP_cross_entropy_with_DAG_and_sparsity_penalty_on_validation'),
                    final_evaluation_model=bundle['model'], frozen_at=frozen_at,
                    task=task, task_extension=('regression_linear_output_MSE' if regression else
                                              'clustering_dirty_activity_label_supervised_surrogate' if clustering_surrogate else 'native_classification'),
                    native_task_support=not (regression or clustering_surrogate),
                    label_scaler=label_scaler,
                    arithmetic_dtype=str(torch_dtype),
                    labels_dropped_train=int((~valid_training).sum()),
                    labels_excluded_validation=int((~valid_validation).sum()),
                    test_feature_access_after_freeze=True, test_labels_accessed=False,
                    command=sys.argv, run_seconds=time.time()-started)
    if regression:
        metadata['adaptations'].append('Non-native regression extension changes only the head output to one real value and CE to MSE; dirty-train fitted target scaling; validation R2 selects epoch')
        metadata['adaptations'].append('Float64 regression arithmetic prevents gradient-norm overflow from repeated author polynomial operations under MSE; classification arithmetic remains original float32')
        metadata['author_training'] = 'Original DAG/sparsity/Adam/clip logic, task loss extended from cross entropy to MSE'
    if clustering_surrogate:
        metadata['adaptations'].append('Non-native HAR clustering uses dirty activity labels as supervised proxy; final result is canonical agglomerative silhouette, not proxy classification accuracy')
    protocol.write_transformed(args.dataset, out, frozen_training, raw_y, frozen_test,
                               np.asarray(bundle['idx_train'])[valid_training], test['idx_test'],
                               method='DAGPipe', gt_cells=0, metadata=metadata, X_val=frozen_validation)
    score_command=[args.score_python,args.protocol,'--dataset',args.dataset,'--out',str(out)]
    score_result=subprocess.run(score_command,capture_output=True,text=True,env=os.environ.copy())
    (out/'score.log').write_text(score_result.stdout+score_result.stderr)
    json_write(out/'score_command.json',score_command)
    if score_result.returncode:
        raise RuntimeError(f'Frozen DAGPipe trained and exported, canonical scoring failed: {score_result.stderr[-2000:]}')
    initial.update(status='completed',finished_at=time.time(),elapsed_seconds=time.time()-started,
                   best_epoch=best_epoch,epochs_run=len(epoch_records),
                   score=json.loads((out/'scores.json').read_text()))
    json_write(out/'run_status.json',initial)
    print(json.dumps({'event':'completed',**initial}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--protocol',default=str(ROOT/'baselines'/'runners'/'common.py'))
    ap.add_argument('--score-python',default=os.environ.get('SCORER_PYTHON',sys.executable))
    ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    try:
        run(args)
    except Exception as error:
        out=Path(args.out)
        out.mkdir(parents=True,exist_ok=True)
        old=json.loads((out/'run_status.json').read_text()) if (out/'run_status.json').exists() else {}
        old.update(dataset=args.dataset,method='DAGPipe',status='failed',pid=os.getpid(),
                   error=f'{type(error).__name__}: {error}',finished_at=time.time(),gt_cells=0)
        json_write(out/'run_status.json',old)
        (out/'failure_traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        raise SystemExit(1)


if __name__=='__main__':
    main()
