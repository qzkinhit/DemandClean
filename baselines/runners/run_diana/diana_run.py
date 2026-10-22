#!/usr/bin/env python
"""Official DIANA KB recommendations and operators, evaluated under frozen paper protocol.

    DIANA_SOURCE=<clone of camillasancricca/diana-repo> SCORER_PYTHON=<python with scikit-learn 1.3.2> \
        <python of the DIANA environment> baselines/runners/run_diana/diana_run.py --dataset beers

The run writes results/runs/diana/<dataset>/; diana_rescore.py then stores the cleaned training
matrix as cleaned_encoded.npz in results/runs/diana_clean/<dataset>/ and scores it.
"""
import os, sys, json, time, argparse, traceback, math, random, subprocess, hashlib
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import pandas as pd
# Official model snapshots were serialized with NumPy 2's private module spelling.
import numpy.core as _numpy_core
sys.modules.setdefault('numpy._core', _numpy_core)
for _part in ('multiarray','numeric','_multiarray_umath'):
    sys.modules.setdefault('numpy._core.'+_part, __import__('numpy.core.'+_part,fromlist=[_part]))
ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0,str(ROOT/'baselines'/'runners'))
import common
BASE=ROOT/'results'/'runs'
SOURCE=Path(os.environ.get('DIANA_SOURCE',str(ROOT/'baselines'/'Methods'/'DIANA'/'diana-repo')))
SCORER=os.environ.get('SCORER_PYTHON',sys.executable)
UPSTREAM=dict(repository='https://github.com/camillasancricca/diana-repo',commit='88e9b54161453893b93c679939206b420a472e1a')

def run(ds):
    out=BASE/os.environ.get('DIANA_RESULT_TAG','diana')/ds;out.mkdir(parents=True,exist_ok=True)
    b=common.load_bundle(ds)
    meta=dict(dataset=ds,method='DIANA',gt_cells=0,gt_cap_cells=b['gt_cap_cells'],started=time.time(),pid=os.getpid())
    if b['task']=='regression':
        median=float(np.nanmedian(b['y_train']))
        surrogate=(np.asarray(b['y_train'])>median).astype(int)
        meta['task_adaptation']=dict(original_task='regression',recommendation_model='RandomForest classification',surrogate='dirty training target above training median',threshold=median,counts=np.bincount(surrogate).tolist(),note='Official KB consumes data profiles and model family, not target labels. Final evaluation remains original regression.')
    elif b['task']=='clustering':
        meta['task_adaptation']=dict(original_task='clustering',recommendation_model='RandomForest classification',surrogate='dirty training activity labels',note='Official KB consumes data profiles and model family. Final evaluation remains original agglomerative clustering silhouette.')
    src=SOURCE;scripts=src/'running_example'/'scripts'
    meta['upstream']=dict(UPSTREAM)
    os.chdir(str(scripts));sys.path.insert(0,str(scripts))
    import kb_suggestions as kb
    import numerical as num
    import data_profile_extraction as dp
    import imputation as imp
    import outlier_detection as od
    np.random.seed(42);random.seed(42)
    # These wrappers replace repeated row scans with algebraically identical ordered counts.
    def probs(df,col,drop):
        s=df[col];den=s.notna().sum() if drop else len(s)
        if den==0:return np.array([]),0
        counts=s.value_counts(sort=False,dropna=drop)
        vals=[]
        for x in s.unique():
            if pd.isna(x):
                if not drop:vals.append(0.)
            else:vals.append(float(counts.loc[x])/den)
        return np.asarray(vals),s.nunique()
    def entropy_num(df,col):
        p,_=probs(df,col,True);p=p[p>0];return float(-(p*np.log(p)).sum())
    def entropy_dp(df,col):
        p,_=probs(df,col,False);p=p[p>0];return float(-(p*np.log(p)).sum())
    def density(df,col,drop):
        p,n=probs(df,col,drop)
        if n==0:return np.nan
        val=0.
        for x in p:val=(val+abs(x-1./n))/n
        return val*100
    num.entropy=entropy_num;num.density=lambda df,col:density(df,col,True)
    dp.entropy=entropy_dp;dp.density=lambda df,col:density(df,col,False)
    native_column_profile=num.get_features_num
    def finite_column_profile(*args,**kwargs):
        # Match official extract_suggestion_ranking's nan_to_num convention for
        # undefined skewness/correlation of constant columns in per-column KBs.
        return tuple(np.nan_to_num(native_column_profile(*args,**kwargs),nan=0.,posinf=0.,neginf=0.))
    num.get_features_num=finite_column_profile
    name=({'rf':'RandomForest','lr':'LogisticRegression','dt':'DecisionTree','knn':'KNN','gb':'AdaBoost','svm':'SVC'}[b['model']] if b['task']=='classification' else 'RandomForest')
    X=pd.DataFrame(np.asarray(b['X_train_missing'],float),columns=b['features'])
    original=X.copy()
    empty=[c for c in X if not X[c].notna().any()]
    # No empirical distribution exists for these columns. Keep the canonical
    # dimensions, use the official standard imputer, and exclude only their
    # undefined profiles from the pretrained KB recommendation call.
    for c in empty:X=imp.impute(X,'impute_standard',c)
    meta['all_missing_columns_standard_imputed']=empty
    features=[c for c in X if c not in empty];kb.models=[name]
    print('DIANA_START',ds,X.shape,'model',name,flush=True)
    profile_frame=X[features]
    accuracy=sum(len(od.ZSB(profile_frame,c)) for c in features)
    # Preserve the exact aggregate supplied to official running_example validation.
    quality=round((100-(accuracy/profile_frame.shape[0]*profile_frame.shape[1])*100)+(profile_frame.notnull().sum().sum()/profile_frame.shape[0]*profile_frame.shape[1])*100,2)
    rank=kb.extract_suggestion_ranking(profile_frame,features,quality)[name]
    print('RANK',rank,flush=True)
    complete=kb.extract_suggestion_completeness(profile_frame,features)[name]
    print('COMPLETENESS',complete,flush=True)
    accuracy=kb.extract_suggestion_accuracy(profile_frame,features)
    plan=dict(first_dimension=rank,completion=complete,accuracy=accuracy,feature_order=features,model=name,all_missing_standard_imputed=empty)
    (out/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
    print('ACCURACY',accuracy,flush=True)
    def completeness(frame):
        for c in features:
            if frame[c].isna().any():
                print('IMPUTE',c,complete[c],int(frame[c].isna().sum()),flush=True)
                frame=imp.impute(frame,complete[c],c)
        return frame
    def improve_accuracy(frame):
        # Official example obtains all column outlier sets before modifying any column.
        detected={c:od.outliers(frame,accuracy[c],c) for c in features}
        for c,rows in detected.items():frame.loc[rows,c]=np.nan
        return completeness(frame)
    if rank=='completeness':X=completeness(X)
    X=improve_accuracy(X)
    if not np.array_equal(X.index.to_numpy(),original.index.to_numpy()):raise ValueError('Unexpected row reordering by official DIANA operator')
    np.savez_compressed(out/'candidate_validation.npz',X=np.asarray(X,float),y=b['y_train'],Xv=b['X_val'])
    vp=subprocess.run([SCORER,str(ROOT/'baselines'/'runners'/'validation_bridge.py'),ds,str(out/'candidate_validation.npz')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    if vp.returncode:raise RuntimeError('canonical validation failed: '+vp.stdout[-2000:])
    val=float([x for x in vp.stdout.splitlines() if x.startswith('CANONICAL_VALIDATION ')][-1].split(' ',1)[1])
    (out/'validation.json').write_text(json.dumps({'score':val,'model':b['model']}))
    frozen=dict(plan_sha256=hashlib.sha256((out/'frozen_plan.json').read_bytes()).hexdigest(),train_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(X,float)).tobytes()).hexdigest(),validation_score=val,frozen_at=time.time(),test_features_loaded=False)
    (out/'frozen_artifact.json').write_text(json.dumps(frozen,indent=2))
    meta['frozen_artifact']=frozen
    meta.update(status='pipeline_frozen',validation_score=val,plan=plan,adaptations=['canonical dirty-train-fitted numeric representation; target labels unchanged','official KB model recommendations and official per-column impute/outlier functions','dataset feature order fixed to canonical feature order; no factorial permutation search from fixed seven-column demo','ordered-count optimization of entropy/density preserves original formulas','Python3.11/NumPy2/sklearn1.5.2 for native RF missing-value support; canonical final scorer remains sklearn1.3.2'],finished_cleaning=time.time())
    # Test features become accessible only after all recommendations are frozen and applied.
    test=common.load_frozen_test(ds)
    common.write_transformed(ds,out,X.to_numpy(float),b['y_train'],test['X_test'],b['idx_train'],test['idx_test'],'DIANA',metadata=meta,X_val=b['X_val'])
    scoring=subprocess.run([SCORER,str(ROOT/'baselines'/'runners'/'common.py'),'--dataset',ds,'--out',str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    print('SCORE',scoring.stdout,flush=True)
    if scoring.returncode:raise RuntimeError('canonical final scoring failed')
    meta.update(status='completed',finished=time.time());(out/'status.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args()
    try:run(args.dataset)
    except Exception as e:
        traceback.print_exc();out=BASE/os.environ.get('DIANA_RESULT_TAG','diana')/args.dataset;out.mkdir(parents=True,exist_ok=True)
        (out/'status.json').write_text(json.dumps(dict(dataset=args.dataset,method='DIANA',status='failed',error=repr(e),pid=os.getpid(),gt_cells=0,time=time.time()),indent=2));sys.exit(1)
