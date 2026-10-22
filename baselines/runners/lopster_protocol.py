"""Fixed input and output transformations for the released Lopster configurations."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN_RECORDS = {
    'adult': 60, 'beers': 42, 'breast_cancer': 20, 'smartfactory': 54,
    'bike': 170, 'mercedes': 85, 'nasa': 97, 'soilmoisture': 20, 'har': 80,
    'flights': 0, 'soccer': 0, 'hospitals': 0,
}
AMPLE_RECORDS = {
    'adult': 13566, 'beers': 723, 'breast_cancer': 209, 'smartfactory': 7093,
    'bike': 5213, 'mercedes': 1262, 'nasa': 450, 'soilmoisture': 203, 'har': 21000,
}
KEEP_LABEL = {'adult', 'beers', 'breast_cancer', 'mercedes', 'soilmoisture'}
KEEP_LABEL_AMPLE = {'beers', 'breast_cancer', 'soilmoisture'}


def _schema(target, cat=(), ids=(), dates=(), floats=(), negatives=()):
    return dict(id_cols=list(ids), cat_cols=list(cat), date_cols=list(dates),
                target=target, float_cols=list(floats), allow_negatives=list(negatives))


# Dataset facts supplied to the upstream loader; no upstream source code is bundled here.
SCHEMAS = {
    'adult': _schema('income', ['workclass', 'education', 'marital_status', 'occupation',
                               'relationship', 'gender', 'native_country', 'race']),
    'beers': _schema('style', ['city', 'state', 'brewery_name', 'style'],
                     ['brewery_id', 'beer_name'], floats=['abv', 'ibu', 'ounces']),
    'breast_cancer': _schema('class'),
    'smartfactory': _schema('labels', negatives=['i_w_bhr_weg', 'i_w_blo_weg', 'i_w_bru_weg',
                                                'i_w_hr_weg', 'i_w_hl_weg', 'i_w_bhl_weg']),
    'bike': _schema('cnt', dates=['dteday'], floats=['temp', 'atemp', 'hum', 'windspeed']),
    'mercedes': _schema('y', ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], floats=['y']),
    'nasa': _schema('sound_pressure_level', floats=['frequency', 'angle', 'chord_length',
                                                  'velocity', 'thickness', 'sound_pressure_level']),
    'soilmoisture': _schema('soil_moisture', dates=['datetime']),
    'har': _schema('gt', ['gt'], floats=['x', 'y', 'z'], negatives=['x', 'y', 'z']),
    'flights': _schema('arrival_delay_bucket', ['src', 'flight', 'sched_dep_time', 'act_dep_time',
                                               'sched_arr_time', 'act_arr_time', 'arrival_delay_bucket']),
    'soccer': _schema('manager', ['name', 'surname', 'birthplace', 'position', 'team', 'city',
                                 'stadium', 'manager'], floats=['birthyear', 'season']),
    'hospitals': _schema('Condition', ['HospitalName', 'HospitalOwner', 'City', 'State', 'Address1',
                                      'CountyName', 'EmergencyService', 'Condition', 'MeasureName',
                                      'HospitalType', 'MeasureCode', 'Stateavg', 'Score', 'Sample'],
                        ['ProviderNumber', 'Address2', 'Address3'], floats=['PhoneNumber', 'ZipCode']),
}


def read_table(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [c.strip().strip('\ufeff') for c in frame.columns]
    if 'index' not in frame or frame['index'].duplicated().any():
        raise ValueError('an indexed table with unique row identifiers is required')
    return frame


def training_rows(frame):
    indices, _ = train_test_split(np.arange(len(frame)), test_size=0.4, random_state=42)
    return frame.iloc[indices].reset_index(drop=True)


def load_configuration(dataset=None, level='main', result_config=None):
    """Read final settings only. This function never reads a score or compares variants."""
    obj = {}
    manifest = {}
    if result_config:
        path = Path(result_config)
        obj = json.loads(path.read_text())
        supplied_dataset = obj.get('dataset')
        if dataset and supplied_dataset and dataset != supplied_dataset:
            raise ValueError('dataset and result configuration disagree')
        dataset = dataset or supplied_dataset
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else obj
        if 'ample' in path.parent.name:
            level = 'ample'
        elif 'dcbudget' in path.parent.name:
            level = 'dcbudget'
    if dataset not in SCHEMAS:
        raise ValueError('choose a supported dataset or a result configuration containing its name')
    if level == 'ample' and dataset not in AMPLE_RECORDS:
        raise ValueError('the ample-budget table contains the nine benchmark datasets')
    cfg = {
        'dataset': dataset, 'budget_level': level, 'split_seed': 42, 'sampling_seed': 0,
        'clean_records': (AMPLE_RECORDS if level == 'ample' else CLEAN_RECORDS)[dataset],
        'preserve_label': dataset in (KEEP_LABEL_AMPLE if level == 'ample' else KEEP_LABEL),
        'constant_column_fallback': True, 'label_column': SCHEMAS[dataset]['target'],
        'latent_dim': 120, 'epochs': 100, 'learning_rate': 0.001, 'K': 12, 'clean_ratio': 1.0,
        'threads': 1,
    }
    extra = obj.get('configuration', obj)
    for field in ['preserve_label', 'constant_column_fallback', 'latent_dim', 'epochs',
                  'learning_rate', 'K', 'clean_ratio', 'threads']:
        if field in extra:
            cfg[field] = extra[field]
    if 'clean_training_records' in extra:
        cfg['clean_records'] = int(extra['clean_training_records'])
    elif 'clean_records' in extra:
        cfg['clean_records'] = int(extra['clean_records'])
    if 'postprocessing' in obj:
        cfg['preserve_label'] = any('preserve the input target' in str(s).lower()
                                    for s in obj['postprocessing'])
    if obj.get('mode') == 'input_passthrough':
        cfg['clean_records'] = 0
    # The manifest records consumed whole-record GT, so this is a consistency check when the
    # input schema becomes available, not a new budget or a score-based choice.
    if manifest.get('ground_truth_cells') is not None:
        cfg['recorded_gt_cells'] = int(manifest['ground_truth_cells'])
    return cfg


def clean_training_copy(clean, dirty_train, config):
    """Select the recorded number of clean training rows using the fixed sampler."""
    count = int(config['clean_records'])
    if count < 0 or count > len(dirty_train):
        raise ValueError('clean-record budget is outside the training split')
    columns = list(dirty_train.columns)
    if set(columns) - set(clean.columns):
        raise ValueError('the clean reference lacks input columns')
    positions = np.sort(np.random.RandomState(config['sampling_seed']).choice(
        len(dirty_train), count, replace=False))
    chosen_ids = dirty_train['index'].iloc[positions]
    lookup = clean.set_index('index')
    result = lookup.loc[chosen_ids, [c for c in columns if c != 'index']].reset_index()[columns]
    consumed = count * (len(columns) - 1)
    if config.get('recorded_gt_cells') is not None and consumed != config['recorded_gt_cells']:
        raise ValueError('manifest GT cells disagree with the recorded whole-record configuration')
    return result


def apply_postprocessing(raw_output, dirty_train, label_column, preserve_label, guard=True):
    """Apply the final label-preservation and constant-column fallback, in that order."""
    output = raw_output.set_index('index').copy()
    dirty = dirty_train.set_index('index')
    if output.index.has_duplicates or dirty.index.has_duplicates:
        raise ValueError('row identifiers must be unique')
    if not output.index.isin(dirty.index).all():
        raise ValueError('the output contains rows outside the supplied training input')
    common = output.index.intersection(dirty.index)
    if preserve_label:
        if label_column not in output or label_column not in dirty:
            raise ValueError('the target column is required for label preservation')
        output.loc[common, label_column] = dirty.loc[common, label_column].values
    collapsed = []
    if guard:
        collapsed = [c for c in dirty.columns if c in output.columns
                     and output.loc[common, c].nunique() <= 1 < dirty.loc[common, c].nunique()]
        for column in collapsed:
            output.loc[common, column] = dirty.loc[common, column].values
    return output.reset_index(), collapsed
