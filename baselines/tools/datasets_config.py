"""
datasets_config.py - dataset configuration

Defines feature columns, target columns, excluded columns, and task types for each dataset.
All baseline scripts should reference this config to stay consistent.
"""

# Dataset configuration.
DATASETS_CONFIG = {
    'adult': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'income',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],  # no additional columns to exclude
        'feature_columns': [
            'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
        ],
        'rules_file': None,
        'description': 'UCI Adult Census Income dataset'
    },

    'beers': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'style',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': ['id', 'beer_name', 'brewery_id', 'brewery_name', 'city', 'state'],
        'feature_columns': ['ounces', 'abv', 'ibu'],
        'rules_file': None,
        'description': 'Craft Beers dataset'
    },

    'bike': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'cnt',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': ['dteday'],  # date column is not used as a feature
        'feature_columns': [
            'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'
        ],
        'rules_file': 'rules.txt',
        'description': 'Bike Sharing rental dataset'
    },

    'breast_cancer': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'class',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': [
            'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
        ],
        'rules_file': 'rules.txt',
        'description': 'Wisconsin Breast Cancer dataset'
    },

    'har': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'gt',
        'task_type': 'clustering',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': ['x', 'y', 'z'],
        'rules_file': 'rules.txt',
        'description': 'Human Activity Recognition dataset'
    },

    'mercedes': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'y',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': [],  # X0-X385 are all features
        'feature_columns': 'auto',  # auto-detect: every column except index and y
        'rules_file': 'rules.txt',
        'description': 'Mercedes-Benz vehicle test-time dataset'
    },

    'nasa': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'sound_pressure_level',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': ['frequency', 'angle', 'chord_length', 'velocity', 'thickness'],
        'rules_file': 'rules.txt',
        'description': 'NASA Airfoil Self-Noise dataset'
    },

    'smartfactory': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'labels',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': [
            'i_w_blo_weg', 'o_w_blo_power', 'o_w_blo_voltage',
            'i_w_bhl_weg', 'o_w_bhl_power', 'o_w_bhl_voltage',
            'i_w_bhr_weg', 'o_w_bhr_power', 'o_w_bhr_voltage',
            'i_w_bru_weg', 'o_w_bru_power', 'o_w_bru_voltage',
            'i_w_hr_weg', 'o_w_hr_power', 'o_w_hr_voltage',
            'i_w_hl_weg', 'o_w_hl_power', 'o_w_hl_voltage'
        ],
        'rules_file': 'rules.txt',
        'description': 'Smart Factory equipment status dataset'
    },

    'soilmoisture': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'soil_moisture',
        'task_type': 'regression',
        'index_column': 'index',
        'exclude_columns': ['datetime'],  # timestamp column is not used as a feature
        'feature_columns': 'auto',  # auto-detect: every column except index, datetime, soil_moisture
        'rules_file': 'rules.txt',
        'description': 'Hyperspectral Soil Moisture dataset'
    },
    'flights': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'arrival_delay_bucket',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': 'auto',  # src, flight, scheduled/actual dep & arr times
        'rules_file': 'rules.txt',
        'description': 'Flight schedule records (UniClean real-world; native errors)'
    },
    'soccer': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'manager',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': 'auto',  # name, surname, birthyear/place, position, team, city, stadium, season
        'rules_file': 'rules.txt',
        'description': 'Soccer player records (UniClean real-world; native errors)'
    },
    'hospitals': {
        'dirty_file': 'dirty_index.csv',
        'clean_file': 'clean_index.csv',
        'label_column': 'Condition',
        'task_type': 'classification',
        'index_column': 'index',
        'exclude_columns': [],
        'feature_columns': 'auto',  # provider/hospital/address/measure fields
        'rules_file': 'rules.txt',
        'description': 'US hospital quality-measure records (UniClean real-world; native errors)'
    }
}


def get_dataset_config(dataset_name: str) -> dict:
    """Return the config for the given dataset."""
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASETS_CONFIG.keys())}")
    return DATASETS_CONFIG[dataset_name]


def get_all_datasets() -> list:
    """Return all dataset names."""
    return list(DATASETS_CONFIG.keys())


def get_dataset_path(dataset_name: str, base_path: str = 'Data') -> dict:
    """Return the full paths for the given dataset."""
    config = get_dataset_config(dataset_name)
    import os
    return {
        'dirty_path': os.path.join(base_path, dataset_name, config['dirty_file']),
        'clean_path': os.path.join(base_path, dataset_name, config['clean_file']),
        'rules_path': os.path.join(base_path, dataset_name, config['rules_file']) if config['rules_file'] else None
    }


def print_dataset_info(dataset_name: str = None):
    """Print dataset information."""
    if dataset_name:
        config = get_dataset_config(dataset_name)
        print(f"\n=== {dataset_name} ===")
        print(f"  Description: {config['description']}")
        print(f"  Task type: {config['task_type']}")
        print(f"  Label column: {config['label_column']}")
        print(f"  Index column: {config['index_column']}")
        print(f"  Excluded columns: {config['exclude_columns']}")
        if config['feature_columns'] == 'auto':
            print(f"  Feature columns: auto-detect")
        else:
            print(f"  Feature columns: {config['feature_columns']}")
    else:
        print("\n=== All datasets ===")
        for name in get_all_datasets():
            config = DATASETS_CONFIG[name]
            print(f"  {name}: {config['task_type']}, label={config['label_column']}")


if __name__ == '__main__':
    # Quick config test.
    print_dataset_info()
    print("\n" + "=" * 50)
    print_dataset_info('beers')
