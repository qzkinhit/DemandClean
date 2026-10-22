"""
Lopster wrapper: data cleaning with latent operators.

Paper: Generalizable Data Cleaning of Tabular Data in Latent Space (VLDB 2024)
Code: https://github.com/DataManagementLab/data_cleaning_with_latent_operators

Ground truth used: Lopster trains its VAE on clean rows (clean.csv). The official code uses
all of clean.csv, clean_ratio sets the share used here.

Dependencies: tensorflow, keras, scikit-learn, pandas, numpy, matplotlib
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

# Put this directory on the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


class LopsterWrapper:
    """
    Wrapper that calls the official Lopster functions directly.

    Parameters
    - latent_dim: dimension of the latent space (120)
    - K: number of translation steps (12)
    - epochs: training epochs (100)
    - batch_size: batch size (256)
    - clean_ratio: share of the clean rows used for training (1.0)
    """

    def __init__(self,
                 latent_dim: int = 120,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 K: int = 12,
                 clean_ratio: float = 1.0,
                 verbose: bool = False):
        """
        Args:
            latent_dim: dimension of the latent space (official default 120)
            learning_rate: learning rate
            batch_size: batch size
            epochs: training epochs
            K: number of translation steps
            clean_ratio: share of the clean rows used (0.0 to 1.0)
            verbose: print progress information
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.K = K
        self.clean_ratio = min(max(clean_ratio, 0.01), 1.0)  # Clipped to [0.01, 1.0]
        self.verbose = verbose
        self.ground_truth_used = 0

    def _check_requirements(self) -> bool:
        """Check that TensorFlow is installed."""
        try:
            import tensorflow as tf
            if self.verbose:
                print(f"TensorFlow version: {tf.__version__}")
            return True
        except ImportError:
            raise ImportError(
                "Lopster requires TensorFlow. Install it with: pip install tensorflow\n"
                "See Methods/Lopster/requirements.txt"
            )

    def _check_data_format(self, path: str, dataset: str) -> bool:
        """Check that the data has the format of the official code."""
        clean_path = os.path.join(path, dataset, 'clean.csv')
        dirty_path = os.path.join(path, dataset, 'dirty01.csv')
        dirty_path_alt = os.path.join(path, dataset, 'dirty.csv')
        config_path = os.path.join(_current_dir, 'dataset_configuration.json')

        missing = []
        if not os.path.exists(clean_path):
            missing.append(f"clean.csv: {clean_path}")
        if not os.path.exists(dirty_path) and not os.path.exists(dirty_path_alt):
            missing.append(f"dirty01.csv or dirty.csv: {dirty_path}")
        if not os.path.exists(config_path):
            missing.append(f"dataset_configuration.json: {config_path}")

        if missing:
            raise FileNotFoundError(
                f"Lopsterofficial implementationrequires the following files:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )
        return True

    def clean(self,
              dataset: str,
              path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run Lopster through the official functions.

        Args:
            dataset: dataset name (directory {path}/{dataset}/)
            path: root directory of the datasets
            output_path: output path (optional)

        Returns:
            repaired table and run information
        """
        self._check_requirements()
        self._check_data_format(path, dataset)

        # Import TensorFlow and the official modules
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from datasets import (
            get_tf_database, load_regression, load_features_and_data,
            reverse_to_input_domain, get_date_columns
        )
        from latent_operators import LatentOperator
        from utils import create_and_train_LOP
        from error_detection import predict_on_enhanced

        # loadcleandata
        clean_path = os.path.join(path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path)
        total_clean_rows = len(clean_df)

        # Number of clean rows used
        actual_clean_rows = int(total_clean_rows * self.clean_ratio)
        self.ground_truth_used = actual_clean_rows

        if self.verbose:
            print(f"clean rows available: {total_clean_rows}")
            print(f"useratio: {self.clean_ratio:.1%}")
            print(f"clean rows used: {actual_clean_rows}")

        # Sample size
        sample_size = actual_clean_rows if self.clean_ratio < 1.0 else -1

        # 1. loadcleandatatrainingVAE
        MISSING_REPLACE = '3.0'
        T = 'missing_values'

        x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(
            dataset,
            sample_size,  # n_train_instances
            sample_size,  # n_test_instances
            True,
            normalize_sklearn=True,
            path_to_dataset=path
        )

        COLS = x_clean_train.shape[1]

        # 2. trainingVAE
        train_dataset = get_tf_database(x_clean_train, x_clean_train, self.batch_size)
        val_dataset = get_tf_database(x_clean, x_clean, self.batch_size)

        encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(
            train_dataset,
            val_dataset,
            COLS,
            self.latent_dim,
            self.K,
            1,
            T,
            epochs=self.epochs,
            model_name=dataset
        )

        # 3. Translation function
        @tf.function
        def _translate_all_columns_by_1(inputs):
            zs = inputs
            for column in range(zs.shape[0]):
                new_z = LOP.translate_operator(zs[column], 1)
                zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
            return zs

        def generate_cleaned_data(x_, Zs, Ks, decoder):
            xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis=1)), [1, 0, 2]))
            x_p = tf.where(Ks == 0, x_, xs)
            return x_p

        # 4. Clean the dirty data
        headers, target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(
            dataset,
            -1,  # usealldirtydata
            -1,
            MISSING_REPLACE,
            SCALER, CAT_ENCODER,
            True, MAX, MIN,
            normalize_sklearn=True,
            path_to_dataset=path
        )
        filtered_header = headers["filtered_header"]

        # 5. Run cleaning
        X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
        Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_all_columns_by_1)
        clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)

        # 6. Convertresult
        if hasattr(clean_csv, 'numpy'):
            clean_csv_np = clean_csv.numpy()
        else:
            clean_csv_np = np.array(clean_csv)

        dirty_data[filtered_header] = clean_csv_np
        lop_data = reverse_to_input_domain(dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
        lop_data = get_date_columns(dataset, lop_data, dirty_data)

        # 7. Empty values become "empty"
        lop_data = lop_data.fillna('empty')
        lop_data = lop_data.replace('', 'empty')

        # 8. Save results
        default_output = os.path.join(path, dataset, 'lopster.csv')
        lop_data.to_csv(default_output, index=False)

        if output_path and output_path != default_output:
            lop_data.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'total_clean_rows': total_clean_rows,
            'clean_ratio': self.clean_ratio,
            'method': 'Lopster',
            'type': 'data-oriented',
            'auto_level': 2,  # Uses training data (clean.csv)
            'latent_dim': self.latent_dim,
            'K': self.K,
            'epochs': self.epochs,
            'output_path': output_path or default_output,
            'note': f'Used {self.clean_ratio:.1%} of clean.csv for VAE training ({self.ground_truth_used} rows)'
        }

        return lop_data, info

    def get_ground_truth_cost(self) -> int:
        """Ground-truth cells used."""
        return self.ground_truth_used


def lopster_clean(dataset: str,
                  path: str,
                  output_path: str = None,
                  clean_ratio: float = 1.0,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function of the Lopster cleaner.

    Args:
        dataset: dataset name
        path: root directory of the datasets
        output_path: output path
        clean_ratio: share of the clean rows used (1.0 = all)
        **kwargs: passed to LopsterWrapper

    Returns:
        repaired table and run information
    """
    wrapper = LopsterWrapper(clean_ratio=clean_ratio, **kwargs)
    return wrapper.clean(dataset, path, output_path)


def prepare_data_for_lopster(dirty_csv: str, clean_csv: str, dataset_name: str, output_dir: str):
    """
    Write the input files of Lopster.

    dirty.csv and clean.csv become
    - {output_dir}/{dataset_name}/clean.csv
    - {output_dir}/{dataset_name}/dirty01.csv
    """
    import shutil

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    shutil.copy(clean_csv, os.path.join(dataset_dir, 'clean.csv'))
    shutil.copy(dirty_csv, os.path.join(dataset_dir, 'dirty01.csv'))

    print(f"data written to: {dataset_dir}/")
    return dataset_dir
