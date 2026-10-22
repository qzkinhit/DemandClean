# License note: Lopster

Upstream: https://github.com/DataManagementLab/data_cleaning_with_latent_operators

Publication: E. dos Reis, M. Abdelaal, C. Binnig. Generalizable Data Cleaning of Tabular Data in
Latent Space. PVLDB 17(13), 2024.

The upstream repository publishes no redistribution license. Its source is not included in this
release. The wrapper and the public reproduction entry point are written for this repository
and are covered by its license. Obtain the upstream files separately and follow the authors'
terms before using them.

## Reproduction entry point

`baselines/runners/run_lopster.py` reads a final `run.json` or `manifest.json`, or uses the fixed
configuration for `--dataset` and `--budget-level main`, `dcbudget`, or `ample`.
`baselines/runners/lopster_protocol.py` contains the dataset schemas and the final settings.
The existing `baselines/runners/run_lopster/run_lopster_base.py` interface remains available.

```sh
python baselines/runners/run_lopster.py --dataset adult --prepare-only --output_path output/lopster
python baselines/runners/run_lopster.py --result-config results/main_table/adult/baselines/lopster/run.json --postprocess-only raw_cleaned.csv --check results/main_table/adult/baselines/lopster/cleaned.csv.gz
```

Preparation and postprocessing run without TensorFlow or the upstream implementation. The second
command expects the indexed export before final label preservation and constant-column fallback.
`--out FILE` writes that transformed CSV; `--check FILE` compares it with a saved CSV or CSV.gz.
Neither operation trains a model or evaluates a downstream score.

The clean copy consists of the recorded number of training records, sampled without replacement
using NumPy `RandomState(0)` and kept in training-row order. Training rows use the shared split
with seed 42. At the ample budget, the copy contains half of the training records. The VAE uses
100 epochs, latent dimension 120, K=12, learning rate 0.001, and batch size
`max(1, min(256, int(0.7 * clean_records) - 1))`. The sampler seed is separate from model training.

The data adapter removes the index, normalizes missing markers, converts numeric columns and
recognized unit strings, and uses the dataset's categorical, identifier, date and float-column
schema. The upstream encoder/scaler is fitted on the clean copy. After the indexed export is
restored, the fixed configuration optionally preserves the input target column. It then restores
the dirty input values in any output column with at most one distinct value when the input
column has more than one. This order matches the released outputs. No score is read when
choosing these settings. The three organically dirty datasets use the unchanged-input case at
zero clean records; their full indexed CSV is retained, and no VAE is called.

## Required upstream setup

1. Obtain the source from the upstream URL. Keep it in an external directory, or place its
   modules beside `lopster_wrapper.py`. The runner requires `datasets.py`, `latent_operators.py`,
   `utils.py`, `error_detection.py`, and their imports. Pass the directory with `--upstream-root`.
2. In `datasets.py`, ensure `load_regression`, `load_regression_dirty`, and
   `load_features_and_data` accept `path_to_dataset` and use it as the root of
   `<dataset>/clean.csv` and `<dataset>/dirty01.csv` (falling back to `dirty.csv`). Treat a
   nonpositive row limit as the complete table rather than applying a `[:-1]` slice. These are
   the input-path and complete-row adaptations used by the wrapper.
3. The runner writes `dataset_configuration.json` from the supplied benchmark schemas into its
   private runtime directory. The upstream `_get_data_config` must read that file from the
   working directory. The index is already removed from the VAE input; preserve the listed
   identifier/date columns when restoring the indexed export.
4. Use an environment with the upstream dependencies. The recorded training environment used
   Python 3.7.12, TensorFlow 2.11.0, Keras 2.11.0, pandas 1.2.5, NumPy 1.21.6, and
   scikit-learn 0.24.0. Downstream evaluation uses the separate common scoring environment.
5. Run with a fresh output task directory. The runner copies the locally supplied modules into
   that directory and creates an isolated model-cache location. It refuses to reuse an existing
   backend directory, so a different budget cannot silently load a cached VAE.

```sh
python baselines/runners/run_lopster.py --dataset adult --upstream-root /path/to/upstream --output_path results/runs/lopster
```

The upstream command-line `lopster.py` is not needed by this entry point. The wrapper calls its
Python functions directly and binds the recorded learning rate. No upstream files or trained
model caches are added to the released result directories by preparation or postprocessing.
