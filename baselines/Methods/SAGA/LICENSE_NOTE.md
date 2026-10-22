# SAGA

`saga_topk.dml` runs SAGA (Siddiqi, Kern, Boehm, SIGMOD 2024) through the builtins
`topk_cleaning` and `executePipeline` of Apache SystemDS 3.2.0, which is not bundled
(https://archive.apache.org/dist/systemds/3.2.0/systemds-3.2.0-bin.tgz, Apache-2.0). Its
evaluation functions `evalClassification` and `evalRegression` are adapted from the SAGA
scripts of the reproducibility repository https://github.com/damslab/reproducibility
(folder `sigmod2024-SAGA`, Apache-2.0). The changes are noted in the comments of
`saga_topk.dml`. `LICENSE-APACHE-2.0` is the text of the license. `evalClustering`,
`saga_core.py` and `baselines/runners/run_saga.py` are written for this repository.
