# DAGPipe runner

`run_dagpipe.py` trains the hierarchical DAGPipe of the authors' artifact (Zenodo record
[10.5281/zenodo.20492518](https://doi.org/10.5281/zenodo.20492518), archive SHA-256
14d8f10f057cc2322328707888b7cf67cac11cdbddc43c8207d882efd418ad72) under the protocol of
[`baselines/runners/common.py`](../common.py). [`train_stats.py`](train_stats.py) fits the statistics of
the authors' operators on the dirty training rows before validation and inference.

```sh
DAGPIPE_SOURCE=<artifact directory> SCORER_PYTHON=<python with scikit-learn 1.3.2> \
    python baselines/runners/run_dagpipe/run_dagpipe.py --dataset beers --out results/runs/dagpipe/beers/seed0
```

The graph uses the four operator layers of the authors' README, Adam with learning rate 0.01, batch
size 256, a DAG penalty of 0.1, a sparsity penalty of 0.0001 and gradient clipping at norm 1.
Training runs for at most 200 epochs with seed 0 and stops after 20 epochs in which the objective on
the clean validation rows does not improve. This objective is the authors' validation loss for
classification and the negative R² for regression. The training labels are the dirty ones, and no clean training cell is read.
Regression uses a head with one output, MSE loss on the standardized dirty targets and float64
arithmetic. `har` uses its dirty activity labels as a supervised surrogate. After the epoch is
selected, the learned graph transforms the training, validation and test inputs, and `common.py`
trains and tests the downstream model of the comparison table on these features (`transformed.npz`).

Training on a GPU is not bitwise deterministic, and repeated GPU runs with the same arguments give
different scores. The released results of all twelve datasets come from this script run on one CPU
thread (`--device cpu --threads 1`, with `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` set to 1), which reproduces `transformed.npz` byte for byte. These settings are
the defaults of the script.
