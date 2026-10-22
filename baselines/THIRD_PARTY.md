# Third-party code of the baselines

`baselines/Methods/` bundles the implementations of the baseline methods of the paper. The table
lists for each directory the upstream project, its license and the license file shipped with
it. Code written for this repository is distributed under the `LICENSE` of this repository.
Upstream code without a license is not included.

| Directory | Method and publication | Upstream | License | License file in this repository |
|-----------|------------------------|----------|---------|---------------------------------|
| `Baran_Raha/` | Raha (SIGMOD 2019) and Baran (PVLDB 13(12), 2020) | https://github.com/BigDaMa/raha | Apache-2.0 | `Baran_Raha/LICENSE.md` |
| `Baran_Raha/raha/tools/dBoost/`, `Baran_Raha/tools/dBoost/` | dBoost, outlier detector used by Raha | https://github.com/cpitclaudel/dBoost | GPL-3.0 | `Baran_Raha/raha/tools/dBoost/LICENSE`, `Baran_Raha/tools/dBoost/LICENSE` |
| `BoostClean/` | BoostClean (Krishnan et al., 2017), package `activedetect` | https://github.com/sjyk/activedetect | MIT | `BoostClean/LICENSE.txt` |
| `HoloClean/` | HoloClean (PVLDB 10(11), 2017) | https://github.com/HoloClean/holoclean | Apache-2.0 | `HoloClean/LICENSE` |
| `ctxpipe/` | CtxPipe (SIGMOD 2025) | https://github.com/ctxpipe/ctxpipe | Apache-2.0, with HAIPipe code under MIT | `ctxpipe/LICENSE`, `ctxpipe/NOTICE` |
| `Lopster/` | Lopster, "Generalizable Data Cleaning of Tabular Data in Latent Space" (PVLDB 17(13), 2024) | https://github.com/DataManagementLab/data_cleaning_with_latent_operators | none published, so the upstream code is not included (only the wrapper of this repository) | `Lopster/LICENSE_NOTE.md` |
| `ActiveClean/` | ActiveClean (PVLDB 9(12), 2016), reimplementation of the paper in `activeclean_core.py` | the released Python 2 code is bound to IMDB text data and is not used | `LICENSE` of this repository | `LICENSE` |
| `Comet/` | Comet, "Step-by-Step Data Cleaning Recommendations to Improve ML Prediction Accuracy" (EDBT 2025), reimplementation of the paper in `comet.py` | https://github.com/sedirmohammed/COMET (read for the details the paper leaves open, no code copied) | `LICENSE` of this repository | `LICENSE` |
| `Horizon/` | Horizon, "Horizon: Scalable Dependency-driven Data Cleaning" (PVLDB 14(11), 2021) | none, the paper links no implementation | none | `Horizon/LICENSE_NOTE.md` |
| `UniClean/` | UniClean, "UniClean: A Scalable Data Cleaning Solution for Mixed Errors based on Unified Cleaners and Optimized Cleaning Workflow" (PVLDB 18(11), 2025) | no public repository | none | `UniClean/LICENSE_NOTE.md` |
| `MLImputer/` | iterative, kNN and random-forest imputation | wrapper of scikit-learn (https://github.com/scikit-learn/scikit-learn) | wrapper: `LICENSE` of this repository, scikit-learn: BSD-3-Clause (installed as a dependency) | `MLImputer/LICENSE_NOTE.md` |
| `SimpleImputer/` | mean, median, mode and constant imputation, the simple imputer of the REIN benchmark (EDBT 2023) | wrapper of scikit-learn, REIN: https://github.com/mohamedyd/rein-benchmark | wrapper: `LICENSE` of this repository, scikit-learn: BSD-3-Clause (installed as a dependency) | `SimpleImputer/LICENSE_NOTE.md` |
| `SAGA/` | SAGA, "SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications" (SIGMOD 2024), search script `saga_topk.dml` on the `topk_cleaning` builtin of Apache SystemDS 3.2.0 (not bundled) | https://github.com/apache/systemds, evaluation functions adapted from https://github.com/damslab/reproducibility | Apache-2.0 for the adapted functions, the rest `LICENSE` of this repository | `SAGA/LICENSE-APACHE-2.0`, `SAGA/LICENSE_NOTE.md` |
| `DiffPrep/` | DiffPrep, "DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data" (SIGMOD 2023), adapter `diffprep_api.py` | https://github.com/chu-data-lab/diffprep, commit 6499edf | none published, so the upstream code is not included (only the adapter of this repository) | `DiffPrep/LICENSE_NOTE.md` |
| `Learn2Clean/` | Learn2Clean V1, "Learn2Clean: Optimizing the Sequence of Tasks for Web Data Preparation" (The Web Conference 2019), patch `learn2clean_v1.patch` | https://github.com/LaureBerti/Learn2Clean, commit 169bac3 | BSD-3-Clause, the upstream code is not included (only the patch of this repository) | `Learn2Clean/LICENSE_NOTE.md` |
| `DeleteAll/`, `DoNothing/`, `RepairAll/` | reference strategies of the paper | written for this repository | `LICENSE` of this repository | `LICENSE` |

`baselines/runners/`, `baselines/scripts/` and `baselines/tools/` are written for this repository
and fall under its `LICENSE`. Several upstream releases are needed and not bundled. SAGA runs on the
binary release of Apache SystemDS 3.2.0
(https://archive.apache.org/dist/systemds/3.2.0/systemds-3.2.0-bin.tgz) with a Java 17 runtime
(the runs used OpenJDK 17.0.15). DiffPrep runs on a clone of
https://github.com/chu-data-lab/diffprep at commit 6499edf in
`baselines/Methods/DiffPrep/diffprep_official/` (`DiffPrep/LICENSE_NOTE.md`). The runners of DIANA,
Learn2Clean, RetClean and DAGPipe read their upstream code from the directory named by
`DIANA_SOURCE`, `LEARN2CLEAN_SOURCE`, `RETCLEAN_SOURCE` or `DAGPIPE_SOURCE`, and by default from
`baselines/Methods/DIANA/diana-repo/` (https://github.com/camillasancricca/diana-repo, commit
88e9b54, no license published), `baselines/Methods/Learn2Clean/Learn2Clean/`
(https://github.com/LaureBerti/Learn2Clean, commit 169bac3 with `learn2clean_v1.patch` applied,
`Learn2Clean/LICENSE_NOTE.md`), `baselines/Methods/RetClean/RetClean/`
(https://github.com/qcri/RetClean, commit 7e72d93, MIT License) and
`baselines/Methods/DAGPipe/artifact/` (the artifact of the authors' Zenodo record
https://doi.org/10.5281/zenodo.20492518, SHA-256
14d8f10f057cc2322328707888b7cf67cac11cdbddc43c8207d882efd418ad72 of the archive). Apart from the
Learn2Clean patch, the runners use the upstream files unchanged and list each wrapper around an
upstream function in their docstrings and run records. RetClean also needs an Ollama server with `llama3.1:8b` and the
`sentence-transformers/all-MiniLM-L6-v2` embedding model.

## Changes to the upstream code

Every bundled project gets a wrapper module `<method>_wrapper.py` (and `__init__.py`) that
reads the dirty and clean tables of `data/` and returns the cleaned table. Comments and
messages that were not in English are translated in every file listed below. The other
changes are listed per project. Files not listed are byte-identical to the upstream default
branch.

**Raha and Baran.** Modified: `raha/baselines.py`, `raha/benchmark.py`, `raha/correction.py`,
`raha/dataset.py`, `raha/detection.py`, `raha/tools/dBoost/dboost/cli.py`,
`raha/tools/dBoost/dboost/features/__init__.py`,
`raha/tools/KATARA/knowledge-base/zipHasCity.rel.txt`, `requirements.txt`. The files
`detection.py`, `correction.py` and `tools/` next to `raha/` are copies of the modified package
files, kept because the wrapper imports both locations. Scripts of the upstream repository
that the benchmark does not run are not included.

**dBoost.** Modified: `dboost/cli.py` and `dboost/features/__init__.py` (in both copies). The
license of dBoost (GPL-3.0) covers these directories only.

**BoostClean.** Modified: the error detectors (`ErrorDetector.py`, `PatternErrorFinder.py`,
`PuncErrorModule.py`, `QuantitativeErrorModule.py`), the learning modules (`BoostClean.py`,
`CleanClassifier.py`, `EvaluateCleaning.py`, `baselines.py`), `loaders/csv_loader.py`,
`model_based/model_eval_utils.py`, `model_based/preprocessing_utils.py` and
`reporting/CSVLogging.py`. Added: `requirements.txt`, `test.py`.

**HoloClean.** Modified: `dataset/dataset.py`, `dataset/dbengine.py`, `dcparser/dcparser.py`,
`detect/detect.py`, `domain/domain.py`, `domain/estimators/logistic.py`, `evaluate/eval.py`,
`holoclean.py`, `repair/repair.py`, `requirements.txt`. Tests and documentation of the
upstream repository are not included.

**CtxPipe.** Modified: `conda-env.yml` (machine-specific `prefix` removed), `config.py`,
`ctxpipe/agent/dqn.py`, `ctxpipe/ctx.py`, `test.py`, `train.py`, `util.py` and
`ctxpipe/env/pipeline.py`. In `ctxpipe/env/pipeline.py` the calls of `os.setsid` and
`os.killpg` are guarded for systems without them, a `skip_evaluation` option returns before the
built-in evaluation (the benchmark scores the cleaned table with `evaluation/rescore_tables.py`), and
the upstream line

    os.system(f"pkill -f '/home/{os.getlogin()}/anaconda3/envs/ctxpipe.*joblib'")

is removed. That line ended every joblib process of the conda environment of the current user
after each pipeline step, including processes of other runs. Without it, the joblib workers
exit under the control of joblib (after an idle timeout or with the parent process). Added: the policy checkpoints `models/ctxpipe-3linear/ctx_50000_*.pkl`, which the
runner loads (the upstream repository ships the 32000-step checkpoints). The GTE-large
embedding model is not included, see the README.

**Lopster.** The upstream repository publishes no redistribution license, so its files are not
included. `baselines/runners/run_lopster.py` and `lopster_protocol.py` supply the final benchmark
configuration and the input/output transformations. The clean training copy is sampled with
seed 0 from the shared training split. Each dataset has a fixed clean-record budget, label
preservation setting, and constant-column fallback; no test score is used to choose them at
runtime. The wrapper removes the index, parses supported numeric units, applies the listed
categorical and identifier/date schema, then restores the indexed export before those final
transformations. The VAE uses 100 epochs, latent dimension 120, K=12, and a batch size bounded by
the clean copy's training portion. Each run gets an isolated model-cache directory.
`--prepare-only` and `--postprocess-only` operate without the upstream code or model training.
Obtain the upstream modules separately and follow
`baselines/Methods/Lopster/LICENSE_NOTE.md` for the loader-path/row-limit adaptations and the
recorded dependency versions. The existing `run_lopster/run_lopster_base.py` interface is retained.

**SimpleImputer.** The packaging files of the REIN benchmark (`setup.py`, its environment
`requirements.txt`) are not included. `requirements.txt` lists the packages the wrapper
imports.

**SAGA.** `saga_topk.dml` calls `topk_cleaning` in hold-out mode and replays the top-1 pipeline
with `executePipeline`, primitive by primitive, on the full dirty training frame. The
classification and regression evaluation functions follow the SAGA reproducibility scripts with
the changes noted in the comments of the script (the model is selected once on the hold-out pair
instead of the cross-validated `gridSearch`, which fails in SystemDS 3.2.0). The clustering
function is an adaptation for `har`. `saga_core.py` builds the inputs of the search and decodes
the replayed matrices into a table with the original columns.

**DiffPrep.** The upstream modules run unchanged from `diffprep_official/`. `diffprep_api.py`
differs from the upstream experiment script in the split (the split of the benchmark), the column
filters (computed on the training rows), the parameter initialization (without the test table),
the training loop (without the per-epoch test pass) and the label encoding (fitted on the
training and validation labels). It adds optional numerical stabilization (gradient clipping,
reset of non-finite gradients, input clipping). Each difference is listed in its docstring and
recorded in `deviations` of every run record.
`baselines/runners/run_diffprep/diffprep_clean.py` runs this adapter with the search space
restricted to missing-value imputation and outlier cleaning and exports the soft pipeline output of
the training rows. These two differences are also listed in `deviations`.

**Learn2Clean.** The upstream repository is not included. `Learn2Clean/learn2clean_v1.patch` holds
every change to `Learn2Clean_V1/python-package/learn2clean/` of commit 169bac3 that the runs used,
and `Learn2Clean/LICENSE_NOTE.md` describes them.
