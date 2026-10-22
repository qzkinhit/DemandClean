# Baseline reproduction

English | [简体中文](README_zh.md)

[Back to the project overview](../README.md). This page documents source projects, adaptations, GT use, validation use, and the constants of the reported runs. Commands are run from the repository root. Source links distinguish upstream projects from adaptations or reimplementations in this repository.

## Baseline reproduction

Every baseline cleans the training rows of the shared split. No baseline receives a test row except DAGPipe, whose learned graph also transforms the test inputs after its training ends. GT counts the clean cells read on the training rows. "Val." is the use of the validation split. Organic denotes `flights`, `soccer` and `hospital`.

| Method | Implementation / source | GT | Val. | Adaptation |
|---|---|---|---|---|
| HoloClean | adapted [HoloClean/holoclean](https://github.com/HoloClean/holoclean) | 0 | not used | DCs of the Clean4ML rule file, UniClean release DCs on organic data, `mercedes` in groups of 10 columns, `smartfactory` in row chunks, letter case of the `beers` output restored from the dirty input since the released code lowercases every string |
| Horizon | adapted [Horizon implementation](https://github.com/WelkinNi/Automatic-Data-Repair/tree/main/horizon) from the “Automatic Data Repair: Are We Ready to Deploy?” benchmark | 0 | not used | FDs of the Clean4ML rule file, UniClean release FDs on organic data |
| Raha+Baran | adapted [BigDaMa/raha](https://github.com/BigDaMa/raha) | 20 labeled tuples | not used | `mercedes` in 13 and `soilmoisture` in 5 column groups |
| UniClean | cleaning core adapted from [qzkinhit/Uniclean-bench-Result](https://github.com/qzkinhit/Uniclean-bench-Result/tree/main/uniclean_cleaners) | 0 | not used | cleaners of the Clean4ML rule file, cleaners of the UniClean release on organic data |
| Lopster | [DataManagementLab/data_cleaning_with_latent_operators](https://github.com/DataManagementLab/data_cleaning_with_latent_operators), with this repository’s adapter | clean copy of as many training records as the GT of DemandClean covers, and of half of the training rows at the default level of the budget table of the appendix; the main dataset x method table uses the budget of DemandClean | not used | input values kept in a column whose reconstruction collapses to a constant |
| FM | prompting adaptation of [Narayan et al.](https://github.com/HazyResearch/fm_data_tasks) with Claude Opus 4.8 without web access; [replay implementation](runners/run_llm.py) | 0 | not used | reads the dirty training rows without the label column and returns cell edits, the whole training split in one request on `beers`, `breast_cancer`, `flights`, `har`, `hospitals`, `nasa` and `soccer`, and in batches of about 25K tokens on `adult`, `bike`, `mercedes`, `smartfactory` and `soilmoisture` |
| ActiveClean | [paper reimplementation](Methods/ActiveClean/); the original Python 2 release linked by the [authors’ project](https://activeclean.github.io/) is bound to IMDB text | whole records, its default 500 records capped at half of the training rows, and as many as DemandClean's GT cells allow (the budget table of the appendix); the main dataset x method table uses the default level | model selection | hinge-loss linear SVM for classification, squared-loss linear model for regression, activity label of `har` as supervised surrogate, clock times of `flights` as minutes |
| BoostClean | adapted [sjyk/activedetect](https://github.com/sjyk/activedetect) | 0 | not used, internal hold-out of 20% of the training rows | label codes on `breast_cancer`, binary surrogate 1[y > median(y)] for the targets of `nasa`, `mercedes` and `soilmoisture` |
| SAGA | [Apache SystemDS](https://github.com/apache/systemds) 3.2.0 `topk_cleaning` with package defaults; evaluation adapted from [SAGA reproduction scripts](https://github.com/damslab/reproducibility/tree/master/sigmod2024-SAGA) | 0 | search objective | R² of `lm` for regression, k-means silhouette on `har`, unit strings of `beers` parsed as numbers |
| DiffPrep | [chu-data-lab/DiffPrep](https://github.com/chu-data-lab/DiffPrep), [commit 6499edf](https://github.com/chu-data-lab/DiffPrep/commit/6499edfbe2358288fbd0bde75f366f76da24ac27), with this repository's [adapter](Methods/DiffPrep/diffprep_api.py) and [runner](runners/run_diffprep/) | 0 | search objective and learning rate | search space restricted to missing-value imputation and outlier cleaning, dirty training matrix of the shared encoding as input with missing cells kept, soft pipeline output of the training rows exported and the test split left untouched, linear head on standardized targets with R² early stopping for regression, gradient clipping on `mercedes`, `nasa` and `soilmoisture`, activity label of `har` as surrogate |
| Comet | [paper reimplementation](Methods/Comet/), with details from [sedirmohammed/COMET](https://github.com/sedirmohammed/COMET) | an error mask from the clean training cells, its default 50-unit budget, and repairs matched to DemandClean's seed records and GT-repairs (the budget table of the appendix); the main dataset x method table uses the default budget; on `har` Comet runs on a 3,000-row sample of the training rows | utility score | R² utility for regression, silhouette of agglomerative nearest-centroid labels with 3 clusters on `har`, Ridge utility on `mercedes`, Ridge and GB utility on `soilmoisture` |
| CtxPipe | adapted [ctxpipe/ctxpipe](https://github.com/ctxpipe/ctxpipe) and pretrained embeddings | 0 | not used | none |
| ML Imp. | Clean4ML integration of [scikit-learn](https://github.com/scikit-learn/scikit-learn) MICE, KNN and RF imputers | 0 | not used | columns missing in every training row are dropped and restored afterwards |
| Mean/Med./Mode | [scikit-learn](https://github.com/scikit-learn/scikit-learn) `SimpleImputer` | 0 | not used | as ML Imp. |
| DIANA | [camillasancricca/diana-repo](https://github.com/camillasancricca/diana-repo), commit 88e9b54, with this repository's [runner](runners/run_diana/) | 0 | not used | per-column imputation and outlier methods recommended by the knowledge base for the model family of the dataset and applied to the dirty training matrix of the shared encoding, knowledge base of the random forest classifier on the regression datasets and `har`, columns without any training value filled by the standard imputer |
| Learn2Clean | [LaureBerti/Learn2Clean](https://github.com/LaureBerti/Learn2Clean) V1, commit 169bac3, with this repository's [patch](Methods/Learn2Clean/learn2clean_v1.patch) and [runner](runners/run_learn2clean/) | 0 | pipeline selection | Q-learning graph restricted to imputation, outlier removal, deduplication and consistency checking, operators fitted on the training rows, NB, OLS and HCA as goal models of classification, regression and `har`. On `mercedes` the MICE loop of impyute 0.0.8 never meets its stopping rule, so the one candidate that uses MICE runs with the loop capped at ten updates per incomplete column on average ([l2c_mice_capped.py](runners/run_learn2clean/l2c_mice_capped.py)), every other dataset uses the unmodified operators, and the pipeline selected on `mercedes` does not use MICE. |
| RetClean | [qcri/RetClean](https://github.com/qcri/RetClean), commit 7e72d93, with Llama 3.1 (8B) served locally by Ollama and this repository's [runner](runners/run_retclean/) | clean values of a fixed random half of the training rows as its data lake in the main dataset x method table, and at the lower level of the budget table of the appendix a lake no smaller than 1.65% of the cells or the GT of DemandClean, whichever is larger | not used | target cells from missing markers, type errors, and the domain and FD rules of the rule file on the dirty training rows, semantic retrieval of the three closest lake tuples on up to eight other columns, upstream prompt and parser with `ast.literal_eval` in place of `eval` |
| DAGPipe | authors' artifact, [Zenodo 10.5281/zenodo.20492518](https://doi.org/10.5281/zenodo.20492518), with this repository's [runner](runners/run_dagpipe/) | 0 | epoch selection | the authors' hierarchical DAG of imputation, encoding and scaling operators trained with their loss on the dirty training rows, operator statistics fitted on the training rows only, the learned graph also transforms the test inputs and the main-table model is trained and tested on its output, MSE head for regression, dirty activity label of `har` as surrogate. Training on a GPU is not bitwise deterministic, so the released results come from the single-threaded CPU run of the [runner](runners/run_dagpipe/README.md), which reproduces them byte for byte. |

## Per-dataset reward and budget constants

Constants of the reported runs. RF is a 10-tree random forest of depth 8, Ridge uses alpha = 1, KMeans uses 3 initializations, T is the number of training episodes, and lambda = 1e-4 on `bike` and `nasa`.

| Dataset | Proxy | T | r_max | lambda | delta_del | beta |
|---|---|---|---|---|---|---|
| `adult`, `beers` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `breast_c.`, `smartf.` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `bike` | RF | 200 | 1.0 | 1e-4 | -0.05 | 1.0 |
| `mercedes` | Ridge | 300 | 0.3 | 0.03 | -0.05 | 1.0 |
| `nasa` | RF | 200 | 1.0 | 1e-4 | -0.05 | 1.0 |
| `soilmoist.` | Ridge | 300 | B = 0 | 0.03 | -0.05 | 1.0 |
| `har` | KMeans | 300 | 0.3 | 0.1 | -0.02 | 0.2 |
| `flights` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `soccer`, `hospital` | RF | 300 | B = 0 | 0.03 | -0.02 | 0.2 |

## FM reproduction and cost

FM uses Claude Opus 4.8 without web access and reads only dirty training rows with the label column omitted. Under the same GT accounting as the other methods, its GT usage is 0.

Reported API cost includes every round of each final adopted run. Repeated context in each round is charged at the cache-read rate, and cache creation is charged at the applicable write rate. The Opus 4.8 rates per million tokens are:

| Token category | USD per million tokens |
| --- | ---: |
| Uncached input | $5.00 |
| Cache write, 5-minute lifetime | $6.25 |
| Cache write, 1-hour lifetime | $10.00 |
| Cache read | $0.50 |
| Output | $25.00 |

The total reported cost is **$207.30** across the twelve datasets. The largest dataset cost is **$98.27 for mercedes**. Per-dataset usage and charges are recorded in [results/llm/llm_cost.json](../results/llm/llm_cost.json).

The LLM baseline can replay the attached responses offline:

```sh
python baselines/runners/run_llm.py --dataset mercedes --check
```

This validates the replay against the released output without making API calls. Requests, responses, and accounting for the adopted outputs are under [results/llm/](../results/llm/).

## Baselines and licenses

HoloClean's final `beers` output can also be replayed with [restore_case.py](runners/run_holoclean/restore_case.py) using its released [run manifest](../results/main_table/beers/baselines/holoclean/manifest.json). This fixed postprocessing step restores case using the full dirty input vocabulary and requires no clean values.

The fixed Lopster preparation and final-output transformations are available in [baselines/runners/run_lopster.py](runners/run_lopster.py). `--prepare-only` and `--postprocess-only` need no upstream code or model training; [the Lopster setup note](Methods/Lopster/LICENSE_NOTE.md) gives the input format, fixed settings, and source setup.

Baseline-specific dependencies, adapters, and licenses are listed in [baselines/THIRD_PARTY.md](THIRD_PARTY.md). SAGA requires Apache SystemDS 3.2.0 and Java 17. HoloClean requires PostgreSQL, and CtxPipe requires the GTE-large embedding model. Lopster and DiffPrep upstream code is obtained separately because no redistribution license is published; their license notes identify the required source. The upstream code of DIANA, Learn2Clean, RetClean, and DAGPipe is also obtained separately, at the versions in the table, and [baselines/THIRD_PARTY.md](THIRD_PARTY.md) names the directories that their runners read. Learn2Clean runs with the patch of this repository. Run manifests and adapter documentation specify the configuration used here.

Code authored for this repository is covered by [LICENSE](../LICENSE). Datasets and third-party code retain their respective licenses, documented in [data/README.md](../data/README.md) and [baselines/THIRD_PARTY.md](THIRD_PARTY.md).
