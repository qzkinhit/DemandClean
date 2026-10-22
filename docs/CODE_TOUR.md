# Code and reproduction guide

Paths in this guide are relative to the repository root. [The result index](../results/index.json) locates the paper's final runs; [results/README.md](../results/README.md) describes their files.

## Implementation

| Module | Responsibility |
| --- | --- |
| `scripts/run_demandclean.py` | Configuration, data split, encoding, run execution, and output records |
| `demandclean/api/demand_clean.py` | `DemandClean` interface for fitting, cleaning, saving, and loading a policy |
| `demandclean/config/config.py` | System parameters and model, detector, and inference options |
| `demandclean/detectors/` | Hybrid detection, rules, oracle detection, and error injection |
| `demandclean/core/state/` | Task-specific state features |
| `demandclean/core/environments/` | Cleaning actions, reward, and value estimation |
| `demandclean/core/agents/` | Policy networks and action selection |
| `demandclean/training/trainer.py` | Clean-base construction and policy training |
| `demandclean/inference/` | Online cleaning and offline plan execution |
| `demandclean/models/` | Reward-model adapters |
| `evaluation/` | The common downstream scorer, REDR, and attribution |

The runner reads a dataset configuration and a variant from `configs/variants.yaml`, prepares the training split, builds the detector and reward model, trains or loads the policy, and exports the cleaned output. Dataset configuration and the run's manifest together specify the final execution. The five available variant identifiers are `full`, `ngt`, `nd`, `arch`, and `stage`.

For example, the recorded breast_cancer setting uses the default seed:

```sh
python scripts/run_demandclean.py --config configs/datasets/breast_cancer.yaml \
    --variant full --out results/runs/breast_cancer
```

The attached nasa policy can be used through its recorded checkpoint setting:

```sh
python scripts/run_demandclean.py --config configs/datasets/nasa.yaml \
    --checkpoint results/main_table/nasa/demandclean/model --out results/runs/nasa
```

The reported numbers are reproduced by scoring the released run outputs with the scoring scripts below.

## Scoring and attribution

`evaluation/paper_score.py` scores DemandClean's encoded output. `evaluation/rescore_tables.py` aligns reference and baseline tables to training rows by their `index` column and uses the same downstream scorer. Dataset YAML files specify the reported downstream model and metric.

The split is 60/20/20 with split seed 42. Categorical encoders and scaling are fitted on dirty training rows. Classification and regression use clean test rows. HAR uses the silhouette of the cleaned training rows. The `nearest` field of the released score and the run's manifest specify how categories absent from training are encoded. scikit-learn 1.3.2 is the reference version for this common scoring path.

The relative downstream measure is `Tol_rel = (P_method - P_NoFix) / (P_RepairAll - P_NoFix)`, with each performance clipped to [0, 1] before normalization. A zero reference gap has no defined relative value.

GT totals account for clean information supplied to cleaning, including calibration and repairs. GT% divides the recorded total by `total_cells` in the dataset YAML. Baseline run records retain their supplied budget and consumption. Shared evaluation data and the row identifier are outside the GT accounting. FM's API cost is separate from GT usage.

`evaluation/table_redr.py` counts erroneous records before and after cleaning, including errors in retained rows and excluding deleted rows after cleaning. It reports `(R_before - R_after) / R_before`.

`evaluation/attribution_outcome.py` partitions the output into true-value writes, other writes, and deletions by comparison with dirty and clean references. `evaluation/attribution_operation.py` partitions writes and deletions from the dirty data and output. These are post-run analyses; their coalition scores follow the downstream evaluation protocol. The full coalition is checked against the attached run's score.

## Mapping paper results to files

| Paper content | Released material |
| --- | --- |
| Twelve-dataset method comparison | `results/tables/cells.json`, `results/index.json`, per-run `scores.json`, and `results/scores/downstream.jsonl` |
| Comparison GT accounting | `results/tables/gt_final.json` |
| Rule robustness | `results/robustness/index.json`, per-run outputs, and `results/robustness/table.json` |
| Sufficiency of the quality state (appendix) | `results/robustness/quality_state/`, with one record per random policy and `summary.csv` |
| Cross-dataset transfer | `results/transfer/index.json`, per-run outputs, and `results/transfer/table.json` |
| DemandClean GT and detection | Per-run `summary.json` and `scores.json` |
| Four reference strategies | `references/` under the dataset results and `evaluation/build_replaceall.py` |
| Three organically dirty datasets | Their directories in `results/main_table/`, projected into `results/real_world/table.json` |
| 78 budget runs | `results/budget/<dataset>/{activeclean,comet,lopster,retclean}_{dcbudget,ample}/` |
| Outcome attribution | `results/scores/attribution.jsonl` and per-run `attribution.json` |
| Operation attribution | Per-run `attribution_operation.json`, `results/attribution_operation/rows.jsonl`, and `results/scores/attribution_operation.jsonl` |
| Final HAR run | `results/main_table/har/demandclean/` (scores, attribution, and REDR) |
| Figure 4 | `figures/data/fig4_final.json`, `figures/data/dc_detection.json`, and `figures/data/fig_baselines.csv` |
| Figure 5 | `figures/data/abl_final.json`, `figures/data/fig_ablation.csv`, `results/ablation/`, and `configs/variants.yaml` |
| FM prompts, responses, and cost | `results/llm/`, including `llm_cost.json` and `records/` |

The operation-attribution manifest binds each record to its cleaned arrays and ordered row identifiers. The scripts under `results/attribution_operation/` are public reproduction adapters for these archived results.

Recorded execution durations refer to the hardware and scope of their attached run. For the paper's runtime comparison, use the measurement source and hardware identified in the appendix.

The following commands print evaluation records. They fit downstream models but keep the released score files unchanged:

```sh
python evaluation/paper_score.py results/main_table/beers/demandclean
python evaluation/rescore_tables.py --datasets beers
python evaluation/table_redr.py --datasets beers
python evaluation/attribution_outcome.py results/main_table/beers/demandclean
python evaluation/attribution_operation.py results/main_table/beers/demandclean
python evaluation/quality_state_sufficiency.py --datasets beers --n 600 --workers 8 --out output/quality_state
python evaluation/quality_state_sufficiency.py --analyze --out output/quality_state
```

`--nearest` on either scoring entry point selects nearest-category encoding for records that use it. To regenerate the dataset statistics, run `python evaluation/dataset_stats.py`.

## Record and table integrity

```sh
python evaluation/collect_results.py --check
python figures/bigtable_final.py --check
python figures/fig4_final.py --check
python results/robustness/aggregate.py
python results/transfer/aggregate.py
```

The collector verifies the released per-run records and aggregate indexes. The table builder uses the fixed 276 cells in [results/tables/cells.json](../results/tables/cells.json), and `--out output/tables` exports the table rows. The Figure 4 check compares its 152 observations with the attached score, REDR, and GT records. Robustness and transfer aggregation reads the saved run scores. None of these checks fits a model.

## Figures and FM replay

```sh
python figures/build_figure_data.py
python figures/make_figures.py --out output/figures
python baselines/runners/run_llm.py --dataset mercedes --check
```

The figure builder exports final figure inputs. Figure 4 covers eight datasets and nineteen series, DemandClean and the eighteen baselines including FM, using record-level REDR. DAGPipe has no REDR or detection F1 because it outputs transformed features. Figure 5 covers eight datasets and five configurations. Its 32 non-Full runs have final configurations, outputs, and records under `results/ablation/<dataset>/<variant>/`; the Full outputs are in `results/main_table/`. FM has zero GT usage; its API cost remains in `results/llm/llm_cost.json`.

The LLM check replays recorded responses in memory and compares them with the attached cleaned table. It does not call a provider. To export the replayed table, supply `--out output/mercedes_cleaned.csv`. The attached LLM manifest identifies the final model, inputs, responses, and accounting.

[The appendix](appendix.pdf) ([中文](appendix_zh.pdf)) contains the full proofs, the method details and the additional results.

## Baseline implementations

[baselines/THIRD_PARTY.md](../baselines/THIRD_PARTY.md) identifies bundled upstream code, adapters, licenses, and method-specific dependencies. Baseline execution uses the command and parameters in the final run record. SAGA requires SystemDS 3.2.0 and Java 17. Lopster, DiffPrep, DIANA, Learn2Clean, RetClean, and DAGPipe require separately obtained upstream code as described in their license notes and in the third-party file. The runners of DIANA, Learn2Clean, DiffPrep, RetClean, and DAGPipe share the protocol of `baselines/runners/common.py`, which gives a runner the dirty training rows and the clean validation rows and loads the test features only after the method is frozen. `evaluation/rescore_tables.py` scores their outputs, `cleaned_encoded.npz` of DIANA, Learn2Clean, and DiffPrep, `cleaned.csv.gz` of RetClean, and `transformed.npz` of DAGPipe, whose learned graph also transforms the test inputs. `evaluation/encoded_redr.py` computes REDR and detection F1 for the outputs of DIANA, Learn2Clean, RetClean, and DiffPrep. The table scorer accepts `--model` to evaluate the reported second model. Lopster's fixed clean-copy sampling and final label/constant-column transformations are implemented in `baselines/runners/run_lopster.py`; `--result-config` loads the final record, while `--prepare-only` and `--postprocess-only` perform no model training.

New outputs belong in `results/runs/`, `output/`, or an external directory. The released result directories serve as fixed inputs for reproduction.
