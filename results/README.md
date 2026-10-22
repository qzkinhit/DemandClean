# Released results

This directory contains the final outputs used by the paper. [index.json](index.json) maps datasets and methods to their records. The comparison includes twelve datasets, eighteen baselines, and four reference strategies. The nine benchmark datasets are adult, beers, breast_cancer, bike, mercedes, nasa, smartfactory, soilmoisture, and har; flights, soccer, and hospitals are the three organically dirty datasets.

## Layout

| Directory or file | Content |
| --- | --- |
| `main_table/<dataset>/demandclean/` | DemandClean outputs for all twelve datasets |
| `main_table/<dataset>/references/<method>/` | Reference strategies |
| `main_table/<dataset>/baselines/<method>/` | Baseline outputs and run records |
| `real_world/table.json` | Real-data comparison with references to the main result directories |
| `budget/<dataset>/<method>_dcbudget/` | The lower-budget setting of ActiveClean, Comet, or Lopster, and RetClean with a data lake no smaller than 1.65% of the cells or the GT of DemandClean, whichever is larger |
| `budget/<dataset>/<method>_ample/` | The corresponding ample-budget setting, for RetClean its main-table run with a data lake of half of the training rows |
| `ablation/<dataset>/{ngt,nd,arch,stage}/` | Final configurations and outputs for the 32 ablation runs |
| `robustness/` | 126 recorded rule-library inference runs and their aggregate table |
| `transfer/` | 32 recorded source-to-target inference runs and their aggregate tables |
| `tables/cells.json`, `tables/gt_final.json` | Final comparison cells and GT accounting |
| `llm/` | Adopted LLM inputs, requests, responses, and token/cost accounting |
| `llm/llm_cost.json` | Per-dataset adopted-run usage and API charges |
| `scores/downstream.jsonl` | Consolidated downstream scores |
| `scores/redr.jsonl` | Record-level error drop rates |
| `scores/attribution.jsonl` | Outcome-based attribution |
| `scores/attribution_operation.jsonl` | Write/delete operation attribution |
| `attribution_operation/` | The twelve operation-attribution records and their reproduction adapters |

The budget directories contain ActiveClean, Comet, and Lopster on the nine benchmark datasets and RetClean on all twelve datasets, each at two settings, or 78 runs. Use the run manifest and run record for each setting's supplied budget, actual consumption, and scoring configuration.

## Run records

| File | Content |
| --- | --- |
| `manifest.json` | Final command, configuration, seed where recorded, versions, artifact list, and provenance |
| `run.log`, `demandclean.log`, `stderr.log` | Available execution messages; the manifest identifies the attached files |
| `run.json` | Structured baseline execution record, when available |
| `summary.json` | Operations, GT accounting, detection statistics, and execution time for DemandClean |
| `scores.json` | Downstream model, score, encoding, and relevant GT accounting |
| `redr.json` | Erroneous records before and after cleaning and REDR |
| `attribution.json` | Outcome attribution for the attached DemandClean output |
| `attribution_operation.json` | Write/delete attribution for the same attached output |
| `cleaned.csv` or `cleaned.csv.gz` | Cleaned table aligned by the `index` column |
| `cleaned_encoded.npz` | Encoded output used by the scorer, for DemandClean, DIANA, Learn2Clean, and DiffPrep |
| `transformed.npz` | Training and test features of the graph that DAGPipe learns, on which the scorer trains and tests the downstream model |
| `targets.json` | Number of target cells per column that RetClean queries |
| `model/` | Attached policy checkpoints |

LLM outputs retain the full indexed CSV; replayed changes are confined to training features. The downstream scorer selects the training rows by their identifiers. DAGPipe produces transformed features rather than a cleaned table, so it has no REDR or detection F1. The records of DIANA, Learn2Clean, RetClean, DAGPipe, and DiffPrep hold the score of the first-choice model, the model of the comparison table. Attached logs contain the recorded execution output.

## Integrity and table checks

From the repository root:

```sh
python evaluation/collect_results.py --check
python figures/bigtable_final.py --check
python figures/fig4_final.py --check
python results/robustness/aggregate.py
python results/transfer/aggregate.py
```

These commands check or aggregate attached records without model training or rescoring. [tables/cells.json](tables/cells.json) and [tables/gt_final.json](tables/gt_final.json) contain the final method comparison inputs, and [llm/](llm/) holds the adopted LLM run.

## Reproduction

Run commands from the repository root with the dependencies described in [the main README](../README.md). The following examples print their results without changing the released records:

```sh
python evaluation/paper_score.py results/main_table/beers/demandclean
python evaluation/rescore_tables.py --datasets beers
python evaluation/table_redr.py --datasets beers
python evaluation/encoded_redr.py
python evaluation/attribution_outcome.py results/main_table/beers/demandclean
python evaluation/attribution_operation.py results/main_table/beers/demandclean
python baselines/runners/run_llm.py --dataset mercedes --check
```

The final manifest and score record determine the model, seed, and unseen-category encoding. Use `--nearest` only for scores whose record specifies that encoding. Classification and regression evaluate on clean test rows; clustering evaluates silhouette on the cleaned training rows. Recomputing scores fits the downstream models.

For a new cleaning run, write to `results/runs/` or a directory outside the release. The reproduction commands and artifact-to-paper mapping are detailed in [docs/CODE_TOUR.md](../docs/CODE_TOUR.md).
