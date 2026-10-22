# Foundation-model cleaning results

The released results use Claude Opus 4.8 (`claude-opus-4-8`) for foundation-model
cleaning in the style of Narayan et al., "Can Foundation Models Wrangle Your Data?"
(PVLDB 16(4), 2022). All twelve datasets are included. The captured inputs comprise
147 smaller batches and seven whole training splits, for 154 input files in total.

## Replay

Run these commands from the repository root. Replay uses only the Python standard
library and the shipped CSV/JSON files. It does not call an API or fit a model.

```sh
python baselines/runners/run_llm.py --dataset all --check
python baselines/runners/run_llm.py --dataset mercedes --out /tmp/mercedes_cleaned.csv
python baselines/runners/llm_cost.py --check
python baselines/runners/merge_mercedes.py --check
```

`--check` replays in memory and checks the exact CSV bytes against the shipped
result. It writes nothing. `--out` saves the replayed CSV; without either flag,
the output is `results/runs/llm/<dataset>/cleaned.csv`. The script's separate
`--live` option explicitly requests new paid API calls and is not needed to
reproduce the captured results.

## Files

| File | Content |
|------|---------|
| `manifest.json` | Dataset and input membership, split information, hashes, result paths and list prices |
| `prompt_template.txt` | The cleaning instruction used for the captured requests |
| `records/<dataset>.jsonl.gz` | One task record per input file, including its CSV, request, Read records, structured reply and per-turn usage |
| `llm_cost.json` | Number of input files, token usage and list-price cost for each dataset |

The final cleaned tables, scores and manifests are under
`results/main_table/<dataset>/baselines/llm/` for all twelve datasets. Each
`cleaned.csv.gz` holds the indexed CSV of the run, gzip-compressed. The CSV has
the full table's row count; the evaluator selects its training rows by index.

## Input and replay protocol

The model saw only the dirty training rows of the 60/20/20 split (split seed 42).
Each input contains `index` and feature columns; the label column is withheld.
The captured Read calls refer only to that task's input file. The instruction
permits neither web access nor a clean-data lookup.

Replay reads the dirty table, applies the captured replies in batch-filename and
reply order, and keeps the index and label unchanged. Unknown indices, held-out
rows and non-feature columns in a reply are ignored, as in the recorded exporter.
The input CSV hashes, dataset hash, training-index hash and final CSV hash are
checked. Every input value must match the corresponding dirty feature value.

Task records retain the input CSV and the final structured reply. Request and
Read records reference inputs as `input/<filename>`, whose contents are the
record's `input.csv` field. These are task-specific records.

The records include the declared model, effort and response schema of each
request. The runtime was the Claude Agent SDK with Read and StructuredOutput
tools. Temperature and maximum output-token settings were not recorded. The
optional standalone live adapter uses the same dirty inputs and cleaning
instruction but is a new run; its output and token usage are separate from the
reported results.

## Cost

`llm_cost.json` sums the reported usage of every turn in the captured requests, counting
each streamed assistant message once. It includes input, cache reads, cache
writes and output, rather than counting only the final turn. The list prices per
million tokens are $5 for ordinary input, $6.25 for five-minute cache writes, $10
for one-hour cache writes, $0.50 for cache reads and $25 for output.

The total is **$207.30**, including **$98.27** for all 67 mercedes batches.
These are list-price estimates from recorded token usage. They are distinct from
GT-cell use, which is zero for this baseline.
