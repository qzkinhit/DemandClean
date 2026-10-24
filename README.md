# DemandClean

English | [简体中文](README_zh.md)

DemandClean learns a cleaning policy for a downstream model. For each detected error, it can keep the value, request a ground-truth repair, delete the row, or replace the value with an estimate.

This repository contains the implementation, final configurations, datasets, outputs, and reproduction records for the paper. The comparison covers twelve datasets, eighteen baselines, and four reference strategies. The additional budget comparison runs ActiveClean, Comet, and Lopster on nine datasets and RetClean on all twelve, each at two GT levels, for 78 runs.

## Quick start

Use Python 3.8 or newer in a separate environment. The reference downstream scorer uses scikit-learn 1.3.2.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.3.2
```

Check the released records and replay an attached LLM result without training models or making API calls:

```sh
python evaluation/collect_results.py --check
python baselines/runners/run_llm.py --dataset breast_cancer --check
```

To train and clean with the final breast_cancer configuration:

```sh
python scripts/run_demandclean.py --config configs/datasets/breast_cancer.yaml \
    --variant full --out results/runs/breast_cancer
```

Use each run’s `manifest.json` for its recorded parameters and seed. See the [reproduction guide](docs/CODE_TOUR.md) for checkpoint inference, downstream scoring, attribution, and figure generation.

## Documentation

- [Baseline reproduction](baselines/README.md): source repositories, implementation adaptations, GT and validation use, reward and budget constants, and FM costs.
- [Released results](results/README.md): final outputs, run records, and the [result index](results/index.json).
- [Code and evaluation guide](docs/CODE_TOUR.md): implementation structure, evaluation protocol, and commands for the paper’s tables and figures.
- [FM replay and logs](results/llm/README.md): prompts, responses, and token accounting.
- [Figures](figures/): plotting scripts and final inputs.
- Appendix of the paper ([English](docs/appendix.pdf), [中文](docs/appendix_zh.pdf)). It carries what the twelve-page limit left out: the proof that the reachable quality states are dense in the region anchored by the four extreme strategies, the proof that optimal operation selection is NP-complete, the sufficiency analysis that bounds the planning error by the oscillation of performance inside one quality-state fiber, the features of the learned agent and its training setup, the detectors and the value-estimation chain as implemented, and the full per-dataset results with wall-clock runtimes.

The implementation is under [demandclean/](demandclean/), dataset settings under [configs/](configs/), and data with rule libraries under [data/](data/).

## License

Project code is covered by [LICENSE](LICENSE). Third-party code and datasets retain their own licenses; see [baseline licenses](baselines/THIRD_PARTY.md) and [dataset documentation](data/README.md).
