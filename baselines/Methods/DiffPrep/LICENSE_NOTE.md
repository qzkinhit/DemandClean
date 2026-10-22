# DiffPrep

The DiffPrep code (Li, Chen, Chu, Rong, SIGMOD 2023) of https://github.com/chu-data-lab/diffprep
publishes no license, so it is not redistributed here. The reported runs used commit 6499edf
without changes. To run DiffPrep, clone it into this directory:

    git clone https://github.com/chu-data-lab/diffprep baselines/Methods/DiffPrep/diffprep_official
    git -C baselines/Methods/DiffPrep/diffprep_official checkout 6499edf

`diffprep_api.py` imports the upstream modules from `diffprep_official/` and holds every
difference from the upstream experiment script, listed in its docstring and in `DEVIATIONS`. Each
run record names the differences that applied to it (`deviations` of `run.json`).
