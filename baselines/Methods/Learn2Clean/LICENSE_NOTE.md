# Learn2Clean

Learn2Clean V1 (Berti-Équille, The Web Conference 2019) of https://github.com/LaureBerti/Learn2Clean
is distributed under the BSD 3-Clause License. Its code is not included here. The reported runs
used commit 169bac3 with `learn2clean_v1.patch` of this directory applied. To run the Learn2Clean
baseline, clone the repository into this directory and apply the patch:

    git clone https://github.com/LaureBerti/Learn2Clean baselines/Methods/Learn2Clean/Learn2Clean
    git -C baselines/Methods/Learn2Clean/Learn2Clean checkout 169bac3236632d738dbe8b67e9373fd2bd19be16
    git -C baselines/Methods/Learn2Clean/Learn2Clean apply ../learn2clean_v1.patch

The patch changes `Learn2Clean_V1/python-package/learn2clean/` as follows. The imputers keep the
row index of their input, and the KNN imputer fills all numeric columns in one call. Imputation runs
even when no training row is complete. The Q-learning steps reach the last normalization and the last
feature-selection action. The chi-squared feature selector filters columns by name. The IQR outlier
rule counts the anomalous cells of a row, and the LOF rule removes the k rows with the smallest
negative outlier factor. `qlearning/__init__.py` imports only the Q-learner, the MARS regressor is
imported where it is used, and the operators run on one thread.

`baselines/runners/run_learn2clean/` runs the patched package under the protocol of
`baselines/runners/common.py`.
