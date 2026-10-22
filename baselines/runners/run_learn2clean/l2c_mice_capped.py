#!/usr/bin/env python
"""Run the MICE candidate of Learn2Clean on mercedes with a cap on the MICE loop of impyute 0.0.8.

impyute 0.0.8 stops its chained equations only when every imputed cell changed by less than 10%
in the latest update of its column. On mercedes, whose incomplete columns are standardized and
centred near zero, this relative rule is never met and the upstream loop does not terminate. For
this dataset only, the operator keeps the upstream algorithm (column means as placeholders, a
randomly chosen incomplete column regressed on all other columns by least squares, the same 10%
rule) and stops after at most ten updates per incomplete column on average, the ten rounds that
scikit-learn's IterativeImputer runs by default. Every other dataset runs the unmodified operator
through l2c_clean.py. On mercedes Learn2Clean selects a pipeline without MICE (ZSB then OLS), so the
reported score does not depend on the cap.

    python baselines/runners/run_learn2clean/l2c_mice_capped.py --dataset mercedes --candidate 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import impyute
from impyute.util import checks, find_null, preprocess
from sklearn.linear_model import LinearRegression

CYCLES = 10
CALLS = []


@preprocess
@checks
def mice_capped(data, **kwargs):
    null_xy = find_null(data)
    null_xyv = np.append(null_xy, np.zeros((np.shape(null_xy)[0], 1)), axis=1)
    null_xyv = [[int(x), int(y), v] for x, y, v in null_xyv]
    temp = []
    cols_missing = set([y for _, y, _ in null_xyv])
    for x_i, y_i, value in null_xyv:  # step 1 of impyute: column means as placeholders
        col = data[:, [y_i]][~np.isnan(data[:, [y_i]])]
        new_value = np.mean(col)
        data[x_i][y_i] = new_value
        temp.append([x_i, y_i, new_value])
    null_xyv = temp
    converged = [False] * len(null_xyv)
    budget, updates = CYCLES * len(cols_missing), 0
    while not all(converged) and updates < budget:  # the upstream loop plus the update cap
        dependent_col = int(np.random.choice(list(cols_missing)))
        missing_xs = [int(x) for x, y, value in null_xyv if y == dependent_col]
        x_train, y_train = [], []
        for x_i in (x_i for x_i in range(len(data)) if x_i not in missing_xs):
            x_train.append(np.delete(data[x_i], dependent_col))
            y_train.append(data[x_i][dependent_col])
        model = LinearRegression()
        model.fit(x_train, y_train)
        for i, z in enumerate(null_xyv):
            x_i = z[0]
            y_i = z[1]
            value = data[x_i, y_i]
            if y_i == dependent_col:
                new_value = model.predict([np.delete(data[x_i], dependent_col)])
                data[x_i][y_i] = new_value.reshape(1, -1)
                if value == 0.0:
                    delta = (new_value - value) / 0.01
                else:
                    delta = (new_value - value) / value
                if abs(delta) < 0.1:
                    converged[i] = True
                else:
                    converged[i] = False
        updates += 1
    CALLS.append(dict(updates=updates, cap=budget, incomplete_columns=len(cols_missing),
                      cells=len(converged), converged_cells=int(sum(converged)), rule_met=bool(all(converged))))
    print('MICE_CAPPED', json.dumps(CALLS[-1]), flush=True)
    return data


impyute.mice = mice_capped  # Learn2Clean's Imputer calls impyute.mice at run time

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--candidate', type=int, required=True)
    a = ap.parse_args()
    if a.dataset != 'mercedes':
        raise SystemExit('the capped MICE is used only for mercedes; run l2c_clean.py for every other dataset')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import l2c_clean as L
    L.candidate(a.dataset, a.candidate)
    f = L.RESULTS / a.dataset / 'candidates' / ('cand_%02d' % a.candidate) / 'result.json'
    r = json.loads(f.read_text())
    r['mice_operator'] = dict(implementation='impyute 0.0.8 MICE with at most %d updates per incomplete column on average' % CYCLES,
                              calls=CALLS)
    f.write_text(json.dumps(r, indent=2))
    print('CAPPED_DONE', json.dumps(r['mice_operator']), flush=True)
