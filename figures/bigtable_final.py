#!/usr/bin/env python3
"""Generate the final comparison-table rows from public records.

The input contains one fixed released output per dataset and method. The command
formats stored scores and GT accounting; it does not run models or choose runs.

    python figures/bigtable_final.py --check
    python figures/bigtable_final.py --out output/tables
"""
import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ['adult', 'beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har',
      'flights', 'soccer', 'hospitals']
SHORT = {'breast_cancer': 'breast\\_c.', 'smartfactory': 'smartf.', 'soilmoisture': 'soilm.', 'hospitals': 'hospital'}
TASK = {'adult': 'C', 'beers': 'C', 'breast_cancer': 'C', 'smartfactory': 'C', 'bike': 'R', 'mercedes': 'R', 'nasa': 'R',
        'soilmoisture': 'R', 'har': 'Cl', 'flights': 'C', 'soccer': 'C', 'hospitals': 'C'}
# first and second model of Table III (the models DC selects); the organically dirty datasets have one model
MOD = {'adult': ('rf', 'gb'), 'beers': ('rf', 'gb'), 'breast_cancer': ('lr', 'gb'), 'smartfactory': ('rf', 'gb'),
       'bike': ('lr', 'lasso'), 'mercedes': ('rf', 'ridge'), 'nasa': ('lr', 'rf'), 'soilmoisture': ('gb', 'rf'),
       'har': ('agglomerative', 'kmeans'), 'flights': ('dt',), 'soccer': ('dt',), 'hospitals': ('rf',)}
MNAME = {'rf': 'RF', 'gb': 'GB', 'lr': 'LR', 'lasso': 'LASSO', 'ridge': 'RIDGE', 'agglomerative': 'Agglo.',
         'kmeans': 'KMeans', 'dt': 'DT'}
REFS = [('NoFix', '\\textsc{NoFix} ($D_{\\mathit{dirty}}$)'), ('DeleteAll', '\\textsc{DeleteAll}'),
        ('ReplaceAll', '\\textsc{ReplaceAll}'), ('RepairAll', '\\textsc{RepairAll} ($D_{\\mathit{clean}}$)')]
EXT = [('HoloClean', '\\textsc{HoloClean}'), ('Horizon', '\\textsc{Horizon}'), ('Raha+Baran', '\\textsc{Raha}+\\textsc{Baran}'),
       ('UniClean', '\\textsc{UniClean}'), ('Lopster', '\\textsc{Lopster}'), ('ActiveClean', '\\textsc{ActiveClean}'),
       ('BoostClean', '\\textsc{BoostClean}'), ('SAGA', '\\textsc{SAGA}'), ('DiffPrep', '\\textsc{DiffPrep}'),
       ('Comet', '\\textsc{Comet}'), ('CtxPipe', 'CtxPipe'), ('MLImputer', 'ML Imp.'), ('SimpleImputer', 'Mean/Med./Mode'),
       ('LLM', 'FM (Claude Opus~4.8)'), ('DIANA', 'DIANA'), ('Learn2Clean', 'Learn2Clean'), ('RetClean', 'RetClean'),
       ('DAGPipe', 'DAGPipe')]
# records that hold the first-choice model only, the model of the reported block
FIRST_MODEL_ONLY = {'DiffPrep', 'DIANA', 'Learn2Clean', 'RetClean', 'DAGPipe'}
DCROW = [('DC', '\\textbf{\\sys{}}')]



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cells', type=Path, default=ROOT / 'results/tables/cells.json')
    ap.add_argument('--gt', type=Path, default=ROOT / 'results/tables/gt_final.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'output/tables')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    records = json.loads(a.cells.read_text(encoding='utf-8'))
    gt = json.loads(a.gt.read_text(encoding='utf-8'))
    cell = {tuple(key.split('|')): record['scores'] for key, record in records.items()}
    expected = {(ds, method) for ds in DS for method, _ in REFS + EXT + DCROW}
    if set(cell) != expected:
        raise ValueError('The comparison must contain exactly %d final dataset/method cells' % len(expected))
    n_scores = 0
    for (ds, method), values in cell.items():
        for model in (MOD[ds][:1] if method in FIRST_MODEL_ONLY else MOD[ds]):
            if not math.isfinite(values[model]):
                raise ValueError('Invalid observation: %s/%s/%s' % (ds, method, model))
            n_scores += 1
        if not math.isfinite(gt[ds][method]):
            raise ValueError('Missing GT accounting: %s/%s' % (ds, method))
    if a.check:
        print('Validated %d final cells and %d reported model scores.' % (len(cell), n_scores))
        return
    OUT = str(a.out)
    os.makedirs(OUT, exist_ok=True)
    rows = REFS + EXT + DCROW
    missing = []

    def fmt(x):
        if x is None:
            return '\\tbd{TBD}'
        x = 0.0 if round(x, 3) == 0 else x
        s = '%.3f' % x if abs(x) < 0.9995 else '%.2f' % x
        s = s.replace('0.', '.', 1) if s.startswith(('0.', '-0.')) else s
        return s.replace('-', '$-$')

    GROUP = ('Oracle-detection references', 'Cleaners', 'all')

    def block(ds_list, k, with_gt):
        """rows of one block: score of model k per dataset. Among the cleaners (external baselines, FM and DC; the four
        references use oracle detection and are not ranked) the column best is dark blue and bold, the runner-up light
        blue, and the first column ranks the cleaners by their mean rank over the datasets (ties share the mean rank)."""
        cleaners = EXT + DCROW
        top = {}
        meanrank = {}
        for ds in ds_list:
            m = MOD[ds][k]
            vals = sorted({round(cell[(ds, meth)][m], 3) for meth, _ in cleaners if m in cell.get((ds, meth), {})}, reverse=True)
            top[ds] = vals[:2]
            sc = [(round(cell[(ds, meth)][m], 3), meth) for meth, _ in cleaners if m in cell.get((ds, meth), {})]
            for v, meth in sc:
                better = sum(1 for w, _ in sc if w > v); same = sum(1 for w, _ in sc if w == v)
                meanrank.setdefault(meth, []).append(better + (same + 1) / 2.0)
        order = sorted((sum(r) / len(r), meth) for meth, r in meanrank.items())
        place = {meth: i + 1 for i, (_, meth) in enumerate(order)}
        lines = []
        for meth, name in rows:
            out = ['\\Circled{%d}' % place[meth] if meth in place else '', name]
            for ds in ds_list:
                m = MOD[ds][k]
                x = cell.get((ds, meth), {}).get(m)
                if x is None:
                    missing.append((ds, meth, m))
                s_ = fmt(x)
                if x is not None and (meth, name) in cleaners:
                    dc_top = round(cell[(ds, 'DC')][m], 3) == top[ds][0] if m in cell.get((ds, 'DC'), {}) else False
                    if round(x, 3) == top[ds][0]:
                        if meth == 'DC' or not dc_top:  # a tie for the best is marked on DC alone
                            s_ = '\\rkfirst{%s}' % s_
                    elif len(top[ds]) > 1 and round(x, 3) == top[ds][1] and round(x, 3) > round(cell[(ds, 'NoFix')][m], 3):
                        s_ = '\\rksecond{%s}' % s_  # the second value is marked only above the dirty data
                out.append(s_)
            if with_gt:
                out.append(gtcol(meth, ds_list))
            if (meth, name) == REFS[0]:
                lines.append('\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\' % (len(ds_list) + 3, GROUP[0]))
            if (meth, name) == EXT[0]:
                lines.append('\\midrule\n\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\' % (len(ds_list) + 3, GROUP[1]))
            if (meth, name) == DCROW[0]:
                lines.append('\\midrule')
            lines.append(' & '.join(out) + ' \\\\')
        return lines

    def gtcol(meth, ds_list):
        if meth in ('DeleteAll', 'ReplaceAll', 'RepairAll'):  # oracle detection reads every clean training cell
            return GROUP[2]
        # LLM: GT by the same rule as every other method (it reads no clean value), followed by the list-price
        # cost of its API calls over the twelve datasets (results/llm/llm_cost.json)
        if gt is None:
            return ''
        v = [gt[d][meth] for d in ds_list if d in gt and meth in gt[d] and gt[d][meth] is not None]
        out = '%.1f' % (sum(v) / len(v)) if v else '--'
        if meth == 'LLM':
            cost = json.loads((ROOT / 'results/llm/llm_cost.json').read_text(encoding='utf-8'))
            out += ' (\\$%d)' % round(sum(cost[d]['usd'] for d in ds_list))
        return out

    for fname, mlabel in (('main.tex', '\\textbf{Method}'),):
        head = ' & '.join(['', mlabel] + ['\\texttt{%s}$^{\\text{%s}}$' % (SHORT.get(d, d), TASK[d]) for d in DS])
        with open(os.path.join(OUT, fname), 'w') as fo:
            fo.write('%% generated by bigtable_final.py (first model of Table III); columns: rank, method, 12 datasets, GT%\n')
            fo.write(head + ' & \\textbf{GT\\%} \\\\\n')
            fo.write(' & '.join(['', '\\textit{1st Mod.}'] + [MNAME[MOD[d][0]] for d in DS]) + ' & \\\\\n\\midrule\n')
            fo.write('\n'.join(block(DS, 0, True)) + '\n')
    print('missing cells:', len(missing))
    for x in missing:
        print('  ', x)


if __name__ == '__main__':
    main()
