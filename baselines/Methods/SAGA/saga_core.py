"""SAGA (Siddiqi, Kern, Boehm, SIGMOD 2024) adapter around SystemDS topk_cleaning.

This module holds the data handling that surrounds the DML search script
``saga_topk.dml``. It contains no paths, no CLI parsing and no downstream
evaluation:

* ``prepare_inputs`` turns the dirty training rows and the clean validation
  rows into the frames and the 3-row metadata (schema, categorical mask,
  FD mask) that topk_cleaning consumes.
* ``decode_export`` turns the per-primitive matrices written by the DML replay
  of the top-1 pipeline back into a table with the original columns: deleted
  rows are tracked by exact row matching, SMOTE rows (no index) are dropped,
  affine rescaling primitives (scale, normalize) are inverted, categorical
  codes are mapped back to the original strings.
"""
import re

import numpy as np
import pandas as pd

MISS = {'', 'empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null', 'None', '__NULL__'}
ENCODER_OPS = {'dummycoding', 'frequencyEncode', 'WoE', 'pca', 'ppca'}
AFFINE_OPS = {'scale', 'normalize'}
ROW_OPS = {'SMOTE', 'underSampling', 'tomeklink', 'abstain', 'flipLabels'}
_SEP_CANDIDATES = ['|', '~', '^', ';', '\t']
# a number followed only by a unit token made of letters, dots, hyphens, spaces or '%'
# ('12.0 oz.', '16.0 ounce', '12.0 oz. Alumi-Tek', '0.05%'); '1a0' or '..00' do not match
_UNIT_RE = re.compile(r'^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:%|[A-Za-z][A-Za-z.\- ]*)$')


def parse_fd_columns(rules_text):
    """Columns that appear in the [FD] section of a DemandClean rules.txt."""
    cols, in_fd = set(), False
    for raw in rules_text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            in_fd = line.upper().startswith('[FD')
            continue
        if in_fd and '=>' in line:
            lhs, rhs = line.split('=>', 1)
            for part in re.split(r'[,&]', lhs) + re.split(r'[,&]', rhs):
                if part.strip():
                    cols.add(part.strip())
    return cols


def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _sci(v):
    """Java style scientific notation (23 -> '2.3E1', 0.5 -> '5.0E-1').

    SystemDS 3.2.0 dropInvalidType classifies an integral literal such as '23' or
    '23.0' as INT and blanks it in an FP64 column; topk_cleaning runs that
    primitive on every column whenever the frame has a categorical attribute, so
    with plain decimal literals all integral numeric cells would reach the search
    as missing. Scientific notation is classified as FP and parses to the same
    double."""
    m, e = np.format_float_scientific(float(v), unique=True, trim='0').split('e')
    return m + 'E' + str(int(e))


def _strip_unit(v):
    m = _UNIT_RE.match(v)
    return m.group(1) if m else v


def _normalize(df, feature_cols, categorical_cols, sep, parse_units=True):
    """Missing tokens become '' (read as NA). In numeric columns a number carrying
    a unit suffix ('12.0 oz', '0.05%') is reduced to its number when parse_units is
    set. Any other value that does not parse as a number becomes '', which is what
    SAGA's dropInvalidType string primitive does (SystemDS only runs it when
    categorical columns exist).
    Numeric values are written with _sci. Separator and quote characters inside
    strings are replaced; the decoder maps codes back to the untouched original
    strings by row position."""
    out = pd.DataFrame(index=df.index)
    stats = {}
    for c in feature_cols:
        s = df[c].astype(str)
        miss = s.isin(MISS)
        if c in categorical_cols:
            s = s.str.replace(sep, ' ', regex=False).str.replace('"', "'", regex=False)
            s = s.str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)
            s = s.where(~miss, '')
            stats[c] = {'missing': int(miss.sum()), 'invalid_type': 0}
        else:
            n_unit = 0
            if parse_units:
                s2 = s.where(miss | s.map(_is_num), s.map(_strip_unit))
                n_unit = int((s2 != s).sum())
                s = s2
            ok = s.map(_is_num) & ~miss
            ok &= s.where(ok, '0').astype(float).map(np.isfinite)
            stats[c] = {'missing': int(miss.sum()), 'invalid_type': int((~ok & ~miss).sum()),
                        'unit_parsed': n_unit}
            s = pd.Series([_sci(v) if k else '' for v, k in zip(s.values, ok.values)], index=s.index)
        out[c] = s
    return out, stats


def choose_sep(frames):
    for sep in _SEP_CANDIDATES:
        if not any(f.apply(lambda col: col.astype(str).str.contains(sep, regex=False)).any().any() for f in frames):
            return sep
    raise ValueError('no free separator character')


def prepare_inputs(dirty_train, clean_val, feature_cols, label_col, categorical_cols,
                   task, fd_cols=frozenset(), parse_units=True):
    """Build SAGA's training frame, validation frame and metadata rows.

    Returns dict(train=DataFrame, val=DataFrame, meta=list[list[str]], sep=str,
    stats=dict). The label is the last column, as topk_cleaning(isLastLabel=TRUE)
    expects. Classification labels are recoded by topk_cleaning (mask=1).
    task='clustering' writes the row position (0..n-1) as a numeric placeholder
    label: topk_cleaning always splits off the last column and the clustering
    evaluation function ignores it. It is not constant because the bandit skips
    the evaluation when the label has a single value, and its minimum 0 keeps the
    search on the label-free branch (getOps removes SMOTE, tomeklink,
    underSampling, abstain and flipLabels, no class-stratified sampling)."""
    sep = choose_sep([dirty_train[feature_cols + [label_col]], clean_val[feature_cols + [label_col]]])
    tr, st_tr = _normalize(dirty_train, feature_cols, categorical_cols, sep, parse_units)
    va, _ = _normalize(clean_val, feature_cols, categorical_cols, sep, parse_units)
    is_clf = task == 'classification'
    if task == 'clustering':
        tr[label_col] = [f'{i}.0E0' for i in range(len(tr))]
        va[label_col] = [f'{i}.0E0' for i in range(len(va))]
    elif is_clf:
        tr[label_col] = dirty_train[label_col].astype(str).str.replace(sep, ' ', regex=False).values
        va[label_col] = clean_val[label_col].astype(str).str.replace(sep, ' ', regex=False).values
    else:
        tr[label_col] = [_sci(v) for v in dirty_train[label_col].astype(float).values]
        va[label_col] = [_sci(v) for v in clean_val[label_col].astype(float).values]
    schema = ['STRING' if c in categorical_cols else 'FP64' for c in feature_cols]
    schema.append('STRING' if is_clf else 'FP64')
    mask = ['1' if c in categorical_cols else '0' for c in feature_cols] + ['1' if is_clf else '0']
    fd_cat = [c for c in feature_cols if c in categorical_cols and c in fd_cols]
    # executePipeline::imputeByFd hands only the categorical block to discoverFD and
    # compacts the FD mask with removeEmpty, so the two only align when the FD mask
    # equals the categorical mask; it is set that way when the rules list an FD
    # between categorical attributes, otherwise it is all zero.
    use_fd = len(fd_cat) >= 2
    fd = [('1' if (use_fd and c in categorical_cols) else '0') for c in feature_cols] + ['0']
    meta = [['Schema'] + schema, ['Mask'] + mask, ['FD'] + fd]
    return {'train': tr, 'val': va, 'meta': meta, 'sep': sep,
            'stats': {'columns': st_tr, 'fd_mask_on': use_fd, 'fd_columns_in_rules': sorted(fd_cols)}}


def _row_keys(X):
    Z = np.where(np.isnan(X), -1.2345e300, X)
    Z = np.round(Z, 9)
    return [tuple(r) for r in Z]


def _match_rows(X_prev, ids_prev, X_cur):
    """Map every row of X_cur to a row of X_prev with identical values.
    Unmatched rows (synthetic SMOTE rows) get id -1."""
    pool = {}
    for pos, k in enumerate(_row_keys(X_prev)):
        pool.setdefault(k, []).append(pos)
    ids = np.full(X_cur.shape[0], -1, dtype=np.int64)
    for r, k in enumerate(_row_keys(X_cur)):
        lst = pool.get(k)
        if lst:
            ids[r] = ids_prev[lst.pop(0)]
    return ids


def _fit_affine(x_in, x_out):
    ok = ~np.isnan(x_in) & ~np.isnan(x_out)
    if ok.sum() == 0:
        return 1.0, 0.0
    a_in, a_out = x_in[ok], x_out[ok]
    if np.ptp(a_in) < 1e-12:
        return 1.0, float(np.mean(a_out - a_in))
    a, b = np.polyfit(a_in, a_out, 1)
    return float(a), float(b)


def _fmt_num(v):
    if np.isnan(v):
        return 'empty'
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return str(int(round(v)))
    return '%.10g' % v


def decode_export(ops, steps, X0, dirty_train, feature_cols, label_col, categorical_cols,
                  index_col='index'):
    """Rebuild the cleaned training table from the DML replay.

    ops: list of primitive names of the top-1 pipeline (in order).
    steps: list of matrices X_i written after primitive i (same order as ops).
    X0: encoded training matrix before the pipeline (row-aligned with dirty_train).
    Returns (cleaned DataFrame, info dict)."""
    n0, d = X0.shape
    assert n0 == len(dirty_train) and d == len(feature_cols), (X0.shape, len(dirty_train), len(feature_cols))
    cat_idx = [j for j, c in enumerate(feature_cols) if c in categorical_cols]
    num_idx = [j for j, c in enumerate(feature_cols) if c not in categorical_cols]
    ids = np.arange(n0, dtype=np.int64)
    X_prev = X0
    affine = []
    log = []
    for op, X_cur in zip(ops, steps):
        if X_cur.shape[1] != d:
            raise ValueError(f'primitive {op} changed the column count {d} -> {X_cur.shape[1]}')
        if op in ENCODER_OPS:
            log.append({'op': op, 'action': 'skipped encoder'})
            X_cur = X_prev
        elif op in ROW_OPS or X_cur.shape[0] != X_prev.shape[0]:
            new_ids = _match_rows(X_prev, ids, X_cur)
            log.append({'op': op, 'action': 'row match', 'rows_in': int(X_prev.shape[0]),
                        'rows_out': int(X_cur.shape[0]), 'unmatched': int((new_ids < 0).sum())})
            ids = new_ids
        elif op in AFFINE_OPS:
            coef = {j: _fit_affine(X_prev[:, j], X_cur[:, j]) for j in num_idx}
            affine.append(coef)
            log.append({'op': op, 'action': 'affine recorded'})
        else:
            log.append({'op': op, 'action': 'applied'})
        X_prev = X_cur
    X = X_prev.copy()
    for coef in reversed(affine):
        for j, (a, b) in coef.items():
            if abs(a) > 1e-15:
                X[:, j] = (X[:, j] - b) / a
    keep = ids >= 0
    X, ids = X[keep], ids[keep]
    # categorical codes -> most frequent original string carrying that code
    code_maps = {}
    for j in cat_idx:
        col = feature_cols[j]
        codes = X0[:, j]
        orig = dirty_train[col].astype(str).values
        m = {}
        for cd, ov in zip(codes, orig):
            if np.isnan(cd) or ov in MISS:
                continue
            m.setdefault(int(round(cd)), {}).setdefault(ov, 0)
            m[int(round(cd))][ov] += 1
        code_maps[j] = {k: max(v.items(), key=lambda t: t[1])[0] for k, v in m.items()}
    out = pd.DataFrame()
    out[index_col] = dirty_train[index_col].values[ids]
    for j, col in enumerate(feature_cols):
        if j in code_maps:
            cm = code_maps[j]
            out[col] = [cm.get(int(round(v)), 'empty') if not np.isnan(v) else 'empty' for v in X[:, j]]
        else:
            out[col] = [_fmt_num(v) for v in X[:, j]]
    out[label_col] = dirty_train[label_col].values[ids]
    info = {'rows_in': int(n0), 'rows_out': int(len(out)), 'rows_deleted': int(n0 - len(np.unique(ids))),
            'synthetic_dropped': int((~keep).sum()), 'steps': log}
    return out, info


def changed_cells(cleaned, dirty_train, feature_cols, categorical_cols, index_col='index'):
    """Count cells of the exported table that differ from the dirty input."""
    dt = dirty_train.set_index(index_col)
    cl = cleaned.set_index(index_col)
    n = 0
    for c in feature_cols:
        a = dt.loc[cl.index, c].astype(str)
        b = cl[c].astype(str)
        if c in categorical_cols:
            a2 = a.where(~a.isin(MISS), 'empty')
            n += int((a2.values != b.values).sum())
        else:
            fa = pd.to_numeric(a.where(~a.isin(MISS), np.nan), errors='coerce').values
            fb = pd.to_numeric(b.where(b != 'empty', np.nan), errors='coerce').values
            both_nan = np.isnan(fa) & np.isnan(fb)
            diff = ~both_nan & ~(np.isclose(fa, fb, rtol=1e-9, atol=1e-9))
            n += int(diff.sum())
    return n
