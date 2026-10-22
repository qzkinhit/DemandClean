"""
Automatic error detector (AutoDetector)
=======================================

Layered multi-strategy error detector:

1. Missing values: np.isnan (deterministic)
2. Syntactic errors: RAHA + REGEX + DOMAIN rules (format, pattern and range anomalies)
3. Semantic errors: FD + CFD (features) + DC rules (violated relations between values)
4. Label noise: CFD label rules

Design:
- syntactic detection uses RAHA, complemented by REGEX and DOMAIN rules (out-of-range values are syntactic anomalies)
- semantic detection is rule driven (FD/CFD/DC) and uses no statistical outlier test
- label noise uses CFD rules only
- detection rates are computed from the overall TP/FP/FN, not averaged over error types
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import os
import re
import sys
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd

from .rule_parser import (
    ParsedRules, DomainRule, RegexRule, CFDRule, DCRule,
    parse_rules_file, load_rules,
    parse_dc_clause,
)

# Import RAHA (a local Baran_Raha checkout first, then the pip package)
RAHA_AVAILABLE = False
_RAHA_SOURCE = "none"

try:
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    _baran_raha_path = os.path.join(_project_root, 'Baran_Raha')
    if os.path.isdir(_baran_raha_path):
        if _baran_raha_path not in sys.path:
            sys.path.insert(0, _baran_raha_path)
        from detection import Detection as _RahaDetection
        RAHA_AVAILABLE = True
        _RAHA_SOURCE = "Baran_Raha/detection.py"
except ImportError:
    pass

if not RAHA_AVAILABLE:
    try:
        from raha.detection import Detection as _RahaDetection
        RAHA_AVAILABLE = True
        _RAHA_SOURCE = "pip:raha"
    except ImportError:
        pass

# RAHA runs its strategies in multiprocessing.Pool() sized to every CPU of the host.
# DC_RAHA_PROCS caps that pool so many concurrent runs do not oversubscribe a server.
# The cap changes scheduling only, not the strategies RAHA runs or their outputs.
if RAHA_AVAILABLE and os.environ.get('DC_RAHA_PROCS'):
    import multiprocessing as _mp
    _uncapped_pool = _mp.Pool

    def _capped_pool(processes=None, *args, **kwargs):
        return _uncapped_pool(processes or int(os.environ['DC_RAHA_PROCS']), *args, **kwargs)

    _mp.Pool = _capped_pool


class RahaHelper:
    """
    RAHA helper that runs RAHA detection directly on the raw CSV.

    - takes the raw CSV path, no numpy-to-CSV conversion
    - runs RAHA once (RAHA itself iterates over its labelling rounds)
    - splits by columns (more than max_cols columns) or by rows (more than max_rows rows)
    """

    # Chunking thresholds
    MAX_COLS_PER_CHUNK = 20
    MAX_ROWS_PER_CHUNK = 10000

    @staticmethod
    def run_on_csv(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str = "data",
        labeling_budget: int = 20,
        verbose: bool = False,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """
        Run RAHA detection on the raw CSV.

        Chunking is chosen automatically:
        - more than MAX_COLS_PER_CHUNK columns: column chunks
        - more than MAX_ROWS_PER_CHUNK rows: row chunks
        - both: column chunks first, then row chunks (only one dimension is chunked at a time)

        Returns:
            (detected_cells, labeled_tuple_indices, chunk_info)
            detected_cells: {(row, col): "JUST A DUMMY VALUE", ...}
            labeled_tuple_indices: set of row indices labelled for RAHA
            chunk_info: {'n_col_chunks': int, 'n_row_chunks': int,
                         'labeling_budget': int, 'raha_total_cost': int}
        """
        if not RAHA_AVAILABLE:
            if verbose:
                print("  [warn] RAHA is not available, detection skipped")
            return {}, set(), {'n_col_chunks': 0, 'n_row_chunks': 0,
                               'labeling_budget': labeling_budget, 'raha_total_cost': 0}

        # Read the data to decide on chunking
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False, nrows=5)
        n_cols = len(dirty_df.columns)
        # Row count without loading the full table
        with open(dirty_csv_path, 'r') as f:
            n_rows = sum(1 for _ in f) - 1  # Minus the header

        if verbose:
            print(f"  [RAHA] data size: {n_rows} rows x {n_cols} columns")

        if n_cols > RahaHelper.MAX_COLS_PER_CHUNK:
            if verbose:
                print(f"  [RAHA] {n_cols} columns > {RahaHelper.MAX_COLS_PER_CHUNK}, column chunks")
            return RahaHelper._run_column_split(
                dirty_csv_path, clean_csv_path, dataset_name,
                labeling_budget, verbose,
                fd_rules=fd_rules, primary_key=primary_key, label_col=label_col,
            )
        elif n_rows > RahaHelper.MAX_ROWS_PER_CHUNK:
            if verbose:
                print(f"  [RAHA] {n_rows} rows > {RahaHelper.MAX_ROWS_PER_CHUNK}, row chunks")
            # Try grouping-based chunks
            grouping_key = RahaHelper._infer_grouping_key(
                dirty_df=pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False),
                fd_rules=fd_rules,
                primary_key=primary_key,
                label_col=label_col,
            )
            if grouping_key:
                if verbose:
                    print(f"  [RAHA] grouping key: {grouping_key}, grouped chunks")
                return RahaHelper._run_row_split_clustered(
                    dirty_csv_path, clean_csv_path,
                    dataset_name, labeling_budget, verbose,
                    grouping_key=grouping_key,
                )
            else:
                if verbose:
                    print(f"  [RAHA] no grouping key found, equal-size chunks")
                return RahaHelper._run_row_split(
                    dirty_csv_path, clean_csv_path,
                    dataset_name, labeling_budget, verbose, n_rows,
                )
        else:
            detected, labeled = RahaHelper._run_single(
                dirty_csv_path, clean_csv_path, dataset_name,
                labeling_budget, verbose,
            )
            chunk_info = {
                'n_col_chunks': 1,
                'n_row_chunks': 1,
                'labeling_budget': labeling_budget,
                'raha_total_cost': labeling_budget,
            }
            return detected, labeled, chunk_info

    @staticmethod
    def _run_column_split(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """Run RAHA on column chunks of at most MAX_COLS_PER_CHUNK columns.

        - split the feature columns into batches of at most MAX_COLS_PER_CHUNK columns
        - write a sub-CSV per batch and run RAHA on it
        - merge detected_cells (column indices mapped back to the original columns)
        - labeled_tuples is the union over batches
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)

        all_cols = list(dirty_df.columns)
        max_per_chunk = RahaHelper.MAX_COLS_PER_CHUNK

        # Split the columns into batches
        n_col_chunks = (len(all_cols) + max_per_chunk - 1) // max_per_chunk
        if verbose:
            print(f"  [column chunks] {len(all_cols)} columns in {n_col_chunks} batches (at most {max_per_chunk} columns each)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_colsplit_{dataset_name}_')
        total_row_chunks = 0

        try:
            for chunk_idx in range(n_col_chunks):
                start = chunk_idx * max_per_chunk
                end = min(start + max_per_chunk, len(all_cols))
                chunk_cols = all_cols[start:end]

                if verbose:
                    print(f"  [column chunk {chunk_idx+1}/{n_col_chunks}] columns {start}..{end-1}: {chunk_cols[0]}...{chunk_cols[-1]}")

                # Sub-CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_chunk{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_chunk{chunk_idx}.csv')
                dirty_df[chunk_cols].to_csv(chunk_dirty_path, index=False)
                clean_df[chunk_cols].to_csv(chunk_clean_path, index=False)

                # Run RAHA (nested row chunks if the row count also exceeds the threshold)
                n_rows_chunk = len(dirty_df)
                if n_rows_chunk > RahaHelper.MAX_ROWS_PER_CHUNK:
                    if verbose:
                        print(f"  [column chunk {chunk_idx+1}] {n_rows_chunk} rows still above the threshold, nested row chunks")
                    # Try grouping-based chunks (grouping key inferred on the sub-CSV)
                    chunk_dirty_df = pd.read_csv(chunk_dirty_path, dtype=str, keep_default_na=False)
                    chunk_grouping_key = RahaHelper._infer_grouping_key(
                        dirty_df=chunk_dirty_df,
                        fd_rules=fd_rules,
                        primary_key=primary_key,
                        label_col=label_col,
                    )
                    if chunk_grouping_key:
                        if verbose:
                            print(f"  [column chunk {chunk_idx+1}] grouping key: {chunk_grouping_key}")
                        chunk_detected, chunk_labeled, sub_chunk_info = RahaHelper._run_row_split_clustered(
                            chunk_dirty_path, chunk_clean_path,
                            f"{dataset_name}_chunk{chunk_idx}",
                            labeling_budget, verbose,
                            grouping_key=chunk_grouping_key,
                        )
                    else:
                        chunk_detected, chunk_labeled, sub_chunk_info = RahaHelper._run_row_split(
                            chunk_dirty_path, chunk_clean_path,
                            f"{dataset_name}_chunk{chunk_idx}",
                            labeling_budget, verbose, n_rows_chunk,
                        )
                    total_row_chunks = max(total_row_chunks, sub_chunk_info['n_row_chunks'])
                else:
                    chunk_detected, chunk_labeled = RahaHelper._run_single(
                        chunk_dirty_path, chunk_clean_path,
                        f"{dataset_name}_chunk{chunk_idx}",
                        labeling_budget, verbose,
                    )
                    total_row_chunks = max(total_row_chunks, 1)

                # Map column indices back to the original columns
                for (row, chunk_col), val in chunk_detected.items():
                    original_col = start + chunk_col
                    merged_detected[(row, original_col)] = val

                merged_labeled |= chunk_labeled

                if verbose:
                    print(f"  [column chunk {chunk_idx+1}] detected: {len(chunk_detected)}, labelled rows: {len(chunk_labeled)}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [column chunks] merged: {len(merged_detected)} detected, {len(merged_labeled)} labelled rows")

        n_row_chunks_final = max(total_row_chunks, 1)
        raha_total_cost = labeling_budget * n_col_chunks * n_row_chunks_final
        chunk_info = {
            'n_col_chunks': n_col_chunks,
            'n_row_chunks': n_row_chunks_final,
            'labeling_budget': labeling_budget,
            'raha_total_cost': raha_total_cost,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _run_row_split(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        n_rows: int,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """Run RAHA on row chunks of at most MAX_ROWS_PER_CHUNK rows.

        - equal-size chunks (no primary-key grouping, a dataset need not have a key)
        - write a sub-CSV per chunk and run RAHA on it
        - merge detected_cells (row indices mapped back to the original rows)
        - labeled_tuples is the union over chunks (mapped back to the original rows)
        - as few chunks as possible, since every chunk consumes labels
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)

        max_per_chunk = RahaHelper.MAX_ROWS_PER_CHUNK
        n_chunks = (n_rows + max_per_chunk - 1) // max_per_chunk

        # Few chunks, because every chunk consumes 20 ground-truth labels
        # Too many chunks: enlarge the chunk size
        while n_chunks > 4 and max_per_chunk < n_rows:
            max_per_chunk = int(max_per_chunk * 1.5)
            n_chunks = (n_rows + max_per_chunk - 1) // max_per_chunk

        if verbose:
            print(f"  [row chunks] {n_rows} rows in {n_chunks} chunks (at most {max_per_chunk} rows each)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_rowsplit_{dataset_name}_')

        try:
            for chunk_idx in range(n_chunks):
                start = chunk_idx * max_per_chunk
                end = min(start + max_per_chunk, n_rows)

                if verbose:
                    print(f"  [row chunk {chunk_idx+1}/{n_chunks}] rows {start}..{end-1} ({end-start} rows)")

                # Sub-CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_row{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_row{chunk_idx}.csv')
                dirty_df.iloc[start:end].to_csv(chunk_dirty_path, index=False)
                clean_df.iloc[start:end].to_csv(chunk_clean_path, index=False)

                # Run RAHA
                chunk_detected, chunk_labeled = RahaHelper._run_single(
                    chunk_dirty_path, chunk_clean_path,
                    f"{dataset_name}_row{chunk_idx}",
                    labeling_budget, verbose,
                )

                # Map row indices back to the original rows
                for (chunk_row, col), val in chunk_detected.items():
                    original_row = start + chunk_row
                    merged_detected[(original_row, col)] = val

                # Map labeled_tuples back to the original rows
                for chunk_row in chunk_labeled:
                    merged_labeled.add(start + chunk_row)

                if verbose:
                    print(f"  [row chunk {chunk_idx+1}] detected: {len(chunk_detected)}, labelled rows: {len(chunk_labeled)}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [row chunks] merged: {len(merged_detected)} detected, {len(merged_labeled)} labelled rows")

        chunk_info = {
            'n_col_chunks': 1,
            'n_row_chunks': n_chunks,
            'labeling_budget': labeling_budget,
            'raha_total_cost': labeling_budget * n_chunks,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _infer_grouping_key(
        dirty_df: pd.DataFrame,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Infer the column that suits grouping-based chunking best.

        Priority:
        1. an explicit primary key (rules.txt [PRIMARY_KEY])
        2. the most frequent LHS column of the FD rules
        3. automatic inference (categorical column with cardinality ratio between 0.01 and 0.5)
        4. otherwise None (equal-size chunks)
        """
        # 1. Explicit primary key
        if primary_key:
            valid = [c for c in primary_key if c in dirty_df.columns]
            if valid:
                return valid

        # 2. LHS columns of the FDs
        if fd_rules:
            from collections import Counter
            lhs_counter = Counter()
            for lhs_str, rhs_str in fd_rules:
                for col in lhs_str.split(','):
                    col = col.strip()
                    if col in dirty_df.columns:
                        lhs_counter[col] += 1
            if lhs_counter:
                # Most frequent LHS column
                best_col = lhs_counter.most_common(1)[0][0]
                # Cardinality check
                nunique = dirty_df[best_col].nunique()
                nrows = len(dirty_df)
                ratio = nunique / nrows if nrows > 0 else 1
                if 0.001 <= ratio <= 0.8:
                    return [best_col]

        # 3. Automatic inference
        candidates = []
        nrows = len(dirty_df)
        for col in dirty_df.columns:
            if col == label_col:
                continue
            nunique = dirty_df[col].nunique()
            ratio = nunique / nrows if nrows > 0 else 1
            # Neither near-constant nor a unique identifier
            if 0.01 <= ratio <= 0.5:
                # Prefer columns that look categorical (nunique < 100)
                score = 0
                if nunique <= 100:
                    score += 2
                if ratio <= 0.1:
                    score += 1
                # Prefer non-numeric columns
                try:
                    dirty_df[col].astype(float)
                except (ValueError, TypeError):
                    score += 1
                candidates.append((col, score, nunique))

        if candidates:
            candidates.sort(key=lambda x: (-x[1], x[2]))
            return [candidates[0][0]]

        return None  # Fall back to equal-size chunks

    @staticmethod
    def _run_row_split_clustered(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        grouping_key: List[str],
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """Group rows by the grouping key, pack the groups into chunks and run RAHA.

        - group by grouping_key
        - merge small groups with first-fit decreasing bin packing
        - at most MAX_ROWS_PER_CHUNK rows per chunk
        - at most 4 chunks
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)
        max_size = RahaHelper.MAX_ROWS_PER_CHUNK

        # Group by grouping_key (NaN filled so that no group is lost)
        if len(grouping_key) == 1:
            group_col = grouping_key[0]
            grouped = dirty_df.fillna({group_col: '__NA__'}).groupby(group_col)
        else:
            fill_dict = {c: '__NA__' for c in grouping_key}
            grouped = dirty_df.fillna(fill_dict).groupby(grouping_key)

        group_info = [(key, list(indices)) for key, indices in grouped.groups.items()]
        group_info.sort(key=lambda x: len(x[1]), reverse=True)

        # First-Fit Decreasing Bin Packing
        chunks = []  # Each chunk is a list of (key, [indices])
        chunk_sizes = []

        for key, indices in group_info:
            size = len(indices)
            if size > max_size:
                # A group above the limit becomes its own chunk
                chunks.append([(key, indices)])
                chunk_sizes.append(size)
            else:
                placed = False
                for i in range(len(chunks)):
                    if chunk_sizes[i] + size <= max_size:
                        chunks[i].append((key, indices))
                        chunk_sizes[i] += size
                        placed = True
                        break
                if not placed:
                    chunks.append([(key, indices)])
                    chunk_sizes.append(size)

        # At most 4 chunks
        while len(chunks) > 4:
            # Merge the two smallest chunks
            min_idx = sorted(range(len(chunks)), key=lambda i: chunk_sizes[i])[:2]
            i1, i2 = min_idx[0], min_idx[1]
            chunks[i1].extend(chunks[i2])
            chunk_sizes[i1] += chunk_sizes[i2]
            del chunks[i2]
            del chunk_sizes[i2]

        if verbose:
            print(f"  [grouped chunks] grouped by {grouping_key}, {len(group_info)} groups packed into {len(chunks)} chunks")
            for ci, (chunk, size) in enumerate(zip(chunks, chunk_sizes)):
                print(f"  [grouped chunk {ci+1}/{len(chunks)}] {size} rows ({len(chunk)} groups)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_clustered_{dataset_name}_')

        try:
            for chunk_idx, chunk in enumerate(chunks):
                # All row indices of the chunk
                all_indices = []
                for key, indices in chunk:
                    all_indices.extend(indices)
                all_indices.sort()

                if verbose:
                    print(f"  [grouped chunk {chunk_idx+1}/{len(chunks)}] {len(all_indices)} rows")

                # Sub-CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_cluster{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_cluster{chunk_idx}.csv')
                dirty_df.iloc[all_indices].to_csv(chunk_dirty_path, index=False)
                clean_df.iloc[all_indices].to_csv(chunk_clean_path, index=False)

                # Run RAHA
                chunk_detected, chunk_labeled = RahaHelper._run_single(
                    chunk_dirty_path, chunk_clean_path,
                    f"{dataset_name}_cluster{chunk_idx}",
                    labeling_budget, verbose,
                )

                # Map row indices back to the original rows
                for (chunk_row, col), val in chunk_detected.items():
                    original_row = all_indices[chunk_row]
                    merged_detected[(original_row, col)] = val

                for chunk_row in chunk_labeled:
                    if chunk_row < len(all_indices):
                        merged_labeled.add(all_indices[chunk_row])

                if verbose:
                    print(f"  [grouped chunk {chunk_idx+1}] detected: {len(chunk_detected)}, labelled rows: {len(chunk_labeled)}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [grouped chunks] merged: {len(merged_detected)} detected, {len(merged_labeled)} labelled rows")

        chunk_info = {
            'n_col_chunks': 1,
            'n_row_chunks': len(chunks),
            'labeling_budget': labeling_budget,
            'raha_total_cost': labeling_budget * len(chunks),
            'grouping_key': grouping_key,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _run_single(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int]]:
        """Run RAHA once (no chunking).

        The dirty and clean CSVs are normalised cell by cell, so numeric format differences
        such as "1.0" vs "1" are not reported as errors.
        """
        # results_dir carries the pid suffix as raha_name does
        raha_name = f"{dataset_name}_p{os.getpid()}"
        results_dir = os.path.join(
            os.path.dirname(dirty_csv_path),
            f"raha-baran-results-{raha_name}"
        )
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)

        normalized_dirty_path = None
        normalized_clean_path = None
        try:
            from demandclean.tools.csv_normalizer import normalize_cell_format
            dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
            dirty_norm = normalize_cell_format(dirty_df, verbose=False)
            fd, normalized_dirty_path = tempfile.mkstemp(
                suffix='.csv', prefix='raha_norm_dirty_'
            )
            os.close(fd)
            dirty_norm.to_csv(normalized_dirty_path, index=False)

            clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)
            clean_norm = normalize_cell_format(clean_df, verbose=False)
            fd, normalized_clean_path = tempfile.mkstemp(
                suffix='.csv', prefix='raha_norm_clean_'
            )
            os.close(fd)
            clean_norm.to_csv(normalized_clean_path, index=False)

            actual_dirty_path = normalized_dirty_path
            actual_clean_path = normalized_clean_path
        except Exception:
            actual_dirty_path = dirty_csv_path
            actual_clean_path = clean_csv_path

        # RAHA builds /tmp/{name}-{hash}.csv from the name, so equal names collide across processes
        dataset_dictionary = {
            "name": raha_name,
            "path": actual_dirty_path,
            "clean_path": actual_clean_path,
        }

        try:
            app = _RahaDetection()
            app.LABELING_BUDGET = labeling_budget
            app.VERBOSE = verbose
            app.SAVE_RESULTS = False

            # run() is overridden to keep the dataset object d (the original returns only detected_cells)
            d = app.initialize_dataset(dataset_dictionary)
            app.run_strategies(d)
            app.generate_features(d)
            app.build_clusters(d)
            while len(d.labeled_tuples) < app.LABELING_BUDGET:
                app.sample_tuple(d)
                if d.has_ground_truth:
                    app.label_with_ground_truth(d)
            app.propagate_labels(d)
            app.predict_labels(d)
            detected_cells = d.detected_cells

            labeled_tuple_indices = set()
            if hasattr(d, 'labeled_tuples') and d.labeled_tuples:
                labeled_tuple_indices = set(d.labeled_tuples.keys())
            try:
                eval_result = d.get_data_cleaning_evaluation(detected_cells)
                p, r, f1 = eval_result[:3]
                print(f"  RAHA detected: {len(detected_cells)} cells")
                print(f"  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
            except Exception:
                print(f"  RAHA detected: {len(detected_cells)} cells")

            if labeled_tuple_indices:
                print(f"  RAHA labelled rows: {len(labeled_tuple_indices)} (labeling_budget={labeling_budget})")

            return (detected_cells if detected_cells else {}, labeled_tuple_indices)

        except Exception as e:
            print(f"  [error] RAHA failed: {e}")
            import traceback
            traceback.print_exc()
            return {}, set()
        finally:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir, ignore_errors=True)
            for tmp_path in [normalized_dirty_path, normalized_clean_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    @staticmethod
    def map_raha_cells_to_feature_indices(
        detected_cells: Dict[Tuple[int, int], str],
        csv_columns: List[str],
        feature_columns: List[str],
        label_col: str,
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Map RAHA detections (row, csv_col) to feature columns (row, feat_col).

        Returns:
            feature_errors: [(row, feat_col_idx), ...]
            label_error_rows: [row, ...]
        """
        feature_errors = []
        label_error_rows = []

        feat_name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}

        for (row, csv_col_idx), _ in detected_cells.items():
            if csv_col_idx < len(csv_columns):
                col_name = csv_columns[csv_col_idx]
                if col_name in feat_name_to_idx:
                    feature_errors.append((row, feat_name_to_idx[col_name]))
                elif col_name == label_col:
                    label_error_rows.append(row)

        return feature_errors, label_error_rows


class RuleBasedDetector:
    """
    Rule detector that applies REGEX/DOMAIN/CFD/DC rules to the raw CSV (dirty_df).

    Every method is static, takes dirty_df and rule objects and returns a set of error cells.
    """

    # ------------------------------------------------------------------
    # (a) REGEX detection, complements the syntactic detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_regex(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        regex_rules: List[RegexRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Set[Tuple[int, int]]:
        """
        Detect syntactic errors with REGEX rules.

        Args:
            dirty_df: raw dirty DataFrame (str dtype)
            feature_cols: feature column names
            regex_rules: REGEX rules
            excluded_cells: cells already detected (not marked again)
            feat_name_to_idx: feature column name -> index
            verbose: print log lines

        Returns:
            Set[(row, col_idx)], counted as syntactic errors
        """
        detected: Set[Tuple[int, int]] = set()

        for rule in regex_rules:
            # Target columns
            if rule.column == 'ALL_FEATURES':
                target_cols = feature_cols
            elif rule.column in feat_name_to_idx:
                target_cols = [rule.column]
            else:
                continue

            try:
                pattern = re.compile(rule.pattern)
            except re.error:
                continue

            for col_name in target_cols:
                col_idx = feat_name_to_idx.get(col_name)
                if col_idx is None:
                    continue

                if col_name not in dirty_df.columns:
                    continue

                for row_idx in range(len(dirty_df)):
                    if (row_idx, col_idx) in excluded_cells or (row_idx, col_idx) in detected:
                        continue
                    val = dirty_df.iloc[row_idx][col_name]
                    if pd.isna(val) or str(val).strip() == '':
                        continue
                    if pattern.search(str(val)):
                        detected.add((row_idx, col_idx))

        if verbose and detected:
            print(f"  REGEX rules: {len(detected)} syntactic errors")

        return detected

    # ------------------------------------------------------------------
    # (b) DOMAIN detection, syntactic errors (out-of-range values)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_domain(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        domain_rules: List[DomainRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Detect syntactic errors with DOMAIN rules (out-of-range values).

        Returns:
            (syntactic_cells, label_cells)
            syntactic_cells: Set[(row, col_idx)], out-of-domain feature cells
            label_cells: Set[(row, -1)], out-of-domain labels
        """
        semantic: Set[Tuple[int, int]] = set()
        label: Set[Tuple[int, int]] = set()

        for rule in domain_rules:
            col_name = rule.column
            is_label = (col_name == label_col)

            if col_name not in dirty_df.columns:
                continue

            col_idx = feat_name_to_idx.get(col_name) if not is_label else -1

            for row_idx in range(len(dirty_df)):
                cell_key = (row_idx, col_idx)
                if cell_key in excluded_cells:
                    continue
                if not is_label and cell_key in semantic:
                    continue
                if is_label and cell_key in label:
                    continue

                val = dirty_df.iloc[row_idx][col_name]
                if pd.isna(val) or str(val).strip() == '':
                    continue

                val_str = str(val).strip()

                # Skip common placeholders (same list as the replacement in preprocess_data)
                if val_str.lower() in (
                    'empty', 'nan', 'null', 'none', 'n/a', 'na', '?', '-',
                ):
                    continue

                is_violation = False

                if rule.dtype in ('INT', 'FLOAT'):
                    try:
                        num_val = float(val_str)
                        if num_val < rule.min_val or num_val > rule.max_val:
                            is_violation = True
                    except (ValueError, TypeError):
                        # Not numeric: skip (a data type issue, not a range violation)
                        continue
                elif rule.dtype == 'ENUM':
                    if val_str not in rule.enum_vals:
                        # Integer ENUM in float data: str(0.0)='0.0' does not match '0'
                        # Numeric comparison: a value that is numeric and matches after rounding is no violation
                        try:
                            num_val = float(val_str)
                            # Is some enum value numerically equal
                            matched = any(
                                abs(num_val - float(ev)) < 1e-9
                                for ev in rule.enum_vals
                                if ev.replace('.', '', 1).replace('-', '', 1).isdigit()
                            )
                            if not matched:
                                is_violation = True
                        except (ValueError, TypeError):
                            is_violation = True

                if is_violation:
                    if is_label:
                        label.add(cell_key)
                    else:
                        semantic.add(cell_key)

        if verbose:
            if semantic:
                print(f"  DOMAIN rules: {len(semantic)} syntactic errors (feature columns, out of range)")
            if label:
                print(f"  DOMAIN rules: {len(label)} label errors")

        return semantic, label

    # ------------------------------------------------------------------
    # (c) CFD detection, semantic and label errors
    # ------------------------------------------------------------------
    @staticmethod
    def detect_cfd(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        cfd_rules: List[CFDRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Detect errors with CFD rules.

        Returns:
            (semantic_cells, label_cells)
            semantic_cells: CFD violations in feature columns
            label_cells: CFD violations of the label column (row, -1)
        """
        semantic: Set[Tuple[int, int]] = set()
        label: Set[Tuple[int, int]] = set()

        if not cfd_rules:
            return semantic, label

        n_rows = len(dirty_df)

        # Precompute n_anomaly (breast_cancer: number of columns of a row outside the benign/malignant typical range),
        # which CFD conditions can reference as the variable n_anomaly
        # n_anomaly is computed only when a condition references it
        need_n_anomaly = any(
            any(col == 'n_anomaly' for col, _, _ in rule.conditions)
            for rule in cfd_rules
        )

        n_anomaly_map: Dict[int, int] = {}
        if need_n_anomaly:
            n_anomaly_map = RuleBasedDetector._compute_n_anomaly(
                dirty_df, feature_cols, label_col
            )

        for rule in cfd_rules:
            is_label_rule = (rule.target_col == label_col)
            target_col_idx = -1 if is_label_rule else feat_name_to_idx.get(rule.target_col)

            if target_col_idx is None:
                continue

            for row_idx in range(n_rows):
                cell_key = (row_idx, target_col_idx)
                if cell_key in excluded_cells:
                    continue
                if is_label_rule and cell_key in label:
                    continue
                if not is_label_rule and cell_key in semantic:
                    continue

                # Check that every condition holds
                conditions_met = True
                for cond_col, cond_op, cond_val in rule.conditions:
                    # n_anomaly condition
                    if cond_col == 'n_anomaly':
                        actual_val = n_anomaly_map.get(row_idx, 0)
                        if not RuleBasedDetector._check_numeric_condition(
                            actual_val, cond_op, float(cond_val)
                        ):
                            conditions_met = False
                            break
                        continue

                    if cond_col not in dirty_df.columns:
                        conditions_met = False
                        break

                    row_val = dirty_df.iloc[row_idx][cond_col]
                    if pd.isna(row_val):
                        conditions_met = False
                        break

                    row_val_str = str(row_val).strip()

                    # Numeric condition
                    if cond_op in ('<=', '>=', '<', '>', '!='):
                        try:
                            actual_num = float(row_val_str)
                            cond_num = float(cond_val)
                            if not RuleBasedDetector._check_numeric_condition(
                                actual_num, cond_op, cond_num
                            ):
                                conditions_met = False
                                break
                        except (ValueError, TypeError):
                            # String comparison
                            if cond_op == '!=' and row_val_str == cond_val:
                                conditions_met = False
                                break
                            elif cond_op == '=' and row_val_str != cond_val:
                                conditions_met = False
                                break
                    elif cond_op == '=':
                        if row_val_str != cond_val:
                            conditions_met = False
                            break

                if not conditions_met:
                    continue

                # Deviation of the target column
                target_col_name = rule.target_col
                if target_col_name not in dirty_df.columns:
                    continue

                target_val = dirty_df.iloc[row_idx][target_col_name]
                if pd.isna(target_val):
                    continue

                try:
                    target_num = float(str(target_val).strip())
                except (ValueError, TypeError):
                    continue

                deviation = target_num - rule.baseline
                is_violation = False

                if rule.direction == 'EXCESS':
                    if deviation >= rule.threshold:
                        is_violation = True
                elif rule.direction == 'DEFICIT':
                    if -deviation >= rule.threshold:
                        is_violation = True

                if is_violation:
                    if is_label_rule:
                        label.add(cell_key)
                    else:
                        semantic.add(cell_key)

        # Infer label errors from CFD conditions that fix the label
        # A row is a label error only when its CFDs flag at least 2 deviating columns
        # and few of its features are marked as semantic or syntactic errors.
        # Reason: if class=2 but several features look malignant and these features are not errors
        # (no other detector marks them), the label is probably wrong.
        if label_col:
            # Number of CFD semantic errors per row
            cfd_row_counts: Dict[int, int] = {}
            for (r, c) in semantic:
                cfd_row_counts[r] = cfd_row_counts.get(r, 0) + 1

            for row_idx, cfd_count in cfd_row_counts.items():
                # At least 2 deviating feature columns flagged by CFDs
                if cfd_count < 2:
                    continue
                label_key = (row_idx, -1)
                if label_key in label or label_key in excluded_cells:
                    continue
                # Does a CFD condition of the row involve the label column
                # (every breast_cancer CFD contains class=X)
                # Check whether any CFD rule has a label condition
                has_label_cond = any(
                    any(c == label_col and op == '='
                        for c, op, _ in rule.conditions)
                    for rule in cfd_rules
                )
                if has_label_cond:
                    label.add(label_key)

        if verbose:
            if semantic:
                print(f"  CFD rules: {len(semantic)} semantic errors (feature columns)")
            if label:
                print(f"  CFD rules: {len(label)} label errors")

        return semantic, label

    # ------------------------------------------------------------------
    # (d) DC detection, semantic errors
    # ------------------------------------------------------------------
    @staticmethod
    def detect_dc(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        dc_rules: List[DCRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, int]]:
        """
        Detect errors with DC (denial constraint) rules.

        Takes structured DCRule objects parsed by rule_parser.
        DCs have denial semantics: the constraint is violated when every clause holds.

        For an abs_diff rule without MARK only the column that deviates more from its median
        is marked (statistics of the dirty data, the clean data is not used).

        Returns:
            (detected, rule_hit_counts)
            - detected: Set[(row, col_idx)], counted as semantic errors
            - rule_hit_counts: {rule_index: hit_count}, rows hit per rule
        """
        detected: Set[Tuple[int, int]] = set()
        rule_hit_counts: Dict[int, int] = {}

        # Column medians (single-column marking for abs_diff)
        col_medians: Dict[str, float] = {}
        for col_name in feature_cols:
            if col_name in dirty_df.columns:
                try:
                    vals = pd.to_numeric(dirty_df[col_name], errors='coerce')
                    col_medians[col_name] = float(vals.median())
                except Exception:
                    pass

        for rule_idx, dc_rule in enumerate(dc_rules):
            parsed_clauses = dc_rule.clauses
            if not parsed_clauses:
                continue

            # abs_diff rule without MARK
            is_abs_diff_no_mark = (
                not dc_rule.mark_cols
                and len(parsed_clauses) == 1
                and parsed_clauses[0].get('type') == 'abs_diff'
            )

            # With a MARK clause only the MARK columns are marked, otherwise every involved column
            target_cols = dc_rule.mark_cols if dc_rule.mark_cols else dc_rule.involved_cols
            hit_count = 0

            # Check every row for a violation
            for row_idx in range(len(dirty_df)):
                all_clauses_met = True
                for clause in parsed_clauses:
                    if not RuleBasedDetector._evaluate_dc_clause(
                        clause, dirty_df, row_idx
                    ):
                        all_clauses_met = False
                        break

                if all_clauses_met:
                    hit_count += 1

                    if is_abs_diff_no_mark:
                        # abs_diff without MARK: mark only the column that deviates more from its median
                        clause = parsed_clauses[0]
                        col1_name, col2_name = clause['col1'], clause['col2']
                        try:
                            v1 = float(str(dirty_df.iloc[row_idx][col1_name]).strip())
                            v2 = float(str(dirty_df.iloc[row_idx][col2_name]).strip())
                        except (ValueError, TypeError):
                            continue

                        med1 = col_medians.get(col1_name, v1)
                        med2 = col_medians.get(col2_name, v2)
                        dev1 = abs(v1 - med1)
                        dev2 = abs(v2 - med2)

                        # Mark the column with the larger deviation
                        flag_col = col1_name if dev1 >= dev2 else col2_name
                        col_idx = feat_name_to_idx.get(flag_col)
                        if col_idx is not None:
                            cell_key = (row_idx, col_idx)
                            if cell_key not in excluded_cells:
                                detected.add(cell_key)
                    else:
                        # MARK or other clause types: mark every target column
                        for col_name in target_cols:
                            if col_name == label_col:
                                col_idx = -1
                            else:
                                col_idx = feat_name_to_idx.get(col_name)
                            if col_idx is not None:
                                cell_key = (row_idx, col_idx)
                                if cell_key not in excluded_cells:
                                    detected.add(cell_key)

            rule_hit_counts[rule_idx] = hit_count

        if verbose and detected:
            print(f"  DC rules: {len(detected)} semantic errors")

        return detected, rule_hit_counts

    # ------------------------------------------------------------------
    # DC helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dc_clause(clause_str: str) -> Optional[Dict[str, Any]]:
        """Parse one DC clause (delegates to the shared rule_parser.parse_dc_clause)."""
        return parse_dc_clause(clause_str)

    @staticmethod
    def _evaluate_dc_clause(
        clause: Dict[str, Any],
        dirty_df: pd.DataFrame,
        row_idx: int,
    ) -> bool:
        """Evaluate one DC clause on a row, True if the clause holds."""
        clause_type = clause['type']

        if clause_type == 'abs_diff':
            col1, col2 = clause['col1'], clause['col2']
            if col1 not in dirty_df.columns or col2 not in dirty_df.columns:
                return False
            try:
                v1 = float(str(dirty_df.iloc[row_idx][col1]).strip())
                v2 = float(str(dirty_df.iloc[row_idx][col2]).strip())
            except (ValueError, TypeError):
                return False
            actual = abs(v1 - v2)
            return RuleBasedDetector._dc_compare(actual, clause['op'], clause['value'])

        elif clause_type == 'simple':
            col = clause['col']
            if col not in dirty_df.columns:
                return False
            val = dirty_df.iloc[row_idx][col]
            if pd.isna(val):
                return False
            try:
                actual = float(str(val).strip())
            except (ValueError, TypeError):
                return False
            return RuleBasedDetector._dc_compare(actual, clause['op'], clause['value'])

        elif clause_type == 'simple_str':
            col = clause['col']
            if col not in dirty_df.columns:
                return False
            val = dirty_df.iloc[row_idx][col]
            if pd.isna(val):
                return False
            actual = str(val).strip()
            expected = str(clause['value']).strip()
            op = clause['op']
            if op in ('EQ', 'IQ'):
                return actual == expected
            elif op == 'NEQ':
                return actual != expected
            return False

        return False

    @staticmethod
    def _dc_compare(actual: float, op: str, expected: float) -> bool:
        """Numeric DC comparison."""
        if op == 'GT':
            return actual > expected
        elif op == 'GTE':
            return actual >= expected
        elif op == 'LT':
            return actual < expected
        elif op == 'LTE':
            return actual <= expected
        elif op in ('EQ', 'IQ'):
            return abs(actual - expected) < 1e-9
        elif op == 'NEQ':
            return abs(actual - expected) > 1e-9
        return False

    # ------------------------------------------------------------------
    # CFD helper: n_anomaly
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_n_anomaly(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
    ) -> Dict[int, int]:
        """Number of columns of each row outside the typical range (n_anomaly).

        Used by the n_anomaly condition of CFD rules.
        The typical ranges are the P5/P95 per class value.
        They are currently fixed for breast_cancer.
        """
        # Typical ranges of breast_cancer (P5/P95 of the clean data)
        benign_upper = {
            'Clump Thickness': 5, 'Uniformity of Cell Size': 3,
            'Uniformity of Cell Shape': 3, 'Marginal Adhesion': 3,
            'Single Epithelial Cell Size': 3, 'Bare Nuclei': 3,
            'Bland Chromatin': 3, 'Normal Nucleoli': 3, 'Mitoses': 1,
        }
        malignant_lower = {
            'Clump Thickness': 4, 'Uniformity of Cell Size': 3,
            'Uniformity of Cell Shape': 3, 'Bland Chromatin': 3,
        }

        n_anomaly_map: Dict[int, int] = {}

        if label_col and label_col in dirty_df.columns:
            for row_idx in range(len(dirty_df)):
                class_val = str(dirty_df.iloc[row_idx].get(label_col, '')).strip()
                count = 0

                if class_val == '2':
                    for col, upper in benign_upper.items():
                        if col in dirty_df.columns and col in feature_cols:
                            try:
                                v = float(str(dirty_df.iloc[row_idx][col]).strip())
                                if v > upper:
                                    count += 1
                            except (ValueError, TypeError):
                                pass
                elif class_val == '4':
                    for col, lower in malignant_lower.items():
                        if col in dirty_df.columns and col in feature_cols:
                            try:
                                v = float(str(dirty_df.iloc[row_idx][col]).strip())
                                if v < lower:
                                    count += 1
                            except (ValueError, TypeError):
                                pass

                n_anomaly_map[row_idx] = count

        return n_anomaly_map

    @staticmethod
    def _check_numeric_condition(actual: float, op: str, expected: float) -> bool:
        """Generic numeric condition check."""
        if op == '=':
            return abs(actual - expected) < 1e-9
        elif op == '!=':
            return abs(actual - expected) > 1e-9
        elif op == '<=':
            return actual <= expected
        elif op == '>=':
            return actual >= expected
        elif op == '<':
            return actual < expected
        elif op == '>':
            return actual > expected
        return False


class SemanticErrorDetector:
    """
    Semantic error detector.

    RAHA detects syntactic errors (format anomalies, pattern violations) well but does not
    detect semantic errors, whose format and distribution look normal while the relation
    between values is wrong.

    Cells detected by RAHA are syntactic errors and true errors outside RAHA's coverage are semantic errors.

    Strategy:
    1. FD/CFD majority vote: violations of LHS -> RHS constraints
    """

    @staticmethod
    def detect_fd_violations(
        X_dirty: np.ndarray,
        fd_col_pairs: List[Tuple[List[int], int]],
        excluded_cells: Set[Tuple[int, int]],
        verbose: bool = True,
    ) -> List[Tuple]:
        """
        Majority-vote detection with FD rules.

        Returns:
            list of (idx, col, estimated_val, current_val)
        """
        violations = []
        seen_cells: Set[Tuple[int, int]] = set()

        for lhs_indices, rhs_idx in fd_col_pairs:
            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(X_dirty)):
                if (i, rhs_idx) in excluded_cells or (i, rhs_idx) in seen_cells:
                    continue
                lhs_vals = X_dirty[i, lhs_indices]
                if np.isnan(lhs_vals).any() or np.isnan(X_dirty[i, rhs_idx]):
                    continue
                key = tuple(lhs_vals.tolist())
                groups[key].append(i)

            for key, rows in groups.items():
                if len(rows) < 2:
                    continue
                rhs_values = [X_dirty[r, rhs_idx] for r in rows]
                val_counts: Dict[float, int] = defaultdict(int)
                for v in rhs_values:
                    val_counts[v] += 1
                majority_val = max(val_counts, key=val_counts.get)
                majority_count = val_counts[majority_val]

                if majority_count < len(rows):
                    for r in rows:
                        current_val = X_dirty[r, rhs_idx]
                        if (abs(current_val - majority_val) > 1e-6
                                and (r, rhs_idx) not in excluded_cells
                                and (r, rhs_idx) not in seen_cells):
                            violations.append((r, rhs_idx, majority_val, current_val))
                            seen_cells.add((r, rhs_idx))

        if verbose and violations:
            print(f"  FD rules: {len(violations)} semantic errors")

        return violations

class AutoDetector:
    """
    Automatic error detector.

    1. Missing values: np.isnan
    2. Syntactic errors: RAHA + REGEX + DOMAIN rules (out-of-range values are syntactic anomalies)
    3. Semantic errors: FD + CFD (features) + DC rules (violated relations between values)
    4. Label noise: CFD label rules
    """

    def __init__(self,
                 dirty_csv_path: Optional[str] = None,
                 clean_csv_path: Optional[str] = None,
                 dataset_name: str = "data",
                 label_col: Optional[str] = None,
                 csv_columns: Optional[List[str]] = None,
                 column_names: Optional[List[str]] = None,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 labeling_budget: int = 20,
                 enable_confident_learning: bool = False,
                 rules_path: Optional[str] = None,
                 disable_raha: bool = False):
        """
        Args:
            dirty_csv_path: path of the raw dirty CSV
            clean_csv_path: path of the raw clean CSV
            dataset_name: dataset name
            label_col: label column name
            csv_columns: every column of the raw CSV (including index and label)
            column_names: feature column names (without index and label)
            fd_rules: FD rules [("lhs_col", "rhs_col"), ...]
            labeling_budget: RAHA labelling budget (tuples)
            enable_confident_learning: unused, kept for interface compatibility
            rules_path: rule file (data/{dataset}/rules.txt)
        """
        self.dirty_csv_path = dirty_csv_path
        self.clean_csv_path = clean_csv_path
        self.dataset_name = dataset_name
        self.label_col = label_col
        self.csv_columns = csv_columns
        self.column_names = column_names
        self.fd_rules = fd_rules or []
        self.labeling_budget = labeling_budget
        # Kept for interface compatibility only
        self.enable_confident_learning = False
        self.disable_raha = disable_raha
        self.col_stats: Dict[int, Dict[str, float]] = {}
        self.is_fitted = False

        # Rule parsing
        self.rules_path = rules_path
        self.parsed_rules: ParsedRules = ParsedRules()
        if rules_path:
            self.parsed_rules = parse_rules_file(rules_path)

        # Column statistics (col_stats) are computed at run time:
        #   1. in fit(X_clean_subset) (preferred)
        #   2. in detect() from X_dirty as a fallback

        # FD column index map (FDs passed to __init__ merged with the rule file)
        self.fd_col_pairs: List[Tuple[List[int], int]] = []
        self._build_fd_index()

        # RAHA detection cache (one run per table)
        self._raha_cache: Optional[Dict[Tuple[int, int], str]] = None

        # Row indices labelled for RAHA (labeling_budget tuples, used for the pre-repair)
        self.labeled_tuples: Set[int] = set()

        # RAHA cost information (chunks and total labelling cost)
        self.raha_cost_info: Dict[str, Any] = {}

        # dirty_df cache (rule detection runs on the raw CSV)
        self._dirty_df_cache: Optional[pd.DataFrame] = None

        # Feature column name -> index
        self._feat_name_to_idx: Dict[str, int] = {}
        if self.column_names:
            self._feat_name_to_idx = {
                name: idx for idx, name in enumerate(self.column_names)
            }

    def _build_fd_index(self):
        """Column index map of the FD rules.

        Merges the fd_rules passed to __init__ with the FD rules of the rule file.
        """
        if not self.column_names:
            return

        # Merge the FD sources
        all_fd_pairs = list(self.fd_rules)  # FDs passed to __init__
        if self.parsed_rules and self.parsed_rules.fd_rules:
            for lhs, rhs in self.parsed_rules.fd_rules:
                pair = (lhs, rhs)
                if pair not in all_fd_pairs:
                    all_fd_pairs.append(pair)

        seen = set()
        for rule in all_fd_pairs:
            if isinstance(rule, (list, tuple)) and len(rule) == 2:
                lhs_str, rhs_str = rule
                lhs_cols = [c.strip() for c in str(lhs_str).split(',')]
                rhs_col = str(rhs_str).strip()

                lhs_indices = []
                for c in lhs_cols:
                    if c in self.column_names:
                        lhs_indices.append(self.column_names.index(c))

                if rhs_col in self.column_names and lhs_indices:
                    rhs_idx = self.column_names.index(rhs_col)
                    key = (tuple(lhs_indices), rhs_idx)
                    if key not in seen:
                        seen.add(key)
                        self.fd_col_pairs.append((lhs_indices, rhs_idx))

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """Compute column statistics."""
        for col in range(X.shape[1]):
            valid = X[:, col][~np.isnan(X[:, col])]
            if len(valid) > 0:
                self.col_stats[col] = {
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid) + 1e-6),
                    'q1': float(np.percentile(valid, 25)),
                    'q3': float(np.percentile(valid, 75)),
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'median': float(np.median(valid))
                }

    def fit(self, X_clean_subset: np.ndarray = None, verbose: bool = True) -> 'AutoDetector':
        """
        Initialise the detector (column statistics and the clean-data baseline).

        Args:
            X_clean_subset: clean subset for the column statistics
            verbose: print log lines
        """
        if verbose:
            print(f"\n[AutoDetector] initialisation")
            print(f"  RAHA source: {_RAHA_SOURCE}")
            print(f"  RAHA available: {RAHA_AVAILABLE}")
            print(f"  CSV path: {self.dirty_csv_path}")
            print(f"  FD rules: {len(self.fd_col_pairs)}")
            print(f"  Rule file: {self.rules_path or 'none'}")
            if self.parsed_rules.has_any_rules:
                print(f"  Rule summary: {self.parsed_rules.summary()}")

        if X_clean_subset is not None and not self.col_stats:
            self._compute_col_stats(X_clean_subset)

        self.is_fitted = True
        return self

    def _run_raha_once(self, verbose: bool = True) -> Dict[Tuple[int, int], str]:
        """Run RAHA and cache the result (labeled_tuples included)."""
        if self._raha_cache is not None:
            if verbose:
                print(f"  [RAHA] cached result: {len(self._raha_cache)} cells")
            return self._raha_cache

        if not self.dirty_csv_path or not self.clean_csv_path:
            if verbose:
                print("  [RAHA] no CSV path, skipped")
            self._raha_cache = {}
            return self._raha_cache

        if not RAHA_AVAILABLE:
            if verbose:
                print("  [RAHA] library not available, skipped")
            self._raha_cache = {}
            return self._raha_cache

        if verbose:
            print(f"\n[detection] running RAHA on the raw CSV (labeling_budget={self.labeling_budget})...")

        detected_cells, labeled_indices, chunk_info = RahaHelper.run_on_csv(
            dirty_csv_path=self.dirty_csv_path,
            clean_csv_path=self.clean_csv_path,
            dataset_name=self.dataset_name,
            labeling_budget=self.labeling_budget,
            verbose=verbose,
            fd_rules=self.fd_rules,
            primary_key=self.parsed_rules.primary_key if self.parsed_rules else None,
            label_col=self.label_col,
        )
        self._raha_cache = detected_cells
        self.labeled_tuples = labeled_indices
        self.raha_cost_info = {
            **chunk_info,
            'total_labeled_rows': len(labeled_indices),
        }
        return self._raha_cache

    def _load_dirty_df(self) -> Optional[pd.DataFrame]:
        """Load the raw CSV as a DataFrame (cached, used by rule detection)."""
        if self._dirty_df_cache is not None:
            return self._dirty_df_cache

        if not self.dirty_csv_path or not os.path.exists(self.dirty_csv_path):
            return None

        self._dirty_df_cache = pd.read_csv(
            self.dirty_csv_path, dtype=str, keep_default_na=False
        )
        return self._dirty_df_cache

    def detect(self,
               X_dirty: np.ndarray,
               y_dirty: Optional[np.ndarray] = None,
               task_type: str = 'classification',
               semantic_positions: Optional[List[Tuple[int, int]]] = None,
               verbose: bool = True) -> Dict[str, List]:
        """
        Detect errors.

        Steps:
          1. missing values (NaN)
          2. syntactic errors
             2a. RAHA (on the raw CSV)
             2b. REGEX rules (syntactic patterns RAHA misses)
             2c. DOMAIN rules (out-of-range values)
          3. semantic errors
             3a. FD rules (majority vote)
             3b. CFD feature rules (conditional dependency violations)
             3c. DC rules (cross-column integrity constraints)
          4. label noise
             4a. CFD label rules

        Returns:
            detected: {
                'missing': [(idx, col, estimated_val), ...],
                'semantic': [(idx, col, estimated_val, current_val), ...],
                'syntactic': [(idx, col, estimated_val, noise), ...],
                'label_noise': [(idx, -1, estimated_val, dirty_val), ...]
            }
        """
        detected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        n = len(X_dirty)

        # Column statistics (fit(X_clean_subset) first, X_dirty as a fallback)
        if not self.col_stats:
            self._compute_col_stats(X_dirty)

        # Raw dirty_df (needed by rule detection)
        dirty_df = self._load_dirty_df()
        has_rules = self.parsed_rules.has_any_rules
        feat_name_to_idx = self._feat_name_to_idx

        # ================================================================
        # 1. Missing values
        # ================================================================
        missing_cells: Set[Tuple[int, int]] = set()
        for i in range(n):
            for col in range(X_dirty.shape[1]):
                val = X_dirty[i, col]
                if isinstance(val, float) and np.isnan(val):
                    estimated_val = self.col_stats.get(col, {}).get('mean', 0)
                    detected['missing'].append((i, col, estimated_val))
                    missing_cells.add((i, col))

        if verbose:
            print(f"  Missing values: {len(detected['missing'])}")

        # Missing labels
        label_missing_count = 0
        if y_dirty is not None:
            for i in range(len(y_dirty)):
                if isinstance(y_dirty[i], float) and np.isnan(y_dirty[i]):
                    detected['label_noise'].append((i, -1, float('nan'), float('nan')))
                    label_missing_count += 1
            if verbose and label_missing_count > 0:
                print(f"  Missing labels (NaN): {label_missing_count}")

        # ================================================================
        # 2. Syntactic errors
        # ================================================================
        syntactic_cells: Set[Tuple[int, int]] = set()

        # 2a. RAHA (disabled with disable_raha=True)
        raha_detected = {} if self.disable_raha else self._run_raha_once(verbose=verbose)

        if raha_detected and self.csv_columns and self.column_names:
            feature_errors, raha_label_error_rows = RahaHelper.map_raha_cells_to_feature_indices(
                detected_cells=raha_detected,
                csv_columns=self.csv_columns,
                feature_columns=self.column_names,
                label_col=self.label_col or "",
            )

            for row, feat_col in feature_errors:
                if ((row, feat_col) not in missing_cells
                        and (row, feat_col) not in syntactic_cells
                        and row < n and feat_col < X_dirty.shape[1]):
                    estimated_val = self.col_stats.get(feat_col, {}).get('mean', 0)
                    current_val = X_dirty[row, feat_col] if not np.isnan(X_dirty[row, feat_col]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, feat_col, estimated_val, noise))
                    syntactic_cells.add((row, feat_col))

            # Label column errors found by RAHA are recorded too
            if raha_label_error_rows and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for row_idx in raha_label_error_rows:
                    if row_idx < len(y_dirty) and row_idx not in existing_label_rows:
                        dirty_label = y_dirty[row_idx]
                        detected['label_noise'].append((row_idx, -1, float('nan'), dirty_label))
                if verbose:
                    print(f"  RAHA label errors: {len(raha_label_error_rows)} rows")

        if verbose:
            print(f"  Syntactic errors (RAHA): {len(detected['syntactic'])}")

        # 2b. REGEX rules (complement RAHA)
        if dirty_df is not None and self.parsed_rules.regex_rules and self.column_names:
            excluded_for_regex = missing_cells | syntactic_cells
            regex_cells = RuleBasedDetector.detect_regex(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                regex_rules=self.parsed_rules.regex_rules,
                excluded_cells=excluded_for_regex,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in regex_cells:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, col_idx, estimated_val, noise))
                    syntactic_cells.add((row, col_idx))

        if verbose and self.parsed_rules.regex_rules:
            print(f"  Syntactic errors (RAHA+REGEX): {len(detected['syntactic'])}")

        # 2c. DOMAIN rules (out-of-range values, on the raw CSV)
        if dirty_df is not None and self.parsed_rules.domain_rules and self.column_names:
            domain_syntactic, domain_label = RuleBasedDetector.detect_domain(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                domain_rules=self.parsed_rules.domain_rules,
                excluded_cells=missing_cells | syntactic_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in domain_syntactic:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, col_idx, estimated_val, noise))
                    syntactic_cells.add((row, col_idx))
            # DOMAIN label errors
            if domain_label and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for (row, _) in domain_label:
                    if row < len(y_dirty) and row not in existing_label_rows:
                        dirty_label = y_dirty[row]
                        detected['label_noise'].append((row, -1, float('nan'), dirty_label))

        if verbose and self.parsed_rules.domain_rules:
            print(f"  Syntactic errors (RAHA+REGEX+DOMAIN): {len(detected['syntactic'])}")

        # ================================================================
        # 3. Semantic errors (rule driven)
        # ================================================================
        excluded_cells = missing_cells | syntactic_cells
        semantic_errors = []

        # 3a. FD rules (majority vote on the numpy array)
        if self.fd_col_pairs:
            fd_violations = SemanticErrorDetector.detect_fd_violations(
                X_dirty, self.fd_col_pairs, excluded_cells, verbose=verbose
            )
            semantic_errors.extend(fd_violations)
            for item in fd_violations:
                excluded_cells.add((item[0], item[1]))

        # 3b. CFD feature rules (on the raw CSV)
        if dirty_df is not None and self.parsed_rules.cfd_rules and self.column_names:
            cfd_semantic, cfd_label = RuleBasedDetector.detect_cfd(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                cfd_rules=self.parsed_rules.cfd_rules,
                excluded_cells=excluded_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in cfd_semantic:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else estimated_val
                    semantic_errors.append((row, col_idx, estimated_val, current_val))
                    excluded_cells.add((row, col_idx))
            # CFD label errors
            if cfd_label and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for (row, _) in cfd_label:
                    if row < len(y_dirty) and row not in existing_label_rows:
                        dirty_label = y_dirty[row]
                        detected['label_noise'].append((row, -1, float('nan'), dirty_label))

        # 3c. DC rules (on the raw CSV)
        if dirty_df is not None and self.parsed_rules.dc_rules and self.column_names:
            dc_cells, _dc_hits = RuleBasedDetector.detect_dc(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                dc_rules=self.parsed_rules.dc_rules,
                excluded_cells=excluded_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in dc_cells:
                if col_idx == -1:
                    # Label errors marked by DCs
                    if y_dirty is not None:
                        existing_label_rows = {item[0] for item in detected['label_noise']}
                        if row < len(y_dirty) and row not in existing_label_rows:
                            dirty_label = y_dirty[row]
                            detected['label_noise'].append((row, -1, float('nan'), dirty_label))
                elif row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else estimated_val
                    semantic_errors.append((row, col_idx, estimated_val, current_val))
                    excluded_cells.add((row, col_idx))

        # 3e. Externally supplied semantic error positions (backward compatibility)
        if semantic_positions:
            for pos in semantic_positions:
                if len(pos) >= 2:
                    idx, col = pos[0], pos[1]
                    if (idx, col) not in excluded_cells:
                        estimated_val = self.col_stats.get(col, {}).get('mean', 0)
                        current_val = X_dirty[idx, col] if not np.isnan(X_dirty[idx, col]) else estimated_val
                        semantic_errors.append((idx, col, estimated_val, current_val))
                        excluded_cells.add((idx, col))

        detected['semantic'] = semantic_errors

        if verbose:
            sources = []
            if self.fd_col_pairs:
                sources.append("FD")
            if self.parsed_rules.cfd_rules:
                sources.append("CFD")
            if self.parsed_rules.dc_rules:
                sources.append("DC")
            if semantic_positions:
                sources.append("external")
            mode = "+".join(sources) if sources else "no source"
            print(f"  Semantic errors: {len(detected['semantic'])} ({mode})")

        # ================================================================
        # 4. Label noise: totals
        # ================================================================
        # CFD/DC label errors were added to label_noise in steps 3b and 3c
        if verbose:
            label_from_nan = label_missing_count
            label_total = len(detected['label_noise'])
            label_from_rules = label_total - label_from_nan
            print(f"  Label noise: {label_total} (NaN={label_from_nan}, rules={label_from_rules})")

        # ================================================================
        # 4b. Drop label flags of rows with damaged features
        # ================================================================
        # CFD/DC label rules depend on feature conditions. If a feature of the row is detected
        # as a semantic, syntactic or missing error, the label decision based on it is unreliable.
        # Dropping these rows removes many label false positives.
        feature_damaged_removed = 0
        if detected['label_noise']:
            damaged_rows: Set[int] = set()
            for item in detected['semantic']:
                damaged_rows.add(item[0])
            for item in detected['syntactic']:
                damaged_rows.add(item[0])
            for item in detected['missing']:
                damaged_rows.add(item[0])

            if damaged_rows:
                filtered_label = []
                for item in detected['label_noise']:
                    row_idx = item[0]
                    if row_idx in damaged_rows:
                        feature_damaged_removed += 1
                        continue
                    filtered_label.append(item)
                detected['label_noise'] = filtered_label
                if verbose and feature_damaged_removed > 0:
                    print(f"  Label flags dropped for rows with damaged features: {feature_damaged_removed}")

        # ================================================================
        # Summary
        # ================================================================
        if verbose:
            total = (len(detected['missing']) + len(detected['syntactic'])
                     + len(detected['semantic']) + len(detected['label_noise']))
            print(f"\n  Detection summary: {total} errors "
                  f"(missing={len(detected['missing'])}, "
                  f"syntactic={len(detected['syntactic'])}, "
                  f"semantic={len(detected['semantic'])}, "
                  f"label noise={len(detected['label_noise'])})")

        return detected

    def build_error_list(self,
                         detected: Dict[str, List],
                         X_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Convert the detected errors to the format of the cleaning environment.

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
        """
        error_list = []

        # Missing errors (type=0)
        for item in detected['missing']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 0, 'repair_value': repair_value
            })

        # Semantic errors (type=1)
        for item in detected['semantic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 1, 'repair_value': repair_value
            })

        # Syntactic errors (type=2)
        for item in detected['syntactic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 2, 'repair_value': repair_value
            })

        # Label noise errors (type=3, col=-1)
        for item in detected.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                clean_val = item[2] if len(item) > 2 else float('nan')
                repair_value = clean_val if not np.isnan(clean_val) else float('nan')
                error_list.append({
                    'idx': idx, 'col': -1, 'type': 3, 'repair_value': repair_value
                })

        return error_list

    def save(self, path: str) -> None:
        """Save the detector parameters."""
        data = {
            'dirty_csv_path': self.dirty_csv_path,
            'clean_csv_path': self.clean_csv_path,
            'dataset_name': self.dataset_name,
            'label_col': self.label_col,
            'csv_columns': self.csv_columns,
            'column_names': self.column_names,
            'fd_rules': self.fd_rules,
            'labeling_budget': self.labeling_budget,
            'rules_path': self.rules_path,
            'col_stats': self.col_stats,
            'is_fitted': self.is_fitted,
            'labeled_tuples': list(self.labeled_tuples),
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Detector saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'AutoDetector':
        """Load the detector parameters."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(
            dirty_csv_path=data.get('dirty_csv_path'),
            clean_csv_path=data.get('clean_csv_path'),
            dataset_name=data.get('dataset_name', 'data'),
            label_col=data.get('label_col'),
            csv_columns=data.get('csv_columns'),
            column_names=data.get('column_names'),
            fd_rules=data.get('fd_rules'),
            labeling_budget=data.get('labeling_budget', 20),
            rules_path=data.get('rules_path'),
        )
        detector.col_stats = data.get('col_stats', {})
        detector.is_fitted = data.get('is_fitted', True)
        detector.labeled_tuples = set(data.get('labeled_tuples', []))
        print(f"  Detector loaded: fd_rules={len(detector.fd_col_pairs)}, "
              f"rules={detector.parsed_rules.summary()}, "
              f"labeled_tuples={len(detector.labeled_tuples)}")
        return detector


# Backward-compatible alias
RahaBasedDetector = AutoDetector


def is_raha_available() -> bool:
    """Return whether RAHA is available."""
    return RAHA_AVAILABLE
