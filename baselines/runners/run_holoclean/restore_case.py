"""Replay the recorded letter-case restoration of the beers HoloClean output.

    python baselines/runners/run_holoclean/restore_case.py --dirty data/beers/dirty_index.csv --table results/main_table/beers/baselines/holoclean/raw_cleaned.csv.gz --check results/main_table/beers/baselines/holoclean/cleaned.csv.gz

The lookup uses the entire supplied dirty table, including the style column. It
never reads clean data. For each common non-index column, keep the original dirty
value of the same cell when it equals the output ignoring case; otherwise use the
most frequent dirty spelling with the same lowercase form. Ties follow first
appearance in the dirty column. If neither lookup matches, keep the output.
The exact HoloClean marker _nan_ becomes an empty string. No stripping, fuzzy
matching, title-casing or training-split filtering is performed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
from pathlib import Path
import sys


def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def read_table(path):
    rows = [row for row in csv.reader(io.StringIO(read_bytes(path).decode('utf-8-sig'), newline='')) if row]
    if not rows or 'index' not in rows[0]:
        raise ValueError('CSV must have an index column')
    header = rows[0]
    if len(header) != len(set(header)) or any(len(row) != len(header) for row in rows[1:]):
        raise ValueError('CSV headers or row widths are inconsistent')
    return header, rows[1:]


def restore_case(dirty_header, dirty_rows, header, rows):
    """Return restored rows and the same operation counts as the recorded script."""
    dirty_index = dirty_header.index('index')
    index = header.index('index')
    dirty_by_id = {row[dirty_index]: row for row in dirty_rows}
    if len(dirty_by_id) != len(dirty_rows):
        raise ValueError('dirty row indices must be unique')
    restored = [row[:] for row in rows]
    stats = Counter()
    for column, name in enumerate(header):
        if name == 'index' or name not in dirty_header:
            continue
        source_column = dirty_header.index(name)
        domain = defaultdict(Counter)
        for row in dirty_rows:
            value = row[source_column]
            domain[value.lower()][value] += 1
        for row in restored:
            value = row[column]
            if value == '_nan_':
                row[column] = ''
                stats['null'] += 1
                continue
            dirty_row = dirty_by_id.get(row[index])
            dirty_value = dirty_row[source_column] if dirty_row is not None else None
            if value == dirty_value:
                continue
            if dirty_value is not None and value.lower() == dirty_value.lower():
                row[column] = dirty_value
                stats['same_cell'] += 1
            elif value.lower() in domain:
                row[column] = domain[value.lower()].most_common(1)[0][0]
                stats['domain'] += 1
            else:
                stats['kept'] += 1
    return restored, dict(stats)


def csv_bytes(header, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dirty', type=Path, required=True, help='the full original dirty indexed CSV')
    ap.add_argument('--table', type=Path, required=True, help='raw HoloClean CSV before null/case normalization; gzip accepted')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path, help='write the restored CSV; .gz also supported')
    group.add_argument('--check', type=Path, help='compare exact CSV bytes with this saved output; write nothing')
    args = ap.parse_args(argv)
    dirty_header, dirty_rows = read_table(args.dirty)
    header, rows = read_table(args.table)
    restored, stats = restore_case(dirty_header, dirty_rows, header, rows)
    output = csv_bytes(header, restored)
    digest = hashlib.sha256(output).hexdigest()
    if args.check:
        if output != read_bytes(args.check):
            raise ValueError('restored CSV differs from the saved output (replay SHA-256 ' + digest + ')')
        print('HoloClean case restoration: exact CSV match; %d rows; %s' % (len(rows), stats))
    else:
        if args.out.resolve() in {args.dirty.resolve(), args.table.resolve()}:
            ap.error('choose an output path that does not overwrite either input')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(gzip.compress(output, mtime=0) if args.out.suffix == '.gz' else output)
        print('HoloClean case restoration: wrote %s; %s' % (args.out, stats))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        sys.exit(1)
