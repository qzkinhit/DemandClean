"""Resource-only routing. Every request keeps the upstream model and options."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import json
import os
import threading
import time

ROOT = Path(__file__).resolve().parents[3]  # repository root
VERSION = 'resource_v2_four_identical_endpoints'
CONTEXT = threading.local()


class EndpointPool:
    def __init__(self):
        from ollama import Client
        self.endpoints = [x.strip().rstrip('/') for x in os.environ.get(
            'RETCLEAN_ENDPOINTS', os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')).split(',') if x.strip()]
        self.clients = {url: Client(host=url, timeout=600) for url in self.endpoints}
        self.capacity = 6
        self.root = Path(os.environ.get('RETCLEAN_SLOT_DIR', str(ROOT / 'results/runs/retclean_request_slots')))
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = threading.Lock()
        self.cursor = os.getpid() % len(self.endpoints)

    @contextmanager
    def claim(self):
        while True:
            with self.guard:
                start = self.cursor
                self.cursor = (self.cursor + 1) % len(self.endpoints)
            for offset in range(len(self.endpoints)):
                url = self.endpoints[(start + offset) % len(self.endpoints)]
                port = url.rsplit(':', 1)[-1]
                for slot in range(self.capacity):
                    handle = (self.root / f'{port}.{slot}.lock').open('a+')
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    try:
                        yield url, self.clients[url], slot
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                        handle.close()
                    return
            time.sleep(0.025)


def acquire_dataset_writer(output_dir):
    handle = (Path(output_dir) / '.retclean_writer.lock').open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another RetClean writer already owns this dataset output')
    return handle


def load_fixed_reference(bundle, output_dir, expected_rows):
    """Resume uses the frozen paid reference, never performs another oracle query."""
    import hashlib
    import pandas as pd
    output_dir = Path(output_dir)
    reference_path = output_dir / 'reference_lake.csv'
    ledger_path = output_dir / 'gt_access.json'
    if not reference_path.exists() or not ledger_path.exists():
        return None
    reference = pd.read_csv(reference_path)
    ledger = json.loads(ledger_path.read_text())
    expected = [(int(row), str(column)) for row in expected_rows for column in bundle['non_index_columns']]
    if ledger != [list(x) for x in expected]:
        raise RuntimeError('Frozen GT access ledger differs from the original fixed half-training reference')
    if reference['index'].astype(int).tolist() != expected_rows:
        raise RuntimeError('Frozen reference row order changed')
    if len(ledger) > bundle['gt_cap_cells']:
        raise RuntimeError('Frozen GT pool exceeds cap')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference_path, ledger_path)}
    return reference, len(ledger), hashes
