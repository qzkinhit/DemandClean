#!/usr/bin/env python
"""Portable entry point for the two-player attribution of a canonical DemandClean run.

    python results/attribution_operation/attr_op.py results/main_table/beers/demandclean

The implementation is shared with evaluation/attribution_operation.py. The historical
--release flag is accepted; canonical dataset settings determine category encoding.
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation'))


if __name__ == '__main__':
    from attribution_operation import main
    while '--release' in sys.argv:
        sys.argv.remove('--release')
    main()
