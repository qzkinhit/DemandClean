"""
Tools
=====

CSV format normalisation and column statistics used by the automatic detector.
"""

from .csv_normalizer import (
    normalize_dirty_format,
    normalize_dirty_to_file,
)

__all__ = [
    'normalize_dirty_format',
    'normalize_dirty_to_file',
]
