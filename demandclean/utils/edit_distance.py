"""
Edit Distance Utilities
=======================

Provides string similarity, nearest-value lookup, and typo generation based on
edit distance (SequenceMatcher). Has zero external dependencies - only the
standard library modules difflib + random + string are used.

Three core use cases:
  1. ErrorInjector: generate_typo() produces realistic typos for categorical columns.
  2. encode_df(): find_nearest_known() maps dirty values to known categories (as a NaN replacement).
  3. ValueEstimator: find_nearest_known() performs edit-distance-based estimation (to fix obvious typos).
"""

import random
import string
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


def edit_distance_ratio(a: str, b: str) -> float:
    """Compute the similarity ratio between two strings.

    Based on SequenceMatcher.ratio(); the returned value lies in [0, 1].
    1.0 = identical, 0.0 = completely different.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity in [0, 1]
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_nearest_known(
    value: str,
    known_values: List[str],
    threshold: float = 0.6,
) -> Optional[str]:
    """Find the value with the smallest edit distance in a list of known values.

    Iterates over known_values, computes similarity with value, and returns the
    candidate with the highest similarity that is >= threshold.

    Args:
        value: Target string to match
        known_values: List of known valid values
        threshold: Minimum similarity threshold; returns None if not met

    Returns:
        The nearest known value, or None if no match meets the threshold
    """
    if not value or not known_values:
        return None

    best_match: Optional[str] = None
    best_ratio: float = -1.0

    for known in known_values:
        ratio = SequenceMatcher(None, value, known).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = known

    if best_ratio >= threshold:
        return best_match
    return None


def find_top_k_nearest(
    value: str,
    known_values: List[str],
    k: int = 3,
    threshold: float = 0.3,
) -> List[Tuple[str, float]]:
    """Find the top-k nearest values by edit distance from a list of known values.

    Args:
        value: Target string to match
        known_values: List of known valid values
        k: Maximum number of results to return
        threshold: Minimum similarity threshold

    Returns:
        [(known_value, ratio), ...] sorted by similarity in descending order
    """
    if not value or not known_values:
        return []

    scored = []
    for known in known_values:
        ratio = SequenceMatcher(None, value, known).ratio()
        if ratio >= threshold:
            scored.append((known, ratio))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def generate_typo(value: str) -> str:
    """Apply a single random typo to a string.

    Four equiprobable strategies:
      - char_swap:   swap adjacent characters   "Colorado" -> "Clorado"
      - char_delete: delete a random character  "Colorado" -> "Colordo"
      - char_insert: insert a random character  "Colorado" -> "Coloradoo"
      - case_change: change letter case         "Colorado" -> "cOlorado"

    For strings of length <= 1, only char_insert is used.

    Args:
        value: Original string

    Returns:
        The string after applying a typo (guaranteed to differ from the original)
    """
    if not value:
        return value

    chars = list(value)

    if len(chars) <= 1:
        # Short strings can only be extended via insertion
        strategies = ['char_insert', 'case_change']
    else:
        strategies = ['char_swap', 'char_delete', 'char_insert', 'case_change']

    strategy = random.choice(strategies)

    if strategy == 'char_swap' and len(chars) >= 2:
        # Swap adjacent characters
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        # If the swap produces an identical string (e.g. swapping in "aa"), try another position
        result = ''.join(chars)
        if result == value and len(chars) >= 3:
            pos = (pos + 1) % (len(chars) - 1)
            chars = list(value)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    elif strategy == 'char_delete' and len(chars) >= 2:
        # Delete a random character (avoid reducing to an empty string)
        pos = random.randint(0, len(chars) - 1)
        chars.pop(pos)

    elif strategy == 'char_insert':
        # Insert a random letter at a random position
        pos = random.randint(0, len(chars))
        insert_char = random.choice(string.ascii_lowercase)
        chars.insert(pos, insert_char)

    elif strategy == 'case_change':
        # Flip the case of 1-2 random letters
        alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
        if alpha_positions:
            n_changes = min(random.choice([1, 2]), len(alpha_positions))
            positions = random.sample(alpha_positions, n_changes)
            for pos in positions:
                chars[pos] = chars[pos].swapcase()

    result = ''.join(chars)

    # Guarantee the output differs from the original value
    if result == value:
        # Fallback: append a random character at the end
        result = value + random.choice(string.ascii_lowercase)

    return result
