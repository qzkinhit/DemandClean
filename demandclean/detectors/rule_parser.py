"""
Rule parser (for DemandClean)
=============================

Parses every section of data/{dataset}/rules.txt and produces structured rule
data used for semantic error injection and detection.

Supported sections:
  [REGEX]       - regex-based syntactic detection rules (stored only, not used for injection)
  [DOMAIN]      - value-domain constraints (semantic injection: out-of-domain values)
  [FD]          - functional dependencies (semantic injection: FD violations)
  [HORIZON_FD]  - equivalent to [FD]
  [CFD]         - conditional functional dependencies (semantic injection: condition violations)
  [DC]          - cross-column integrity constraints (semantic-injection reference)
  [STATISTICAL] - statistical threshold parameters (stored only, used for detection)

Design notes:
  - All rule-based injections produce semantic errors (rule violations).
  - Syntactic errors are injected by the RAHA-aware statistical method and do
    not use rules.
  - Datasets without rich rules automatically fall back to FD-based or random injection.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class DomainRule:
    """Value-domain constraint rule.

    Examples:
        INT [1, 10]   -> dtype='INT', min_val=1, max_val=10, enum_vals=None
        ENUM {2, 4}   -> dtype='ENUM', min_val=None, max_val=None, enum_vals={2, 4}
    """
    column: str
    dtype: str          # 'INT', 'FLOAT', 'ENUM'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    enum_vals: Optional[Set[str]] = None


@dataclass
class RegexRule:
    """Regex detection rule (stored only; not used for injection)."""
    column: str         # column name or 'ALL_FEATURES'
    pattern: str        # regex pattern


@dataclass
class CFDRule:
    """Conditional functional dependency rule.

    Example:
        class=2, n_anomaly<=2 => Clump Thickness EXCESS >= 5 FROM_BASELINE 5
        -> conditions = [('class', '=', '2'), ('n_anomaly', '<=', '2')]
           target_col = 'Clump Thickness'
           direction = 'EXCESS'   # or 'DEFICIT'
           threshold = 5
           baseline = 5
    """
    conditions: List[Tuple[str, str, str]]  # [(col, op, val), ...]
    target_col: str
    direction: str      # 'EXCESS' or 'DEFICIT'
    threshold: float
    baseline: float


@dataclass
class DCRule:
    """Structured DC (Denial Constraint) rule.

    DCs use denial semantics: the constraint is violated when every clause holds.
    The MARK clause specifies the column to flag when a violation occurs.

    Examples:
        t1&EQ(t1.holiday, 1)&NEQ(t1.workingday, 0)&MARK(t1.workingday)
        -> clauses = [{'type':'simple','op':'EQ','col':'holiday','value':1.0},
                      {'type':'simple','op':'NEQ','col':'workingday','value':0.0}]
           mark_cols = ['workingday']
           involved_cols = ['holiday', 'workingday']

        t1&GT(ABS(t1.454 - t1.458), 0.03)
        -> clauses = [{'type':'abs_diff','op':'GT','col1':'454','col2':'458','value':0.03}]
           mark_cols = []
           involved_cols = ['454', '458']
    """
    raw: str                                    # original string
    clauses: List[Dict[str, Any]]               # parsed condition clauses
    mark_cols: List[str]                        # target columns flagged by MARK (may be empty)
    involved_cols: List[str]                    # all columns referenced (excluding MARK columns)


@dataclass
class ParsedRules:
    """Full parsed rule collection."""
    # Semantic-injection rules
    domain_rules: List[DomainRule] = field(default_factory=list)
    fd_rules: List[Tuple[str, str]] = field(default_factory=list)       # [(lhs, rhs)]
    cfd_rules: List[CFDRule] = field(default_factory=list)
    dc_rules: List[DCRule] = field(default_factory=list)                # structured DC rules

    # Optional
    primary_key: Optional[List[str]] = None     # primary-key columns (used for block clustering)

    # Stored only, used for detection
    regex_rules: List[RegexRule] = field(default_factory=list)
    statistical: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    raw_sections: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def has_rich_rules(self) -> bool:
        """Whether rich rules exist (DOMAIN / CFD / DC).

        When rich rules are available, rule-based semantic injection is
        preferred; otherwise fall back to FD-based or random injection.
        """
        return bool(self.domain_rules or self.cfd_rules or self.dc_rules)

    @property
    def has_any_rules(self) -> bool:
        """Whether any rule at all is defined."""
        return bool(
            self.domain_rules or self.fd_rules or self.cfd_rules
            or self.dc_rules or self.regex_rules
        )

    def summary(self) -> str:
        """Rule summary."""
        parts = []
        if self.regex_rules:
            parts.append(f"REGEX={len(self.regex_rules)}")
        if self.domain_rules:
            parts.append(f"DOMAIN={len(self.domain_rules)}")
        if self.fd_rules:
            parts.append(f"FD={len(self.fd_rules)}")
        if self.cfd_rules:
            parts.append(f"CFD={len(self.cfd_rules)}")
        if self.dc_rules:
            parts.append(f"DC={len(self.dc_rules)}")
        if self.statistical:
            parts.append(f"STAT={len(self.statistical)}")
        return f"ParsedRules({', '.join(parts) or 'empty'})"


# ============================================================================
# Parsing functions
# ============================================================================

def parse_rules_file(rules_path: str) -> ParsedRules:
    """Parse every section of rules.txt.

    Args:
        rules_path: path to the rule file

    Returns:
        A structured ParsedRules object.
    """
    if not rules_path or not os.path.exists(rules_path):
        return ParsedRules()

    # Step 1: group raw lines by section
    raw_sections: Dict[str, List[str]] = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].upper()
                raw_sections.setdefault(current_section, [])
            elif current_section:
                raw_sections[current_section].append(line)

    # Step 2: parse section by section
    result = ParsedRules(raw_sections=raw_sections)

    # REGEX
    for line in raw_sections.get('REGEX', []):
        rule = _parse_regex_line(line)
        if rule:
            result.regex_rules.append(rule)

    # DOMAIN
    for line in raw_sections.get('DOMAIN', []):
        rule = _parse_domain_line(line)
        if rule:
            result.domain_rules.append(rule)

    # FD (merge [FD] and [HORIZON_FD])
    for section_name in ('FD', 'HORIZON_FD'):
        for line in raw_sections.get(section_name, []):
            pair = _parse_fd_line(line)
            if pair:
                result.fd_rules.append(pair)

    # CFD
    for line in raw_sections.get('CFD', []):
        rule = _parse_cfd_line(line)
        if rule:
            result.cfd_rules.append(rule)

    # DC (structured parsing)
    for line in raw_sections.get('DC', []):
        rule = _parse_dc_line(line)
        if rule:
            result.dc_rules.append(rule)

    # PRIMARY_KEY (optional, used for block clustering)
    pk_lines = raw_sections.get('PRIMARY_KEY', [])
    if pk_lines:
        pk_cols = []
        for line in pk_lines:
            for col in line.split(','):
                col = col.strip()
                if col:
                    pk_cols.append(col)
        if pk_cols:
            result.primary_key = pk_cols

    # STATISTICAL
    for line in raw_sections.get('STATISTICAL', []):
        key, val = _parse_statistical_line(line)
        if key == 'COL_STATS' and isinstance(val, tuple):
            # Per-column stats: val = (col_name, {mean, std, ...})
            col_stats_dict = result.statistical.setdefault('col_stats', {})
            col_name, stats = val
            col_stats_dict[col_name] = stats
        elif key:
            result.statistical[key] = val

    return result


# ============================================================================
# Section-level parsers
# ============================================================================

def _parse_regex_line(line: str) -> Optional[RegexRule]:
    """Parse a REGEX rule line.

    Format: COLUMN: pattern
    Example: ALL_FEATURES: ^(\\d)\\1$
    """
    if ':' not in line:
        return None
    col, pattern = line.split(':', 1)
    col = col.strip()
    pattern = pattern.strip()
    if col and pattern:
        return RegexRule(column=col, pattern=pattern)
    return None


def _parse_domain_line(line: str) -> Optional[DomainRule]:
    """Parse a DOMAIN rule line.

    Formats:
        Column Name: INT [min, max]
        Column Name: FLOAT [min, max]
        Column Name: ENUM {val1, val2, ...}
    """
    if ':' not in line:
        return None

    col, spec = line.split(':', 1)
    col = col.strip()
    spec = spec.strip()

    if not col or not spec:
        return None

    # INT / FLOAT range
    range_match = re.match(r'(INT|FLOAT)\s*\[([^,]+),\s*([^\]]+)\]', spec)
    if range_match:
        dtype = range_match.group(1)
        min_val = float(range_match.group(2).strip())
        max_val = float(range_match.group(3).strip())
        return DomainRule(column=col, dtype=dtype, min_val=min_val, max_val=max_val)

    # ENUM set
    enum_match = re.match(r'ENUM\s*\{([^}]+)\}', spec)
    if enum_match:
        vals = {v.strip() for v in enum_match.group(1).split(',')}
        return DomainRule(column=col, dtype='ENUM', enum_vals=vals)

    return None


def _parse_fd_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse an FD rule line.

    Format: LHS => RHS  or  LHS \u21d2 RHS
    """
    for sep in ('=>', '⇒'):
        if sep in line:
            parts = line.split(sep)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                if lhs and rhs:
                    return (lhs, rhs)
    return None


def _parse_cfd_line(line: str) -> Optional[CFDRule]:
    """Parse a CFD rule line.

    Format:
        class=2, n_anomaly<=2 => Clump Thickness EXCESS >= 5 FROM_BASELINE 5
        class=4, n_anomaly<=1 => Clump Thickness DEFICIT >= 3 FROM_BASELINE 4
    """
    if '=>' not in line:
        return None

    lhs, rhs = line.split('=>', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    # Parse conditions (comma-separated "col op val")
    conditions = []
    for cond_str in lhs.split(','):
        cond_str = cond_str.strip()
        # Match "col op val"; supports =, <=, >=, <, >, !=
        m = re.match(r'(\w+)\s*(<=|>=|!=|=|<|>)\s*(.+)', cond_str)
        if m:
            conditions.append((m.group(1).strip(), m.group(2), m.group(3).strip()))

    if not conditions:
        return None

    # Parse RHS:  ColName EXCESS/DEFICIT >= threshold FROM_BASELINE baseline
    rhs_match = re.match(
        r'(.+?)\s+(EXCESS|DEFICIT)\s*>=\s*(\d+(?:\.\d+)?)\s+FROM_BASELINE\s+(\d+(?:\.\d+)?)',
        rhs
    )
    if not rhs_match:
        return None

    target_col = rhs_match.group(1).strip()
    direction = rhs_match.group(2)
    threshold = float(rhs_match.group(3))
    baseline = float(rhs_match.group(4))

    return CFDRule(
        conditions=conditions,
        target_col=target_col,
        direction=direction,
        threshold=threshold,
        baseline=baseline,
    )


# ---- DC clause parsing (shared; auto_detector uses it too) ----

def parse_dc_clause(clause_str: str) -> Optional[Dict[str, Any]]:
    """Parse a single DC clause.

    Supported formats:
        EQ(t1.col, val)           -> {'type':'simple', 'op':'EQ', 'col':'col', 'value':val}
        GTE(t1.col, val)          -> {'type':'simple', 'op':'GTE', 'col':'col', 'value':val}
        GT(ABS(t1.c1-t1.c2), val) -> {'type':'abs_diff', 'op':'GT', 'col1','col2','value'}
        MARK(t1.col)              -> {'type':'mark', 'col':'col'}
    """
    # MARK format: MARK(t1.col)
    mark_match = re.match(r'MARK\(t1\.(.+?)\)', clause_str)
    if mark_match:
        col = mark_match.group(1).strip()
        return {
            'type': 'mark',
            'col': col,
            'columns': [col],
        }

    # ABS-diff format: GT(ABS(t1.col1 - t1.col2), val)
    abs_match = re.match(
        r'(GT|GTE|LT|LTE|EQ|NEQ)\(ABS\(t1\.(.+?)\s*-\s*t1\.(.+?)\)\s*,\s*(.+?)\)',
        clause_str
    )
    if abs_match:
        op = abs_match.group(1)
        col1 = abs_match.group(2).strip()
        col2 = abs_match.group(3).strip()
        val = abs_match.group(4).strip()
        try:
            val = float(val)
        except ValueError:
            return None
        return {
            'type': 'abs_diff',
            'op': op,
            'col1': col1,
            'col2': col2,
            'value': val,
            'columns': [col1, col2],
        }

    # Simple format: OP(t1.col, val)
    simple_match = re.match(
        r'(GT|GTE|LT|LTE|EQ|NEQ|IQ)\(t1\.(.+?)\s*,\s*(.+?)\)',
        clause_str
    )
    if simple_match:
        op = simple_match.group(1)
        col = simple_match.group(2).strip()
        val = simple_match.group(3).strip()
        try:
            val_num = float(val)
            return {
                'type': 'simple',
                'op': op,
                'col': col,
                'value': val_num,
                'columns': [col],
            }
        except ValueError:
            return {
                'type': 'simple_str',
                'op': op,
                'col': col,
                'value': val,
                'columns': [col],
            }

    return None


def _parse_dc_line(line: str) -> Optional[DCRule]:
    """Parse a DC rule line.

    Format: t1&CLAUSE1&CLAUSE2&...&MARK(t1.col)
    Examples:
        t1&EQ(t1.holiday, 1)&NEQ(t1.workingday, 0)&MARK(t1.workingday)
        t1&GT(ABS(t1.454 - t1.458), 0.03)
    """
    line = line.strip()
    if not line or not line.startswith('t1'):
        return None

    parts = line.split('&')
    if len(parts) < 2:
        return None

    clauses = []
    mark_cols = []
    involved_cols = []

    for part in parts[1:]:  # skip "t1"
        clause = parse_dc_clause(part.strip())
        if clause:
            if clause['type'] == 'mark':
                mark_cols.append(clause['col'])
            else:
                clauses.append(clause)
                involved_cols.extend(clause.get('columns', []))

    if not clauses:
        return None

    # Deduplicate while preserving order
    seen = set()
    unique_cols = []
    for c in involved_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return DCRule(
        raw=line,
        clauses=clauses,
        mark_cols=mark_cols,
        involved_cols=unique_cols,
    )


def _parse_statistical_line(line: str) -> Tuple[Optional[str], Any]:
    """Parse a STATISTICAL configuration line.

    Format 1 (global parameter): KEY: value
    Examples:
        IQR_MULTIPLIER: 2.5
        ZSCORE_THRESHOLD: 3.0

    Format 2 (per-column stats): COL_STATS: col_name | mean=0.123 | std=0.456 | ...
    Example:
        COL_STATS: abv | mean=0.059 | std=0.014 | q1=0.050 | q3=0.068 | min=0.001 | max=0.128 | median=0.056
    """
    if ':' not in line:
        return None, None

    key, val = line.split(':', 1)
    key = key.strip()
    val = val.strip()

    if not key:
        return None, None

    # Per-column stats
    if key == 'COL_STATS' and '|' in val:
        parts = [p.strip() for p in val.split('|')]
        col_name = parts[0]
        stats = {}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                try:
                    stats[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        if col_name and stats:
            return 'COL_STATS', (col_name, stats)
        return None, None

    # Try numeric
    try:
        return key, float(val)
    except ValueError:
        pass

    # Comma-separated column-name list
    if ',' in val:
        return key, [v.strip() for v in val.split(',')]

    return key, val


# ============================================================================
# Convenience helpers
# ============================================================================

def load_rules(rules_path: Optional[str]) -> ParsedRules:
    """Load and parse a rule file (null-safe).

    Args:
        rules_path: path to the rule file; None returns an empty rule set

    Returns:
        A ParsedRules object.
    """
    if not rules_path:
        return ParsedRules()
    return parse_rules_file(rules_path)


def extract_fd_pairs(parsed: ParsedRules) -> List[Tuple[str, str]]:
    """Extract FD pairs (compatible with both HORIZON_FD and FD sections).

    Returns a deduplicated list of (lhs, rhs) tuples.
    """
    seen = set()
    result = []
    for lhs, rhs in parsed.fd_rules:
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def get_domain_range(parsed: ParsedRules, column: str) -> Optional[Tuple[float, float]]:
    """Return the domain range for a column.

    Returns:
        (min_val, max_val) or None
    """
    for rule in parsed.domain_rules:
        if rule.column == column and rule.min_val is not None:
            return (rule.min_val, rule.max_val)
    return None


def get_domain_enum(parsed: ParsedRules, column: str) -> Optional[Set[str]]:
    """Return the set of valid enum values for a column.

    Returns:
        Set of valid values, or None.
    """
    for rule in parsed.domain_rules:
        if rule.column == column and rule.enum_vals is not None:
            return rule.enum_vals
    return None


def get_cfd_rules_for_class(parsed: ParsedRules, class_val: str) -> List[CFDRule]:
    """Return all CFD rules conditioned on a given class value."""
    result = []
    for rule in parsed.cfd_rules:
        for col, op, val in rule.conditions:
            if col == 'class' and op == '=' and val == str(class_val):
                result.append(rule)
                break
    return result


def get_col_stats_from_rules(
    parsed: ParsedRules,
    column_names: List[str],
) -> Dict[int, Dict[str, float]]:
    """Extract per-column-index statistics from parsed_rules.statistical['col_stats'].

    Args:
        parsed: parsed rules
        column_names: list of feature column names

    Returns:
        {col_idx: {'mean': ..., 'std': ..., 'q1': ..., 'q3': ..., 'min': ..., 'max': ..., 'median': ...}}
    """
    col_stats_by_name = parsed.statistical.get('col_stats', {})
    if not col_stats_by_name or not column_names:
        return {}

    name_to_idx = {name: idx for idx, name in enumerate(column_names)}
    result = {}
    for col_name, stats in col_stats_by_name.items():
        if col_name in name_to_idx:
            result[name_to_idx[col_name]] = stats
    return result


def rules_to_dict(parsed: ParsedRules) -> Dict[str, Any]:
    """Convert ParsedRules to a serializable dict (for passing through config)."""
    return {
        'has_rich_rules': parsed.has_rich_rules,
        'has_any_rules': parsed.has_any_rules,
        'domain_rules': [
            {
                'column': r.column,
                'dtype': r.dtype,
                'min_val': r.min_val,
                'max_val': r.max_val,
                'enum_vals': list(r.enum_vals) if r.enum_vals else None,
            }
            for r in parsed.domain_rules
        ],
        'fd_rules': parsed.fd_rules,
        'cfd_rules': [
            {
                'conditions': r.conditions,
                'target_col': r.target_col,
                'direction': r.direction,
                'threshold': r.threshold,
                'baseline': r.baseline,
            }
            for r in parsed.cfd_rules
        ],
        'dc_rules': [
            {
                'raw': r.raw,
                'clauses': r.clauses,
                'mark_cols': r.mark_cols,
                'involved_cols': r.involved_cols,
            }
            for r in parsed.dc_rules
        ],
        'primary_key': parsed.primary_key,
        'statistical': parsed.statistical,
    }


# ============================================================================
# Test entry point
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'breast_cancer', 'rules.txt'
        )

    parsed = parse_rules_file(path)
    print(parsed.summary())

    if parsed.domain_rules:
        print(f"\nDOMAIN ({len(parsed.domain_rules)}):")
        for r in parsed.domain_rules:
            print(f"  {r}")

    if parsed.fd_rules:
        print(f"\nFD ({len(parsed.fd_rules)}):")
        for lhs, rhs in parsed.fd_rules:
            print(f"  {lhs} => {rhs}")

    if parsed.cfd_rules:
        print(f"\nCFD ({len(parsed.cfd_rules)}):")
        for r in parsed.cfd_rules:
            print(f"  {r.conditions} => {r.target_col} {r.direction} >= {r.threshold}")

    if parsed.dc_rules:
        print(f"\nDC ({len(parsed.dc_rules)}):")
        for r in parsed.dc_rules:
            mark_str = f" MARK={r.mark_cols}" if r.mark_cols else ""
            print(f"  {r.raw}  ->  clauses={len(r.clauses)}, cols={r.involved_cols}{mark_str}")

    if parsed.statistical:
        print(f"\nSTATISTICAL: {parsed.statistical}")
