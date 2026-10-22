"""Detector module"""

from .error_injector import ErrorInjector, LabelErrorPattern, analyze_label_error_pattern
from .auto_detector import AutoDetector, RahaBasedDetector, RuleBasedDetector
from .oracle_detector import OracleDetector
from .rule_parser import (
    ParsedRules, DomainRule, RegexRule, CFDRule,
    parse_rules_file, load_rules, extract_fd_pairs,
    get_domain_range, get_domain_enum, get_cfd_rules_for_class,
    rules_to_dict,
)

__all__ = [
    'ErrorInjector', 'LabelErrorPattern', 'analyze_label_error_pattern',
    'AutoDetector', 'RahaBasedDetector', 'RuleBasedDetector', 'OracleDetector',
    'ParsedRules', 'DomainRule', 'RegexRule', 'CFDRule',
    'parse_rules_file', 'load_rules', 'extract_fd_pairs',
    'get_domain_range', 'get_domain_enum', 'get_cfd_rules_for_class',
    'rules_to_dict',
]
