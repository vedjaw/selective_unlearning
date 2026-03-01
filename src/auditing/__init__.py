"""
Auditing module for machine unlearning verification.

Provides information-theoretic tools to audit whether approximate
unlearning has truly removed information about the forget set.

Core components:
- RINE: Redundant Information Neural Estimator
- ResNetActivationExtractor: Hook-based representation extraction
- extract_for_audit: Convenience function for the full extraction pipeline
"""

from .rine import RINE, LogisticProbe
from .activation_extractor import (
    ResNetActivationExtractor,
    extract_for_audit,
)

__all__ = [
    'RINE',
    'LogisticProbe',
    'ResNetActivationExtractor',
    'extract_for_audit',
]
