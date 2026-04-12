from __future__ import annotations

from typing import Iterable, Set

STRICT_EPSILON = 0.001


def _normalize(values: Iterable[str]) -> Set[str]:
    return {str(value).strip().lower() for value in values}


def _clamp_01(score: float) -> float:
    return max(STRICT_EPSILON, min(1.0 - STRICT_EPSILON, score))


def easy_grader(identified_fields: Iterable[str], expected_fields: Iterable[str]) -> float:
    """EASY: score = correctly identified PII / total expected PII."""
    identified = _normalize(identified_fields)
    expected = _normalize(expected_fields)
    if not expected:
        return 1.0 - STRICT_EPSILON
    return round(_clamp_01(len(identified & expected) / len(expected)), 3)


def medium_grader(flagged_violations: Iterable[str], expected_violations: Iterable[str]) -> float:
    """MEDIUM: score = F1 for violation detection."""
    flagged = _normalize(flagged_violations)
    expected = _normalize(expected_violations)
    if not expected:
        return 1.0 - STRICT_EPSILON

    true_positives = len(flagged & expected)
    precision = true_positives / len(flagged) if flagged else 0.0
    recall = true_positives / len(expected)

    if precision + recall == 0.0:
        return STRICT_EPSILON

    f1 = 2.0 * precision * recall / (precision + recall)
    return round(_clamp_01(f1), 3)


def hard_grader(
    identified_pii: Iterable[str],
    expected_pii: Iterable[str],
    flagged_violations: Iterable[str],
    expected_violations: Iterable[str],
    correct_remediations: int,
    required_remediations: int,
    audit_completed: bool,
) -> float:
    """
    HARD weighted score:
    - PII quality (35%)
    - Violation detection quality (40%)
    - Integrity quality (25%), where integrity combines remediation and completion.
    """
    pii_score = easy_grader(identified_pii, expected_pii)
    violation_score = medium_grader(flagged_violations, expected_violations)

    if required_remediations <= 0:
        remediation_ratio = 1.0
    else:
        remediation_ratio = _clamp_01(correct_remediations / required_remediations)

    completion_score = 1.0 if audit_completed else 0.0
    integrity_score = (0.70 * remediation_ratio) + (0.30 * completion_score)

    total = (0.35 * pii_score) + (0.40 * violation_score) + (0.25 * integrity_score)
    return round(_clamp_01(total), 3)
