import pandas as pd
import pytest

from config import EXPECTED_SCORES, T_ADAPTED, T_ORIGINAL
from status import AI, LIKELY_AI, LIKELY_AUTHENTIC, REAL, REVIEW, decide, evaluate_scores, platform_status


def test_frozen_thresholds_unchanged():
    assert T_ORIGINAL == 0.35996127128601074
    assert T_ADAPTED == 0.5848991870880127


def test_decision_rule_boundary():
    # score >= threshold -> AI-GENERATED (frozen rule from Experiments 1, 3, 4)
    assert decide(T_ORIGINAL, T_ORIGINAL) == AI
    assert decide(T_ORIGINAL - 1e-12, T_ORIGINAL) == REAL
    assert decide(1.0, T_ADAPTED) == AI and decide(0.0, T_ADAPTED) == REAL


@pytest.mark.parametrize("o,a,expected", [(REAL, REAL, LIKELY_AUTHENTIC), (AI, AI, LIKELY_AI), (AI, REAL, REVIEW), (REAL, AI, REVIEW)])
def test_platform_status_truth_table(o, a, expected):
    assert platform_status(o, a) == expected


def test_platform_status_rejects_unknown():
    with pytest.raises(ValueError):
        platform_status("MAYBE", REAL)


def test_no_distance_band():
    # a score a hair above/below a threshold is treated exactly like any other score (no "near threshold" band)
    r = evaluate_scores(T_ORIGINAL + 1e-9, T_ADAPTED + 1e-9)
    assert r["platform_status"] == LIKELY_AI and r["models_agree"]
    r = evaluate_scores(T_ORIGINAL - 1e-9, T_ADAPTED + 0.4)
    assert r["platform_status"] == REVIEW and not r["models_agree"]


def test_expected_scores_file_is_consistent_with_rules():
    """Saved research scores of the bundled images -> the documented decisions/statuses (includes 3 real disagreements)."""
    exp = pd.read_csv(EXPECTED_SCORES)
    assert len(exp) == 12
    for r in exp.itertuples():
        e = evaluate_scores(r.expected_original_score, r.expected_adapted_score)
        assert (e["original_decision"], e["adapted_decision"], e["platform_status"]) == \
               (r.expected_original_decision, r.expected_adapted_decision, r.expected_platform_status), r.filename
    assert (exp.expected_platform_status == REVIEW).sum() == 3
