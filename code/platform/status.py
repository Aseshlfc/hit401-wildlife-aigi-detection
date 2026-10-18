"""Decision and platform-status rules (pure functions, no model needed).

Per-detector decision: the frozen research rule, score >= frozen threshold -> AI-GENERATED, otherwise REAL.
Platform status: agreement between the two frozen detectors only. There is deliberately NO score-distance band
(Prototype v1's "within 0.10 of the threshold" note was removed: it had no validated basis).
"""
from config import T_ORIGINAL, T_ADAPTED

REAL, AI = "REAL", "AI-GENERATED"
LIKELY_AUTHENTIC, LIKELY_AI, REVIEW = "LIKELY AUTHENTIC", "LIKELY AI-GENERATED", "REVIEW SUGGESTED"
STATUSES = (LIKELY_AUTHENTIC, LIKELY_AI, REVIEW)

STATUS_EXPLANATION = {
    LIKELY_AUTHENTIC: "Both research detectors classified this image as real. This is not proof of authenticity: "
                      "both detectors can miss AI-generated images (in the project's held-out-generator test, 15 of 326 "
                      "FLUX.1-schnell images were classified as real by both).",
    LIKELY_AI: "Both research detectors classified this image as AI-generated. This is not forensic proof: "
               "authentic photographs can also be misclassified.",
    REVIEW: "The two research detectors **disagree**, so human inspection is recommended. "
            "This is not a calibrated measure of uncertainty.",
}


def decide(score, threshold):
    """Frozen decision rule used in Experiments 1, 3 and 4 and in Prototype v1."""
    return AI if score >= threshold else REAL


def platform_status(original_decision, adapted_decision):
    for d in (original_decision, adapted_decision):
        if d not in (REAL, AI):
            raise ValueError(f"unknown decision: {d!r}")
    if original_decision == adapted_decision:
        return LIKELY_AUTHENTIC if original_decision == REAL else LIKELY_AI
    return REVIEW


def evaluate_scores(original_score, adapted_score):
    """Scores -> decisions, agreement and platform status, using the frozen thresholds."""
    do, da = decide(original_score, T_ORIGINAL), decide(adapted_score, T_ADAPTED)
    return {"original_score": float(original_score), "original_threshold": T_ORIGINAL, "original_decision": do,
            "adapted_score": float(adapted_score), "adapted_threshold": T_ADAPTED, "adapted_decision": da,
            "models_agree": do == da, "platform_status": platform_status(do, da)}
