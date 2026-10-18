"""CSV and plain-text summary export. No GPS values, owner/serial fields or file paths are exported."""
import csv
import datetime
from pathlib import Path

from status import LIKELY_AI, LIKELY_AUTHENTIC, REVIEW

CSV_COLUMNS = ["filename", "dimensions", "format", "file_size_bytes", "colour_mode", "exif_present", "gps_metadata_present",
               "original_score", "original_threshold", "original_decision",
               "adapted_score", "adapted_threshold", "adapted_decision",
               "models_agree", "platform_status", "processing_status", "processing_note"]


def _safe_cell(v):
    """Neutralise spreadsheet formula injection in text cells (e.g. a file named '=cmd.jpg')."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return repr(v)
    return "" if v is None else v


def write_csv(records, path):
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: _safe_cell(_fmt(r.get(k))) for k in CSV_COLUMNS})
    return str(path)


def summarise(records, elapsed_s=None):
    analysed = [r for r in records if r.get("processing_status") == "ANALYSED"]
    n = len(analysed)
    s: dict = {"total_submitted": len(records), "analysed": n,
         "likely_authentic": sum(r["platform_status"] == LIKELY_AUTHENTIC for r in analysed),
         "likely_ai_generated": sum(r["platform_status"] == LIKELY_AI for r in analysed),
         "review_suggested": sum(r["platform_status"] == REVIEW for r in analysed),
         "failed_or_unsupported": len(records) - n,
         "models_agree": sum(bool(r["models_agree"]) for r in analysed)}
    s["agreement_rate"] = (s["models_agree"] / n) if n else None
    s["processing_seconds"] = elapsed_s
    s["seconds_per_analysed_image"] = (elapsed_s / n) if (elapsed_s and n) else None
    return s


def summary_text(records, elapsed_s=None):
    s = summarise(records, elapsed_s)
    pct = lambda k: f" ({100 * s[k] / s['analysed']:.0f}%)" if s["analysed"] else ""  # noqa: E731
    lines = [
        "Wildlife Image Authenticity Analysis Platform - batch summary",
        f"Generated: {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        "",
        f"Images submitted:        {s['total_submitted']}",
        f"Successfully analysed:   {s['analysed']}",
        f"Likely authentic:        {s['likely_authentic']}{pct('likely_authentic')}",
        f"Likely AI-generated:     {s['likely_ai_generated']}{pct('likely_ai_generated')}",
        f"Review suggested:        {s['review_suggested']}{pct('review_suggested')}",
        f"Failed / unsupported:    {s['failed_or_unsupported']}",
        f"Detector agreement rate: {s['agreement_rate']:.1%}" if s["agreement_rate"] is not None else "Detector agreement rate: n/a",
    ]
    if s["processing_seconds"] is not None:
        lines.append(f"Processing time:         {s['processing_seconds']:.1f} s (application timing only)")
    lines += ["", "Status rules: both detectors REAL -> LIKELY AUTHENTIC; both AI-GENERATED -> LIKELY AI-GENERATED; "
              "disagreement -> REVIEW SUGGESTED (human inspection recommended; not a calibrated uncertainty).",
              "Model scores are raw detector outputs, not calibrated probabilities or confidence.",
              "These counts describe this batch only. They are not accuracy figures and not research results.",
              "Research prototype: do not use as the sole basis for fraud, legal, disciplinary, misconduct or publication decisions."]
    failed = [r for r in records if r.get("processing_status") != "ANALYSED"]
    if failed:
        lines += ["", "Files not analysed:"] + [f"  - {r['filename']}: {r.get('processing_note', '')}" for r in failed]
    return "\n".join(lines) + "\n"


def write_summary(records, path, elapsed_s=None):
    Path(path).write_text(summary_text(records, elapsed_s), encoding="utf-8")
    return str(path)
