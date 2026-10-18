"""Batch bookkeeping and CSV export with a STUB detector (fixed numbers, not model output)."""
import csv

from PIL import Image

from batch_processing import analyse_files
from exports import CSV_COLUMNS, summarise, summary_text, write_csv
from status import LIKELY_AI, LIKELY_AUTHENTIC, REVIEW


def _make(tmp_path):
    files = {}
    for n in ("a.jpg", "b.jpg", "c.png", "d.jpg", "=evil.jpg"):
        p = tmp_path / n; Image.new("RGB", (40, 30), (10, 20, 30)).save(p); files[n] = p
    t = tmp_path / "readme.txt"; t.write_text("x"); files["readme.txt"] = t
    c = tmp_path / "broken.jpg"; c.write_bytes(b"\xff\xd8\xff garbage"); files["broken.jpg"] = c
    return files


SCORES = {"a.jpg": {"original": 0.10, "adapted": 0.10},    # both REAL
          "b.jpg": {"original": 0.90, "adapted": 0.90},    # both AI
          "c.png": {"original": 0.50, "adapted": 0.20},    # original AI, adapted REAL
          "d.jpg": {"original": 0.20, "adapted": 0.70},    # original REAL, adapted AI
          "=evil.jpg": {"original": 0.10, "adapted": 0.10}}


def test_batch_no_silent_skips(tmp_path, stub_cls):
    files = _make(tmp_path)
    det = stub_cls(SCORES)
    recs, elapsed = analyse_files(det, [(str(p), n) for n, p in files.items()])
    assert len(recs) == len(files) == 7
    by = {r["filename"]: r for r in recs}
    assert by["a.jpg"]["platform_status"] == LIKELY_AUTHENTIC
    assert by["b.jpg"]["platform_status"] == LIKELY_AI
    assert by["c.png"]["platform_status"] == REVIEW and by["d.jpg"]["platform_status"] == REVIEW
    assert by["readme.txt"]["processing_status"] == "UNSUPPORTED"
    assert by["broken.jpg"]["processing_status"] == "FAILED"
    # invalid files never reach the model
    sent = {p.split("/")[-1] for call in det.calls for p in call}
    assert "readme.txt" not in sent and "broken.jpg" not in sent
    s = summarise(recs, elapsed)
    assert (s["total_submitted"], s["analysed"], s["likely_authentic"], s["likely_ai_generated"], s["review_suggested"],
            s["failed_or_unsupported"]) == (7, 5, 2, 1, 2, 2)
    assert abs(s["agreement_rate"] - 3 / 5) < 1e-12


def test_model_side_failure_is_reported_not_skipped(tmp_path, stub_cls):
    files = _make(tmp_path)
    det = stub_cls(SCORES, drop={"b.jpg"})   # simulates the official dataset substituting an unreadable file
    recs, _ = analyse_files(det, [(str(files["a.jpg"]), "a.jpg"), (str(files["b.jpg"]), "b.jpg")])
    assert [r["processing_status"] for r in recs] == ["ANALYSED", "FAILED"]


def test_scoring_exception_keeps_app_alive(tmp_path):
    class Boom:
        def score_files(self, paths):
            raise RuntimeError("CUDA out of memory")
    files = _make(tmp_path)
    recs, _ = analyse_files(Boom(), [(str(files["a.jpg"]), "a.jpg")])
    assert recs[0]["processing_status"] == "FAILED" and "CUDA out of memory" in recs[0]["processing_note"]


def test_chunking_covers_every_file(tmp_path, stub_cls):
    scores = {}
    files = []
    for i in range(37):
        p = tmp_path / f"img{i:02d}.jpg"; Image.new("RGB", (16, 16)).save(p)
        scores[p.name] = {"original": 0.1, "adapted": 0.1}; files.append((str(p), p.name))
    det = stub_cls(scores)
    recs, _ = analyse_files(det, files)
    assert len(recs) == 37 and all(r["processing_status"] == "ANALYSED" for r in recs)
    assert sum(len(c) for c in det.calls) == 37 and max(len(c) for c in det.calls) <= 16


def test_csv_export(tmp_path, stub_cls):
    files = _make(tmp_path)
    recs, el = analyse_files(stub_cls(SCORES), [(str(p), n) for n, p in files.items()])
    out = write_csv(recs, tmp_path / "out.csv")
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 7 and list(rows[0]) == CSV_COLUMNS
    assert all("path" not in k and "gps" not in k or k == "gps_metadata_present" for k in rows[0])
    text = open(out, encoding="utf-8").read()
    assert str(tmp_path) not in text                        # no local paths exported
    evil = [r for r in rows if "evil" in r["filename"]][0]
    assert evil["filename"].startswith("'=")                # formula injection neutralised
    a = [r for r in rows if r["filename"] == "a.jpg"][0]
    assert float(a["original_threshold"]) == 0.35996127128601074 and float(a["adapted_threshold"]) == 0.5848991870880127
    assert a["models_agree"] == "yes" and a["platform_status"] == LIKELY_AUTHENTIC
    txt = summary_text(recs, el)
    assert "Review suggested:        2" in txt and "not accuracy figures" in txt and "broken.jpg" in txt
