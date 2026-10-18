"""DEVELOPMENT ONLY - UI wiring check without a GPU. NOT part of the distributable package.

No model runs. The stub replays the SAVED research scores for the bundled verification images and returns 0.5/0.5 for
anything else. The page carries a banner saying so. Never use this for screenshots, evidence or demonstrations.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402
from config import EXPECTED_SCORES  # noqa: E402


class ReplayStub:
    def __init__(self):
        e = pd.read_csv(EXPECTED_SCORES)
        self.s = {r.filename: {"original": r.expected_original_score, "adapted": r.expected_adapted_score} for r in e.itertuples()}

    def score_files(self, paths):
        return {p: dict(self.s.get(Path(p).name, {"original": 0.5, "adapted": 0.5})) for p in paths}


if __name__ == "__main__":
    demo = app.build_ui(ReplayStub(), "<b>UI WIRING TEST - NO MODEL IS RUNNING. Numbers are replayed saved scores or placeholders. "
                                      "Not for screenshots or evidence.</b>")
    demo.queue().launch(server_port=7861, allowed_paths=[str(app.WORK_DIR), str(app.VERIFY_IMAGES_DIR)])
