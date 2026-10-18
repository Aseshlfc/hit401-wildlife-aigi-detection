import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StubDetectors:
    """TEST DOUBLE for model-free unit tests only. Returns fixed numbers keyed by file name; these are NOT model outputs
    and are never shown in the app, screenshots or verification evidence."""

    def __init__(self, scores, drop=()):
        self.scores, self.drop, self.calls = scores, set(drop), []

    def score_files(self, paths):
        self.calls.append(list(paths))
        return {p: ({} if Path(p).name in self.drop else dict(self.scores[Path(p).name])) for p in paths}


@pytest.fixture
def stub_cls():
    return StubDetectors
