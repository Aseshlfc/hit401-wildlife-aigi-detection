"""Static proof that the v2 loader keeps Prototype v1's inference code (no model or GPU needed).

Compares inference.py with the verified v1 file kept in verification/reference_v1/omniaid_models.py."""
import ast
import hashlib
import json
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "verification" / "reference_v1" / "omniaid_models.py"
V2 = ROOT / "inference.py"
V1_SHA256 = "803992aad09e95a57445d161cd4700e78e19fb1c88396cde5d6847c4da3c5ca5"   # verified Prototype v1 file


def _func(src, cls, name):
    tree = ast.parse(src)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _consts(src):
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(n.value)
                    except Exception:
                        pass
                elif isinstance(t, ast.Tuple):
                    try:
                        for tt, vv in zip(t.elts, ast.literal_eval(n.value)):
                            out[tt.id] = vv
                    except Exception:
                        pass
    return out


def test_reference_is_the_verified_v1_file():
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == V1_SHA256


def test_score_files_identical_to_v1():
    a, b = _func(V1.read_text(), "Detectors", "score_files"), _func(V2.read_text(), "Detectors", "score_files")
    assert ast.dump(a) == ast.dump(b)


def test_model_construction_identical_to_v1_except_paths():
    """Every statement of __init__ from 'import main_finetune' to the end is identical, except the three path lines."""
    def tail(src):
        f = _func(src, "Detectors", "__init__")
        lines = [ast.unparse(s) for s in f.body]
        start = next(i for i, l in enumerate(lines) if "os.chdir(REPO)" in l)
        return lines[start:]
    v1, v2 = tail(V1.read_text()), tail(V2.read_text())
    assert len(v1) == len(v2)
    diffs = [(x, y) for x, y in zip(v1, v2) if x != y]
    assert len(diffs) == 2, diffs
    # 1) architecture config path, 2) the unused --output_dir path inside the otherwise identical argument list
    assert diffs[0] == ("cfg = ADAPT / 'config_omniaid_v2_recovered.json'", "cfg = ARCH_CONFIG")
    assert diffs[1][0].replace("'/content/prototype_run'", "str(RUN_DIR)") == diffs[1][1]


def test_frozen_constants_identical_to_v1():
    c1 = _consts(V1.read_text())
    for k in ("OMNIAID_REPO_URL", "OMNIAID_COMMIT", "HF_REPO", "HF_REVISION", "HF_FILENAME", "BASE_CKPT_SHA256",
              "ADAPTED_SHA256", "T_ORIGINAL", "T_ADAPTED"):
        assert getattr(config, k) == c1[k], k


def test_arch_config_file_is_the_project_file():
    assert hashlib.sha256(config.ARCH_CONFIG.read_bytes()).hexdigest() == config.ARCH_CONFIG_SHA256
    assert json.loads(config.ARCH_CONFIG.read_text())["CLIP_path"] == "openai/clip-vit-large-patch14-336"
