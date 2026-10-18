# Loader changes: Prototype v1 `omniaid_models.py` → Prototype v2 `inference.py`

`inference.py` started as a byte copy of the verified v1 file (SHA-256 `803992aa…`, kept unchanged in
`reference_v1/omniaid_models.py`). The full unified diff is in `loader_v1_to_v2.diff`.
`tests/test_loader_integrity.py` checks these points automatically with Python's `ast` module:
- `score_files` is AST-identical to v1;
- the model-construction block differs in exactly two statements (both paths);
- all frozen constants equal v1's.

## Unchanged (inference semantics)
- The official OmniAID repository at commit `40749406…`, plus `main_finetune.get_args_parser` with the **identical
  argument list**. The only difference is the value of the unused `--output_dir`.
- `setup_for_training`, `create_model_and_ema`, and a strict load of the base checkpoint.
- The adapted model is a `deepcopy` of the original plus a non-strict load of `adapted_state`, with the same key and
  shape assertions.
- Both models in `eval()`, fp32, `torch.no_grad()`.
- `score_files`, verbatim:
  - files are symlinked, never re-encoded;
  - the official `GenerativeImageDataset(is_train=False)` eval transform;
  - `DataLoader(batch_size=8, num_workers=0)`;
  - output `net(x)["prob"]`.
- Thresholds 0.35996127128601074 / 0.5848991870880127, and decision rule `score >= threshold` → AI-GENERATED.

## Changed (locations and checks only)
| # | v1 | v2 | Why |
|---|---|---|---|
| 1 | Constants defined in the file, with Drive paths `DRIVE`, `ADAPT`, `REPO=/content/OmniAID_prototype` | Constants imported from `config.py` (same values); paths overridable by `HIT401_*` environment variables | Portability: no dependency on a personal Drive layout |
| 2 | Adapted file path read from `frozen_selection.json` (an absolute Drive path) | `config.ADAPTED_FILE` (default `models/adapted_epoch_1.pt`); clear error if missing. The frozen epoch and threshold are asserted from config (copied from `frozen_selection.json`) | Portability |
| 3 | `cfg = ADAPT / "config_omniaid_v2_recovered.json"` | `cfg = ARCH_CONFIG` (default `models/config_omniaid_v2_recovered.json`, the **same project file**, SHA-256 `213ae1d6…` checked at start) | Portability. The official HF `config_omniaid_v2.json` differs in training-only fields, so v2 keeps the project file used by Exp 3/4 and v1 |
| 4 | `--output_dir /content/prototype_run` | `--output_dir <RUN_DIR>` | Path only. Not used at inference |
| 5 | Base checkpoint: `hf_hub_download(...)` | Same call, unless `HIT401_BASE_CKPT` points to a local copy. Its SHA-256 is still checked | Optional offline use |
| 6 | Hash checks: base + adapted, both skippable with `HIT401_SKIP_HASH=1` | Adapted file and config are **always** hashed. Only the 3.24 GB base hash is skippable, as in v1. The app shows whether the base hash was re-checked | Stricter |
| 7 | No commit check after checkout | Asserts `HEAD == 40749406…` and a clean working tree (the same check `exp4_evaluate.py` made) | Stricter |
| 8 | `self_check()` re-scored 4 Drive validation images against saved validation CSVs (tolerance 1e-4) | `self_check()` re-scores the 12 **bundled** verification images against `expected_scores.csv`. These are saved scores copied from the same validation CSVs and from `exp4_predictions.csv`. Tolerance still 1e-4, plus a check that no decision flips | Works without the project Drive |

## Platform-level additions outside the loader (they do not touch inference)
- `metadata.validate_image` rejects unsupported, corrupt or truncated files **before** scoring. Two behaviours of the
  official dataset code make this necessary:
  - `ImageFile.LOAD_TRUNCATED_IMAGES = True`;
  - an unreadable image is replaced by a random other image.
- `batch_processing` calls `score_files` in chunks of 16 files (only to show progress). It cross-checks that every
  file received its own two scores.
- v1's "score within 0.10 of the threshold" note was **removed**. There is no score-distance band.
