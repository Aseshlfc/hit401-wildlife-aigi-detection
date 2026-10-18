# Prototype v2 verification summary

**Software verification, not a research experiment.**
- Nothing was trained, tuned or re-selected, and no TEST-split image was scored.
- Demo-batch counts are not accuracy figures.

**Environment:** Google Colab, Tesla T4, 2026-09-29. Packages: Python 3.13.15, torch 2.11.0+cu128,
transformers 4.57.3, timm 1.0.10, gradio 5.50.0, huggingface_hub 0.36.0. This is the same torch/transformers stack as
Experiments 3–4 and Prototype v1.

**Evidence** (identical results in both runs):
- `results_colab_2026-09-29/` (run 1, 13:27 UTC): `v2_verification.json`, `v2_verification_summary.md`,
  `sample_batch_results.csv`.
- `results_colab_2026-09-29_run2/` (run 2, after a full runtime restart, 13:37 UTC).

## Results
| Check | Result |
|---|---|
| **A. Model integrity** | PASS |
| **B. Known-image score reproduction** | PASS |
| **C. Batch-size consistency** | PASS |
| **D. Mixed batch (17 files)** | PASS |
| **E. Review Suggested** | PASS |
| **F. File handling** | PASS |
| Start-up self-check (inside the app) | PASS |
| Model-free unit tests | 31 / 31 PASS |

**A. Model integrity**
- Base checkpoint SHA-256 `919338928cab…f906`, adapted file `733ab455…cd66` and config `213ae1d6…ff4a` all match.
- OmniAID commit `40749406` is clean.
- Thresholds are exactly 0.35996127128601074 and 0.5848991870880127.
- Both models are in eval mode, with 507,797,575 parameters.
- The adapted model differs from the original in exactly the 1,734 tensors of the adapted file (identical to v1).

**B. Known-image score reproduction**
- 12 bundled images: max |diff| **3.6e-6** vs the saved research scores (tolerance 1e-4); all decisions and statuses
  match.
- The 4 Prototype v1 examples in the bundle are **identical to v1's recorded scores at 6 d.p.**
- v1's 4 start-up self-check images and v1's iNaturalist example (Drive): max |diff| 8.9e-7; decisions match.

**C. Batch-size consistency:** the same 12 images scored one at a time vs in one batch differ by at most 3.2e-6.

**D. Mixed batch of 17 files** (12 bundled images + no-EXIF JPEG + RGBA PNG + `.txt` + corrupt JPEG + truncated JPEG)
- 17 records and 17 CSV rows.
- 14 analysed, 3 failed or unsupported: `.txt` → UNSUPPORTED; corrupt and truncated → FAILED.
- No silent skips, no paths in the CSV, and metadata extracted.

**E. Review Suggested:** all 3 real saved disagreements (iNaturalist skunk, SDXL bobcat, FLUX bobcat) are shown as
REVIEW SUGGESTED.

**F. File handling:** see D, plus the unit tests: text renamed `.jpg`, empty file, tiny image, `.tif`, and GPS/EXIF
privacy.

**Start-up self-check:** "12 verification images reproduced their saved research scores (max difference 3.6e-06,
no decision changes)".

## Known-image comparison (bundled images; saved = research prediction files)
| File | Original v2 / saved | Adapted v2 / saved | Status |
|---|---|---|---|
| cct_opossum_5976a4fe… | 0.169544 / 0.169544 | 0.052635 / 0.052635 | LIKELY AUTHENTIC |
| cct_raccoon_59c4c055… | 0.019895 / 0.019895 | 0.018220 / 0.018220 | LIKELY AUTHENTIC |
| inat_bobcat_…_1997958 | 0.035581 / 0.035581 | 0.041286 / 0.041286 | LIKELY AUTHENTIC |
| inat_bobcat_…_3789805 | 0.018543 / 0.018543 | 0.014920 / 0.014919 | LIKELY AUTHENTIC |
| inat_skunk_…_5178747 | 0.527663 / 0.527663 | 0.235496 / 0.235497 | REVIEW SUGGESTED |
| sdxl_opossum_community_500004 | 0.846297 / 0.846297 | 0.979384 / 0.979384 | LIKELY AI-GENERATED |
| sdxl_opossum_camera_night_ir_504002 | 0.621137 / 0.621137 | 0.919846 / 0.919845 | LIKELY AI-GENERATED |
| sdxl_bobcat_camera_night_ir_544080 | 0.316678 / 0.316676 | 0.799990 / 0.799990 | REVIEW SUGGESTED |
| flux_bobcat_community_640016 | 0.638472 / 0.638472 | 0.896347 / 0.896345 | LIKELY AI-GENERATED |
| flux_coyote_camera_night_ir_634000 | 0.928481 / 0.928480 | 0.962741 / 0.962741 | LIKELY AI-GENERATED |
| flux_bobcat_community_640037 | 0.322940 / 0.322936 | 0.787237 / 0.787233 | REVIEW SUGGESTED |
| flux_squirrel_camera_day_652005 | 0.203043 / 0.203044 | 0.028488 / 0.028488 | LIKELY AUTHENTIC (a FLUX fake: known limitation) |

## Finding recorded, not a failure: razor-edge validation image
`sdxl_opossum_camera_night_ir_504061.jpg` (Dataset v1 validation, SDXL):

| Original-model score | Value | Relative to threshold 0.35996127128601074 | Decision |
|---|---|---|---|
| Saved (research) | 0.35996109 | just below | REAL |
| v2 | 0.35996187 | just above | AI-GENERATED |

- The difference, 7.7e-7, is within tolerance, but the decision flips.
- As agreed in advance, this image was excluded from decision verification and only its score reproduction is
  recorded.
- The cause is ordinary floating-point variation between batched GPU runs of the same unchanged code (Prototype v1's
  pathway behaves the same way).
- Consequence for users: an image whose score lies within about 1e-5 of a threshold can receive the opposite decision
  from the research record. Such cases are extremely rare and are covered by the platform's general caution that
  decisions are not proof.

## Defects found and fixed during v2 development (UI and launch only; inference unchanged)
1. Unit-test fixture did not write GPS/EXIF sub-IFDs (Pillow API) → fixture fixed.
2. Gallery captions and detector-table headers truncated → CSS and column widths.
3. Redundant "detectors disagree" sentence → removed.
4. The 12-column table wrapped letter by letter → wide tables now scroll horizontally.
5. PNG previews were slow → JPEG previews (still EXIF-free and in stored orientation).
6. A single image filled the whole gallery width → fixed-size gallery grid.
7. The launcher's `eval_js` proxy-URL call could hang if the notebook tab was unresponsive → added `timeout_sec=60`.
8. Package test: cancelling the launcher's upload button raised an unhelpful `StopIteration` → clear message instead.
9. Package test: the upload button lives in a sandboxed output frame that browser automation cannot operate → the
   launcher now also picks up files uploaded through Colab's Files panel (`/content/…`). This became the documented
   QUICK START method, and it is the path that was tested end to end.
