# Prototype v2 verification (2026-09-29T13:27:56+00:00)

Software verification only; not a research experiment. No TEST image was scored.

Environment: {'gpu': 'Tesla T4', 'python': '3.13.15', 'torch': '2.11.0+cu128', 'transformers': '4.57.3', 'timm': '1.0.10', 'gradio': '5.50.0', 'huggingface_hub': '0.36.0', 'PIL': '11.3.0'}

| Check | Result |
|---|---|
| A. Model integrity (hashes, commit, thresholds, eval mode, 507,797,575 params, 1,734 adapted tensors) | PASS |
| B. Known-image score reproduction (max diff bundled 3.61e-06; Drive 8.94e-07; tolerance 0.0001) | PASS |
| C. Batch vs single-image scoring (max diff 3.22e-06) | PASS |
| D. Mixed batch of 17 files: counts, statuses, CSV rows, metadata, edge files | PASS |
| E. Review Suggested on the 3 real saved disagreement examples | PASS |

## Bundled images: v2 vs saved research scores

| File | v2 original | saved | v2 adapted | saved | max diff | status |
|---|---|---|---|---|---|---|
| cct_opossum_5976a4fe-23d2-11e8-a6a3-ec086b02610b.jpg | 0.169544 | 0.169544 | 0.052635 | 0.052635 | 2.4e-07 | LIKELY AUTHENTIC |
| cct_raccoon_59c4c055-23d2-11e8-a6a3-ec086b02610b.jpg | 0.019895 | 0.019895 | 0.018220 | 0.018220 | 2.2e-07 | LIKELY AUTHENTIC |
| inat_bobcat_1628425_1997958.jpg | 0.035581 | 0.035581 | 0.041286 | 0.041286 | 1.5e-07 | LIKELY AUTHENTIC |
| inat_bobcat_3277526_3789805.jpg | 0.018543 | 0.018543 | 0.014920 | 0.014919 | 7.5e-09 | LIKELY AUTHENTIC |
| inat_skunk_4310670_5178747.jpg | 0.527663 | 0.527663 | 0.235496 | 0.235497 | 4.8e-07 | REVIEW SUGGESTED |
| sdxl_opossum_community_500004.jpg | 0.846297 | 0.846297 | 0.979384 | 0.979384 | 6.0e-08 | LIKELY AI-GENERATED |
| sdxl_opossum_camera_night_ir_504002.jpg | 0.621137 | 0.621137 | 0.919846 | 0.919845 | 1.0e-06 | LIKELY AI-GENERATED |
| sdxl_bobcat_camera_night_ir_544080.jpg | 0.316678 | 0.316676 | 0.799990 | 0.799990 | 1.8e-06 | REVIEW SUGGESTED |
| flux_bobcat_community_640016.jpg | 0.638472 | 0.638472 | 0.896347 | 0.896345 | 1.6e-06 | LIKELY AI-GENERATED |
| flux_coyote_camera_night_ir_634000.jpg | 0.928481 | 0.928480 | 0.962741 | 0.962741 | 7.2e-07 | LIKELY AI-GENERATED |
| flux_bobcat_community_640037.jpg | 0.322940 | 0.322936 | 0.787237 | 0.787233 | 3.6e-06 | REVIEW SUGGESTED |
| flux_squirrel_camera_day_652005.jpg | 0.203043 | 0.203044 | 0.028488 | 0.028488 | 9.8e-07 | LIKELY AUTHENTIC |

## Drive-only validation images

| File | role | max diff | decisions match |
|---|---|---|---|
| cct_opossum_5976a4fe-23d2-11e8-a6a3-ec086b02610b.jpg | Prototype v1 start-up self-check image | 5.1e-07 | True |
| cct_opossum_5a2c8638-23d2-11e8-a6a3-ec086b02610b.jpg | Prototype v1 start-up self-check image | 8.3e-07 | True |
| sdxl_opossum_community_500004.jpg | Prototype v1 start-up self-check image | 1.2e-07 | True |
| sdxl_opossum_community_500008.jpg | Prototype v1 start-up self-check image | 7.7e-07 | True |
| inat_opossum_486213_614582.jpg | Prototype v1 example | 5.6e-17 | True |
| sdxl_opossum_camera_night_ir_504061.jpg | razor-edge case (score reproduction only; decision NOT verified) | 8.9e-07 | not checked (razor edge) |
