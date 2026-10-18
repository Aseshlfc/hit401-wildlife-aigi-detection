# Prototype v2 screenshots (2026-09-29, Google Colab Tesla T4, Gradio 5.50.0)

All screenshots show **real outputs of the frozen detectors** from the live app, captured after the verification checks
passed (`verification/results_colab_2026-09-29*/`). Nothing was mocked or edited. They are software demonstrations,
not research results. Display note: the page was shown at 80% browser zoom (display only; the app is unchanged).

| File | What it shows | Input |
|---|---|---|
| v2_01_single_image_analysis.jpg | Single-image mode: preview, both detectors' model score / frozen threshold / decision, agreement, status, file metadata | sdxl_opossum_community_500004.jpg (Dataset v1 validation, SDXL) |
| v2_02_batch_overview_gallery.jpg | Batch of 14 files: counters (14 submitted, 12 analysed, 5/4/3, 2 failed/unsupported), gallery with status captions | 12 bundled verification images + notes.txt + corrupt.jpg |
| v2_03_review_suggested_filter_detail.jpg | "Review Suggested" filter (3 real saved disagreements) and detail of flux_bobcat_community_640037 (original REAL, adapted AI-GENERATED) | same batch |
| v2_04_batch_dashboard_results_table.jpg | Batch Dashboard: counters, per-file results table incl. the UNSUPPORTED .txt | same batch |
| v2_05_csv_and_summary_export.jpg | CSV export (2.8 KB) and text summary export generated | same batch |
| v2_06_file_metadata_tab.jpg | File Metadata tab for all files with the "metadata is not proof" note | same batch |
| v2_07_exif_gps_privacy_generated_test_file.jpg | EXIF extraction and GPS privacy: camera fields shown, "GPS metadata present: Yes", no coordinates | GENERATED_TEST_FILE_exif_gps_not_wildlife.jpg: a plain synthetic colour field with fake EXIF/GPS written by the test code. **Its detector scores are meaningless** (not a wildlife image). |
| v2_08_about_methodology.jpg | About / Methodology: frozen models, score ≠ confidence, status rules, research findings incl. the FLUX negative result | – |
