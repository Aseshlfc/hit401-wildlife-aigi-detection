# Sample output

`sample_batch_results.csv` is the real CSV written by the platform during the verified Colab run of 2026-09-29
(`verification/verify_v2.py`, check D). The batch had 17 files: the 12 bundled verification images plus five generated
test files (a plain no-EXIF JPEG, an RGBA PNG, a `.txt`, a corrupt JPEG and a truncated JPEG). The two generated
images are plain colour fields, so their scores have no meaning; they only exercise the software. Counts in this file
are not accuracy figures.
