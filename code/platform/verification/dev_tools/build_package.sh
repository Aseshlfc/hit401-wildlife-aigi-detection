#!/bin/bash
# Build HIT401_Wildlife_Authenticity_Platform.zip from a CLEAN staging folder (explicit allow-list; development-only
# material such as v1_protection/, dev_tools/ and v1's verification JSON with Drive paths is never copied).
# Usage: bash verification/dev_tools/build_package.sh <staging_dir> <output_dir>
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"; STG="$1"; OUT="$2"; P="$STG/HIT401_Wildlife_Authenticity_Platform"
rm -rf "$STG"; mkdir -p "$P"/{launcher,models,licenses,verification_images,screenshots,tests,verification/reference_v1,sample_outputs} "$OUT"
cd "$SRC"
cp -p app.py inference.py config.py status.py batch_processing.py metadata.py exports.py requirements.txt README.md ARCHITECTURE.md THIRD_PARTY_LICENSES.md ATTRIBUTIONS.md "$P/"
cp -p launcher/HIT401_Platform_Colab.ipynb launcher/build_launcher.py "$P/launcher/"
cp -p models/README_MODEL_SETUP.md models/config_omniaid_v2_recovered.json "$P/models/"
cp -p licenses/CDLA-Permissive-1.0.txt "$P/licenses/"
cp -p verification_images/*.jpg verification_images/expected_scores.csv verification_images/README.md verification_images/ATTRIBUTION.csv "$P/verification_images/"
cp -p screenshots/*.jpg screenshots/SCREENSHOT_LOG.md "$P/screenshots/"
cp -p tests/*.py "$P/tests/"
cp -p verification/verify_v2.py verification/VERIFICATION_SUMMARY.md verification/LOADER_CHANGES.md verification/loader_v1_to_v2.diff "$P/verification/"
cp -p verification/reference_v1/omniaid_models.py "$P/verification/reference_v1/"
for d in verification/results_colab_*; do cp -rp "$d" "$P/verification/"; done
cp -p sample_outputs/* "$P/sample_outputs/"
find "$STG" \( -name .DS_Store -o -name '*.pyc' \) -delete; find "$STG" -name __pycache__ -prune -exec rm -rf {} +
(cd "$STG" && rm -f "$OUT/HIT401_Wildlife_Authenticity_Platform.zip" && zip -q -X -r -9 "$OUT/HIT401_Wildlife_Authenticity_Platform.zip" HIT401_Wildlife_Authenticity_Platform)
(cd "$OUT" && shasum -a 256 HIT401_Wildlife_Authenticity_Platform.zip | tee HIT401_Wildlife_Authenticity_Platform.zip.sha256)
