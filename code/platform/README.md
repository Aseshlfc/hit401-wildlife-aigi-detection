# Wildlife Image Authenticity Analysis Platform (HIT401 Prototype v2)

A research prototype for **comparing the original and the wildlife-adapted AI-generated-image detectors** from the
HIT401 capstone project on one or many wildlife images.

It is a **software artefact built around the frozen research models**:
- it does not train, tune or re-select anything;
- it is not a new experiment;
- its outputs are not forensic proof.

Status: **verified on Google Colab (Tesla T4) on 2026-09-29.** See `verification/VERIFICATION_SUMMARY.md`.

---

## QUICK START FOR SUPERVISOR

You need:
- a Google account (for Colab);
- `HIT401_Wildlife_Authenticity_Platform.zip` (this package);
- `adapted_epoch_1.pt` (6.4 MB), which the project team sends you separately.

About 10 minutes the first time.

1. **Open the launcher.** Unzip the package on your computer. Go to https://colab.research.google.com →
   **Upload notebook** → choose `launcher/HIT401_Platform_Colab.ipynb`.
2. **Select the GPU.** The launcher requests a T4 GPU automatically. Check that the top-right shows **T4**; if not,
   use **Runtime → Change runtime type → T4 GPU → Save**.
3. **Upload the two files.** Click the **folder icon** (Files panel) on the left → **Upload to session storage**
   (page-with-arrow icon) → select **both** `HIT401_Wildlife_Authenticity_Platform.zip` and `adapted_epoch_1.pt`. Wait
   until both appear in the file list. If Colab warns that uploaded files are deleted when the runtime ends, click
   **OK**.
4. **Run setup.** **Runtime → Run all.**
   - Cell 1 checks the GPU.
   - Cell 2 unpacks the ZIP.
   - Cell 3 installs the pinned packages (1–2 min).
   - Cell 4 checks `adapted_epoch_1.pt` against its frozen SHA-256; a wrong file is rejected.
   If you skipped step 3, cells 2 and 4 show an upload button instead.
5. **Official base model.** Cell 5 downloads the official OmniAID checkpoint (3.24 GB) from Hugging Face. No
   account or token is needed. If Colab asks about an `HF_TOKEN` secret, click **Cancel**.
6. **Integrity checks run automatically.** Cell 5 also checks every model hash and re-scores the 12 bundled
   verification images against their saved research scores. If anything differs, the platform **refuses to start**.
7. **Launch.** Cell 5 prints `App URL (...)`. Open that link in a new browser tab.
8. **Upload images.** On **Analyse Images**, drop one or more JPEG/PNG/WebP wildlife images and click **Analyse**, or
   click **Analyse bundled verification images**.
9. **Inspect results.**
   - Click any thumbnail to see both detectors' scores, thresholds and decisions, the status and the file metadata.
   - Use **Show → Review Suggested** or the **Review Suggested** tab for images where the detectors disagree.
10. **Export.** On **Batch Dashboard**, click **Export results (CSV)** and/or **Export summary (text)**, then click the
    file to download it.

Optional: cell 6 re-runs the full verification and writes the evidence to `verification/results/` in the Colab session.
If Colab disconnects, re-run the cells. The checkpoint download is repeated in a new runtime.

---

## What the platform does
- **Intended users:** ecologists, wildlife and camera-trap researchers, citizen-science image reviewers.
- **Two detectors per image:**
  - the **original OmniAID v2** detector (Experiment 1);
  - the **wildlife-adapted OmniAID v2** (Experiment 3: parameter-efficient fine-tuning of OmniAID v2).
- **Reported per detector:** the **model score**, its **frozen threshold** and its **decision**.
- **Platform status:** the two decisions are combined into one status.

| Original | Adapted | Platform status | Meaning |
|---|---|---|---|
| REAL | REAL | **LIKELY AUTHENTIC** | Both research detectors classified the image as real. **Not proof of authenticity.** |
| AI-GENERATED | AI-GENERATED | **LIKELY AI-GENERATED** | Both classified it as synthetic. **Not forensic proof.** |
| disagree | disagree | **REVIEW SUGGESTED** | The detectors disagree, so human inspection is recommended. This is **not** calibrated uncertainty. |

- **No uncertainty band.** There is no score-distance "uncertainty band". Prototype v1's "within 0.10 of the
  threshold" note was removed because it had no validated basis.
- **Model score is not confidence.** It is the detector's raw output between 0 and 1 (higher = more AI-like), not a
  calibrated probability.

### Single-image workflow
Upload one image → **Analyse**. You get:
- a preview, in the stored pixel orientation, exactly as the model sees it;
- both detectors' score, threshold and decision;
- whether the detectors agree, and the platform status with its caveat;
- the file metadata.

### Batch workflow
Upload many images → **Analyse**.
- **Dashboard counters:** submitted, analysed, likely authentic, likely AI-generated, review suggested,
  failed/unsupported, detector agreement rate, processing time.
- **Views:** a filterable gallery and results table, and a **Review Suggested** queue.
- **No file is silently skipped.** Unsupported or corrupt files are listed with a reason.
- **Counts are not accuracy.** Counts describe the uploaded batch only; they are not accuracy figures or research
  results.

### File metadata
Shown fields: file type, dimensions, size, colour mode, EXIF present, camera make and model, capture date/time, ISO,
exposure, aperture, focal length, software, other embedded metadata keys, and **GPS metadata present: Yes/No**.

- **Privacy:** precise GPS coordinates, owner, artist, copyright and serial-number fields are **never shown or
  exported**.
- **Context only:** metadata can be modified, removed, copied or fabricated, so it is never used in the decision.

### CSV export
- **Columns:** `filename, dimensions, format, file_size_bytes, colour_mode, exif_present, gps_metadata_present,
  original_score, original_threshold, original_decision, adapted_score, adapted_threshold, adapted_decision,
  models_agree, platform_status, processing_status, processing_note`.
- **Not exported:** GPS values and file paths.
- **Summary file:** the text summary gives the totals and the agreement rate.

---

## How the frozen models are used
- **Original:** the official checkpoint `Yunncheng/OmniAID@279cae73:ckpt/checkpoint_omniaid_v2.pth` (SHA-256
  `919338928cab…f906`) with threshold **0.35996127128601074**.
- **Adapted:** the same checkpoint plus the Experiment 3 parameter file `adapted_epoch_1.pt` (epoch 2, SHA-256
  `733ab455…cd66`) with threshold **0.5848991870880127**.
- **Thresholds:** both were chosen on validation data only (Youden's J) during the research.
- **Loading:** the model-loading and scoring code is the **verified Prototype v1 pathway**, the same as Experiment 4:
  - official OmniAID code at commit `40749406`;
  - official evaluation transform (RGB, 336×336 resize, CLIP normalisation, no EXIF rotation);
  - fp32, eval mode;
  - models loaded once and reused.
  Only file paths were made portable (`verification/LOADER_CHANGES.md`).
- **Model files:** see `models/README_MODEL_SETUP.md`.

## Architecture
See `ARCHITECTURE.md`. Summary:
- validation and metadata (`metadata.py`);
- unchanged v1 scoring (`inference.py`);
- decisions and agreement (`status.py`);
- batch bookkeeping (`batch_processing.py`);
- export (`exports.py`);
- Gradio UI (`app.py`).

## Requirements and setup
- **Recommended:** Google Colab with a **T4 GPU** (tested). `requirements.txt` pins the research versions
  (transformers 4.57.3, timm 1.0.10, huggingface_hub 0.36.0, gradio 5.50.0). Torch comes from Colab (tested with
  2.11.0+cu128).
- **Local Python: not tested.** A CUDA GPU with at least about 8 GB of memory is required: the pinned OmniAID code runs
  on `cuda`, and CPU-only machines are not supported. It needs about 4 GB of free disk for the checkpoint, and has not
  been tested outside Colab. If you try it:
  ```
  python -m venv venv && source venv/bin/activate
  pip install torch torchvision          # CUDA build matching your GPU
  pip install -r requirements.txt
  cp /path/to/adapted_epoch_1.pt models/
  python app.py                          # then open http://127.0.0.1:7860
  ```
- **Tests** (no GPU needed): `pip install pytest pillow pandas && python -m pytest tests`

## Limitations
- **Scope of study:** seven North American species, two authentic sources (Caltech Camera Traps, iNaturalist) and two
  generators (SDXL, FLUX.1-schnell). Other species, regions, cameras, generators, editing, compression or resizing may
  behave differently.
- **Held-out generator:** the wildlife adaptation improved Dataset v1 results (accuracy 99.85% vs 97.14%) but did
  **not** improve detection of the held-out generator.
  - FLUX fake recall was 92.33% (adapted) vs 92.94% (original); p = 0.81.
  - **15 of 326 FLUX images were classified as real by both detectors.**
- **Scores and thresholds:** scores are not calibrated, and the thresholds come from this project's validation data.
- **Near-threshold scores:** a score within about 1e-5 of a threshold can occasionally get the opposite decision from
  the research record, because of floating-point variation between GPU runs (documented in the verification summary).
- **Runtime:** requires a GPU. Through the Colab proxy the interface can take several seconds per action.
- **Log noise:** the console line "Found N images: Real images: N" is the official loader's internal folder label, not a
  prediction.

## Responsible use
This is a research demonstration. Do **not** use it as the sole basis for:
- fraud accusations;
- legal or disciplinary decisions;
- scientific-misconduct conclusions;
- automatic rejection of ecological evidence;
- definitive publication-authenticity decisions.

Interpret results together with image provenance, metadata, source context, human review and other forensic methods.

## Privacy
- **Uploads:** images are processed in temporary storage for the session and are not kept or logged by the app.
- **Metadata and exports:** GPS coordinates and personal EXIF fields are not read into results or exports.
- **Public link:** the app does not create a public link unless `HIT401_SHARE=1` is set.

## Licences and attribution
See `THIRD_PARTY_LICENSES.md`, `ATTRIBUTIONS.md`, `verification_images/ATTRIBUTION.csv` and `licenses/`.
