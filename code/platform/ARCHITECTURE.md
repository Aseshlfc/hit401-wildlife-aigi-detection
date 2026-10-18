# Architecture: Wildlife Image Authenticity Analysis Platform (Prototype v2)

Prototype v2 is a **software layer around the frozen research models**. It adds no model, threshold or training step.

```
Uploaded image(s)  (one or many; JPEG / PNG / WebP)
        │
        ▼
File validation + metadata extraction ─────────────── metadata.py
  • extension must be one the official loader reads (.jpg .jpeg .png .webp)
  • strict full decode with Pillow (corrupt / truncated → FAILED; other types → UNSUPPORTED)
  • safe metadata: type, size, dimensions, colour mode, EXIF fields, GPS present yes/no (never coordinates)
  • the uploaded file is never modified or re-encoded
        │ valid files only
        ▼
Verified preprocessing + scoring (unchanged Prototype v1 pathway) ── inference.py
  • official OmniAID code @ 40749406, official GenerativeImageDataset eval transform:
    RGB → Resize 336×336 → ToTensor → CLIP normalisation (no EXIF rotation)
  • both models loaded ONCE at start-up, eval mode, fp32, reused for every image
        │
        ├──────────────► Original frozen detector   (OmniAID v2 checkpoint)           threshold 0.35996127128601074
        └──────────────► Adapted frozen detector    (+ Exp 3 parameters, epoch 2)     threshold 0.5848991870880127
        │
        ▼
Per-model scores and decisions ───────────────────── status.py
  score ≥ frozen threshold → AI-GENERATED, otherwise REAL
        │
        ▼
Agreement logic
  both REAL          → LIKELY AUTHENTIC      (not proof of authenticity)
  both AI-GENERATED  → LIKELY AI-GENERATED   (not forensic proof)
  disagreement       → REVIEW SUGGESTED      (human inspection recommended; not calibrated uncertainty)
        │
        ▼
Dashboard / gallery / review queue / metadata / CSV + text export ── app.py, exports.py, batch_processing.py
```

## Start-up integrity gate (the app will not start unless all of these pass)
1. The OmniAID checkout is exactly commit 40749406 and unmodified.
2. SHA-256 of the adapted parameter file (always) and of the architecture config (always), plus the 3.24 GB base
   checkpoint (unless `HIT401_SKIP_HASH=1`).
3. Self-check: the 12 bundled verification images are re-scored. Each score must match its saved research score
   within 1e-4 (the same tolerance as Prototype v1 and Experiment 4), and no decision may change.

## Modules
| File | Responsibility |
|---|---|
| `config.py` | Portable paths (env-overridable) and the frozen constants |
| `inference.py` | Copy of the verified Prototype v1 loader; only paths changed (see `verification/LOADER_CHANGES.md`) |
| `metadata.py` | Upload validation, privacy-safe metadata, display previews in stored orientation |
| `status.py` | Decision rule and agreement → platform status (pure functions) |
| `batch_processing.py` | One record per submitted file; valid files scored in chunks of 16 via `score_files`; checks every file got its own two scores |
| `exports.py` | CSV (fixed columns, no GPS or paths, formula-injection safe) and text summary |
| `app.py` | Gradio interface: Analyse Images, Batch Dashboard, Review Suggested, File Metadata, About / Methodology |
| `launcher/HIT401_Platform_Colab.ipynb` | Colab launcher (GPU check, package, packages, adapted file, start, optional verification) |
| `verification/verify_v2.py` | Software verification run against saved research evidence |
| `tests/` | 31 model-free unit tests (status rules, file handling, metadata privacy, batch bookkeeping, CSV, loader integrity) |

## Design decisions
- **Reuse over rewrite.** `score_files` and the model construction are v1's code. A unit test proves this with Python's
  `ast` module against a copy of the verified v1 file.
- **Batching.** Scores do not depend on batch size beyond about 3e-6: verified batch-of-12 against single-image
  scoring (3.2e-6) and against saved research scores (3.6e-6).
- **No uncertainty band.** Prototype v1's "within 0.10 of the threshold" note was removed because it had no
  validated basis.
- **Previews are display copies.** They show the stored pixel orientation, exactly what the model sees, and are never
  used for scoring.
- **Privacy.** GPS coordinates, owner, artist, copyright and serial-number fields are never read into results or
  exports. Uploads live only in temporary storage for the session.
