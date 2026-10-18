"""Wildlife Image Authenticity Analysis Platform - HIT401 research prototype (Prototype v2).

Compares the ORIGINAL OmniAID v2 detector with the project's WILDLIFE-ADAPTED OmniAID v2 on one or many images.
Both models, both thresholds and the decision rule are frozen from the completed research (Experiments 1-4).
Run (GPU required):  python app.py        or use launcher/HIT401_Platform_Colab.ipynb
Uploaded images are processed in temporary storage only; the app does not keep or log them.
"""
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from batch_processing import analyse_files
from config import MAX_BATCH, SELF_CHECK_TOLERANCE, T_ADAPTED, T_ORIGINAL, VERIFY_IMAGES_DIR
from exports import summarise, write_csv, write_summary
from metadata import METADATA_NOTE, human_size, make_preview, orientation_note
from status import LIKELY_AI, LIKELY_AUTHENTIC, REVIEW, STATUS_EXPLANATION

TITLE = "Wildlife Image Authenticity Analysis Platform"
SUBTITLE = "Research prototype for comparing original and wildlife-adapted AI-generated-image detectors."
FILTERS = {"All": None, "Likely Authentic": LIKELY_AUTHENTIC, "Likely AI-Generated": LIKELY_AI, "Review Suggested": REVIEW}
WORK_DIR = Path(tempfile.mkdtemp(prefix="hit401_platform_"))  # previews and exports for this app session only

RESPONSIBLE_USE = (
    "**Research prototype.** Results come from two research detectors and are not forensic proof. Do not use them as the "
    "sole basis for fraud accusations, legal, disciplinary or scientific-misconduct decisions, automatic rejection of "
    "ecological evidence, or publication-authenticity decisions. Interpret them together with provenance, metadata, "
    "source context, human review and other forensic methods.")

CSS = """
.status-badge {display:inline-block; padding:6px 12px; border-radius:4px; font-weight:600; letter-spacing:.02em; border:1px solid;}
.st-auth {background:#eef6f0; color:#1d5b33; border-color:#9cc7a9;}
.st-ai {background:#f9eeee; color:#7a1f1f; border-color:#d9a3a3;}
.st-review {background:#fdf6e7; color:#6b4a00; border-color:#e2c27a;}
.st-na {background:#f1f1f1; color:#444; border-color:#c8c8c8;}
.stat-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px;}
.stat {border:1px solid #d0d4da; border-radius:4px; padding:8px 10px; background:#fafbfc;}
.stat .v {font-size:1.45em; font-weight:600; color:#1f2933;}
.stat .l {font-size:.82em; color:#52606d;}
.muted {color:#52606d; font-size:.9em;}
#gallery .grid-container, #rgallery .grid-container {grid-template-columns:repeat(auto-fill, minmax(150px, 190px)) !important;}
.caption-label {white-space:normal !important; font-size:11px !important; line-height:1.25 !important; max-width:95% !important;}
"""
DET_WIDTHS = ["36%", "19%", "23%", "22%"]
_BADGE = {LIKELY_AUTHENTIC: "st-auth", LIKELY_AI: "st-ai", REVIEW: "st-review"}


# ---------------------------------------------------------------- rendering helpers
def badge(status):
    return f'<span class="status-badge {_BADGE.get(status, "st-na")}">{status}</span>'


def stats_html(records, elapsed):
    if not records:
        return '<p class="muted">No images analysed yet.</p>'
    s = summarise(records, elapsed)
    cells = [("Submitted", s["total_submitted"]), ("Analysed", s["analysed"]), ("Likely authentic", s["likely_authentic"]),
             ("Likely AI-generated", s["likely_ai_generated"]), ("Review suggested", s["review_suggested"]),
             ("Failed / unsupported", s["failed_or_unsupported"]),
             ("Detector agreement", f"{s['agreement_rate']:.0%}" if s["agreement_rate"] is not None else "n/a"),
             ("Processing time", f"{elapsed:.1f} s" if elapsed else "n/a")]
    body = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{k}</div></div>' for k, v in cells)
    return (f'<div class="stat-grid">{body}</div><p class="muted">Counts describe this batch only; they are not accuracy '
            'figures or research results. Processing time is application timing only.</p>')


def results_table(records, status=None):
    rows = []
    for r in records:
        if status and r["platform_status"] != status:
            continue
        ok = r["processing_status"] == "ANALYSED"
        rows.append({"File": r["filename"], "Status": r["platform_status"] if ok else r["processing_status"],
                     "Original score": f"{r['original_score']:.4f}" if ok else "", "Original decision": r["original_decision"],
                     "Adapted score": f"{r['adapted_score']:.4f}" if ok else "", "Adapted decision": r["adapted_decision"],
                     "Agree": ("Yes" if r["models_agree"] else "No") if ok else "",
                     "Dimensions": r["dimensions"], "Format": r["format"], "Size": human_size(r["file_size_bytes"]),
                     "EXIF": "Yes" if r["exif_present"] else "No", "Note": r["processing_note"]})
    cols = ["File", "Status", "Original score", "Original decision", "Adapted score", "Adapted decision", "Agree",
            "Dimensions", "Format", "Size", "EXIF", "Note"]
    return pd.DataFrame(rows, columns=cols)


def metadata_table(records):
    rows = [{"File": r["filename"], "Format": r["format"], "Dimensions": r["dimensions"], "Size": human_size(r["file_size_bytes"]),
             "Colour mode": r["colour_mode"], "EXIF present": "Yes" if r["exif_present"] else "No",
             "Camera make": r["camera_make"], "Camera model": r["camera_model"], "Capture date/time": r["capture_datetime"],
             "ISO": r["iso"], "Exposure": r["exposure_time"], "Aperture": r["f_number"], "Focal length": r["focal_length"],
             "Software": r["software"], "GPS metadata present": "Yes" if r["gps_metadata_present"] else "No",
             "Other embedded metadata": r["other_embedded_metadata"]} for r in records]
    return pd.DataFrame(rows, columns=["File", "Format", "Dimensions", "Size", "Colour mode", "EXIF present", "Camera make",
                                       "Camera model", "Capture date/time", "ISO", "Exposure", "Aperture", "Focal length",
                                       "Software", "GPS metadata present", "Other embedded metadata"])


def gallery_items(records, status=None):
    idx = [i for i, r in enumerate(records) if r.get("preview") and (status is None or r["platform_status"] == status)]
    return [(records[i]["preview"], f"{records[i]['platform_status']} | {records[i]['filename']}") for i in idx], idx


def detail(records, i):
    """Outputs for the detail panel: preview, heading+status, detector table, explanation, metadata table."""
    if records is None or i is None or i >= len(records):
        return None, "Select an image to see its detailed results.", None, "", None
    r = records[i]
    head = f"#### {r['filename']}\n"
    if r["processing_status"] != "ANALYSED":
        return (r.get("preview"), head + badge(r["processing_status"]), None,
                f"This file was not analysed: {r['processing_note']}", None)
    det = pd.DataFrame([["Original OmniAID v2 (Experiment 1)", f"{r['original_score']:.6f}", f"{T_ORIGINAL:.6f}", r["original_decision"]],
                        ["Wildlife-adapted OmniAID v2 (Experiment 3)", f"{r['adapted_score']:.6f}", f"{T_ADAPTED:.6f}", r["adapted_decision"]]],
                       columns=["Detector", "Model score", "Frozen threshold", "Decision"])
    agree = "The two detectors **agree**. " if r["models_agree"] else ""
    expl = f"{agree}{STATUS_EXPLANATION[r['platform_status']]}"
    if r["processing_note"]:
        expl += f"\n\n{r['processing_note']}"
    on = orientation_note(r)
    if on:
        expl += f"\n\n{on}"
    md = pd.DataFrame([["File type", r["format"]], ["Dimensions", r["dimensions"]], ["File size", human_size(r["file_size_bytes"])],
                       ["Colour mode", r["colour_mode"]], ["EXIF present", "Yes" if r["exif_present"] else "No"],
                       ["Camera make / model", " ".join(x for x in (r["camera_make"], r["camera_model"]) if x)],
                       ["Capture date/time", r["capture_datetime"]], ["ISO", r["iso"]], ["Exposure", r["exposure_time"]],
                       ["Aperture", r["f_number"]], ["Focal length", r["focal_length"]], ["Software", r["software"]],
                       ["GPS metadata present", "Yes" if r["gps_metadata_present"] else "No"],
                       ["Other embedded metadata", r["other_embedded_metadata"] or "None found"]], columns=["Field", "Value"])
    return r.get("preview"), head + badge(r["platform_status"]), det, expl, md


# ---------------------------------------------------------------- UI
def build_ui(detectors, startup_text):
    def run(files, progress=gr.Progress()):
        if not files:
            raise gr.Error("Upload at least one image first.")
        if len(files) > MAX_BATCH:
            raise gr.Error(f"Please upload at most {MAX_BATCH} files at a time.")
        pairs = [(f, Path(f).name) for f in files]
        records, elapsed = analyse_files(detectors, pairs, progress=progress)
        pdir = Path(tempfile.mkdtemp(dir=WORK_DIR))
        for r in records:
            try:
                r["preview"] = make_preview(r["path"], pdir) if r["processing_status"] == "ANALYSED" else None
            except Exception:
                r["preview"] = None
        return refresh(records, elapsed, "All")

    def run_bundled(progress=gr.Progress()):
        files = sorted(str(p) for p in VERIFY_IMAGES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        return run(files, progress)

    def refresh(records, elapsed, filt):
        items, idx = gallery_items(records, FILTERS[filt])
        ritems, ridx = gallery_items(records, REVIEW)
        first = idx[0] if idx else (0 if records else None)
        d = detail(records, first)
        return (records, elapsed, stats_html(records, elapsed), gr.update(value=filt), items, idx, *d,
                stats_html(records, elapsed), results_table(records), ritems, ridx,
                results_table(records, REVIEW), metadata_table(records), None, None)

    def on_filter(records, filt):
        items, idx = gallery_items(records or [], FILTERS[filt])
        return items, idx, *detail(records, idx[0] if idx else None)

    def on_select(records, idx, evt: gr.SelectData):
        return detail(records, idx[evt.index] if idx and evt.index < len(idx) else None)

    def table_filter(records, filt):
        return results_table(records or [], FILTERS[filt])

    def export_csv(records):
        if not records:
            raise gr.Error("Nothing to export yet.")
        d = Path(tempfile.mkdtemp(dir=WORK_DIR))
        return write_csv(records, d / "wildlife_authenticity_results.csv")

    def export_summary(records, elapsed):
        if not records:
            raise gr.Error("Nothing to export yet.")
        d = Path(tempfile.mkdtemp(dir=WORK_DIR))
        return write_summary(records, d / "wildlife_authenticity_summary.txt", elapsed)

    theme = gr.themes.Default(primary_hue="slate", neutral_hue="slate", font=["Helvetica Neue", "Arial", "sans-serif"])
    with gr.Blocks(title=TITLE, theme=theme, css=CSS) as demo:
        records_st, elapsed_st = gr.State([]), gr.State(None)
        gidx_st, ridx_st = gr.State([]), gr.State([])
        gr.Markdown(f"# {TITLE}\n{SUBTITLE}")
        gr.Markdown(RESPONSIBLE_USE)
        gr.Markdown(f'<p class="muted">{startup_text}</p>')

        with gr.Tabs():
            with gr.Tab("Analyse Images"):
                with gr.Row():
                    with gr.Column(scale=2):
                        files_in = gr.File(label="Wildlife images (JPEG, PNG or WebP) - one or many", file_count="multiple", height=170)
                        with gr.Row():
                            go = gr.Button("Analyse", variant="primary")
                            demo_btn = gr.Button("Analyse bundled verification images")
                    with gr.Column(scale=3):
                        stats1 = gr.HTML(stats_html([], None))
                filt = gr.Radio(list(FILTERS), value="All", label="Show")
                gallery = gr.Gallery(label="Analysed images (click one for details)", columns=6, height=300, elem_id="gallery",
                                     allow_preview=False, object_fit="cover")
                with gr.Row():
                    with gr.Column(scale=2):
                        d_img = gr.Image(label="Image preview (stored pixel orientation, as analysed)", height=380, interactive=False)
                    with gr.Column(scale=3):
                        d_head = gr.Markdown("Select an image to see its detailed results.")
                        d_det = gr.Dataframe(label="Frozen detectors", interactive=False, wrap=True, column_widths=DET_WIDTHS)
                        d_expl = gr.Markdown()
                        with gr.Accordion("File metadata", open=False):
                            d_md = gr.Dataframe(interactive=False, wrap=True)
                            gr.Markdown(f'<p class="muted">{METADATA_NOTE} Precise GPS coordinates are never shown or exported.</p>')

            with gr.Tab("Batch Dashboard"):
                stats2 = gr.HTML(stats_html([], None))
                tfilt = gr.Radio(list(FILTERS), value="All", label="Show")
                table = gr.Dataframe(label="Results", interactive=False, wrap=False, max_height=520)
                with gr.Row():
                    csv_btn = gr.Button("Export results (CSV)")
                    sum_btn = gr.Button("Export summary (text)")
                with gr.Row():
                    csv_out = gr.File(label="CSV file", interactive=False)
                    sum_out = gr.File(label="Summary file", interactive=False)
                gr.Markdown('<p class="muted">The CSV contains technical file information, metadata presence, both detectors\' '
                            'scores, thresholds and decisions, agreement and platform status. It contains no GPS coordinates, '
                            'no owner/serial fields and no file paths.</p>')

            with gr.Tab("Review Suggested"):
                gr.Markdown("Images where the **original** and the **wildlife-adapted** detector disagree. Disagreement means "
                            "only that human inspection is recommended; it is not a calibrated uncertainty estimate. "
                            "Check provenance, source and context for these images first.")
                rgallery = gr.Gallery(label="Review suggested (click one for details)", columns=6, height=260, elem_id="rgallery",
                                      allow_preview=False, object_fit="cover")
                with gr.Row():
                    with gr.Column(scale=2):
                        r_img = gr.Image(label="Image preview", height=340, interactive=False)
                    with gr.Column(scale=3):
                        r_head = gr.Markdown()
                        r_det = gr.Dataframe(label="Frozen detectors", interactive=False, wrap=True, column_widths=DET_WIDTHS)
                        r_expl = gr.Markdown()
                        r_md = gr.Dataframe(label="File metadata", interactive=False, wrap=True, visible=False)
                rtable = gr.Dataframe(label="Review-suggested images", interactive=False, wrap=False, max_height=520)

            with gr.Tab("File Metadata"):
                gr.Markdown(f"{METADATA_NOTE}\n\nOnly technical fields are shown. GPS is reported as present or absent; "
                            "coordinates are never displayed or exported. Owner, artist, copyright and serial-number fields are not read.")
                mtable = gr.Dataframe(label="Metadata for all submitted files", interactive=False, wrap=False, max_height=520)

            with gr.Tab("About / Methodology"):
                gr.Markdown(ABOUT)

        detail_out = [d_img, d_head, d_det, d_expl, d_md]
        outs = [records_st, elapsed_st, stats1, filt, gallery, gidx_st, *detail_out, stats2, table, rgallery, ridx_st,
                rtable, mtable, csv_out, sum_out]
        go.click(run, inputs=files_in, outputs=outs)
        demo_btn.click(run_bundled, inputs=None, outputs=outs)
        filt.input(on_filter, inputs=[records_st, filt], outputs=[gallery, gidx_st, *detail_out])
        gallery.select(on_select, inputs=[records_st, gidx_st], outputs=detail_out)
        rgallery.select(on_select, inputs=[records_st, ridx_st], outputs=[r_img, r_head, r_det, r_expl, r_md])
        tfilt.input(table_filter, inputs=[records_st, tfilt], outputs=table)
        csv_btn.click(export_csv, inputs=records_st, outputs=csv_out)
        sum_btn.click(export_summary, inputs=[records_st, elapsed_st], outputs=sum_out)
    return demo


ABOUT = f"""
### What this platform does
For every uploaded image it runs **two frozen research detectors** and reports each detector's *model score*, its
*frozen threshold* and its *decision*, then combines the two decisions into a platform status. It supports single images
and batches, with a dashboard, a review queue, file-metadata context and CSV export.

### The two detectors
- **Original detector:** the official OmniAID v2 checkpoint (a mixture-of-experts detector built on a CLIP ViT-L/14
  backbone at 336 px), used off the shelf. Its threshold **{T_ORIGINAL}** was chosen on the project's validation split
  (Youden's J) in Experiment 1.
- **Wildlife-adapted detector:** the same model after the project's *parameter-efficient fine-tuning of OmniAID v2*
  (Experiment 3). Only about 1.45 million of 507.8 million parameters were trained (the low-rank experts, the router
  and the output layer), on the Dataset v1 training split: Caltech Camera Traps and iNaturalist photographs of seven
  North American species (opossum, raccoon, rabbit, coyote, bobcat, squirrel, skunk) and SDXL images in matching
  styles. Epoch and threshold (**{T_ADAPTED}**) were selected on validation data only.
- Both models, both thresholds and the decision rule (*score >= threshold -> AI-GENERATED*) are **frozen** from the
  completed research. The platform does not train, tune or re-select anything. At start-up it checks the SHA-256 hashes of
  the model files and re-scores bundled verification images; it refuses to run if they do not reproduce their saved
  research scores within {SELF_CHECK_TOLERANCE:g}.

### Model score is not confidence
The model score is the detector's raw output between 0 and 1 (higher = more AI-like). It is **not a calibrated
probability** and not a measure of certainty. The two detectors use different thresholds, so their scores are not
directly comparable with each other.

### Platform status
| Original decision | Adapted decision | Platform status |
|---|---|---|
| REAL | REAL | **LIKELY AUTHENTIC**: both research detectors classified the image as real. This is not proof of authenticity. |
| AI-GENERATED | AI-GENERATED | **LIKELY AI-GENERATED**: both detectors classified it as synthetic. This is not forensic proof. |
| different | different | **REVIEW SUGGESTED**: the detectors disagree, so human inspection is recommended. |

"Review suggested" means only that the two research detectors disagree. It is not calibrated uncertainty, and there is
no score-distance "uncertainty band".

### What the research found (completed Experiments 1-4)
- On the frozen Dataset v1 test split (664 images, SDXL fakes), the adapted detector was more accurate than the original
  (accuracy 99.85% vs 97.14%; accuracy on authentic iNaturalist photographs 99.43% vs 89.77%).
- On synthetic images from a generator **held out from project adaptation** (326 FLUX.1-schnell images, Experiment 4),
  the adaptation did **not** improve detection: fake recall was 92.33% for the adapted detector vs 92.94% for the
  original (exact McNemar p = 0.81). The original detector missed 23 (7.1%) and the adapted detector 25 (7.7%) of these images, and **15 of the 326 FLUX images were
  classified as real by both detectors**, which this platform would show as "LIKELY AUTHENTIC".

### What the preprocessing does
The detectors use the official OmniAID evaluation transform: the file is read with Pillow and converted to RGB (alpha is
dropped; animated files use the first frame), resized to 336 x 336 without keeping the aspect ratio, and normalised with
the CLIP mean and standard deviation. EXIF orientation is **not** applied, so the preview shows the image in its stored
orientation, exactly as the detectors see it. Uploaded files are never re-encoded before scoring.

### File metadata
EXIF and other embedded metadata are shown as context only. Metadata can be modified, removed, copied or fabricated,
many websites strip it on upload, and AI-generated images can carry camera-like metadata. Metadata is therefore never
used in the decision. GPS is reported only as present or absent.

### Main limitations
- Studied only on seven North American species, two authentic sources (Caltech Camera Traps, iNaturalist) and two
  generators (SDXL, FLUX.1-schnell). Other species, regions, cameras, generators, edits, compression or resizing may
  behave differently.
- The wildlife adaptation did not improve detection of the held-out generator.
- Scores are not calibrated; thresholds were chosen on this project's validation data.
- Batch counts shown here describe the uploaded images only; they are not accuracy measurements.

### Responsible use
This is a research demonstration from a university capstone project (HIT401, Charles Darwin University). It must not be
the sole basis for fraud accusations, legal or disciplinary decisions, scientific-misconduct conclusions, automatic
rejection of ecological evidence, or definitive publication-authenticity decisions. Interpret results together with
image provenance, metadata, source context, human review and other forensic methods.
"""


def main():
    from inference import Detectors
    print("Loading both frozen detectors (first start downloads the 3.24 GB official checkpoint and hashes the model files)...", flush=True)
    det = Detectors(verify_hashes=os.environ.get("HIT401_SKIP_HASH") != "1")
    check = det.self_check()
    print("Self-check on bundled verification images:", {k: v for k, v in check.items() if k != "per_image"}, flush=True)
    if not check["passed"]:
        raise SystemExit("Frozen models did not reproduce their saved research scores - the platform will not start.")
    base = "verified" if det.base_hash_verified else "NOT re-hashed this session (HIT401_SKIP_HASH=1)"
    startup = (f"Start-up check passed: model-file hashes checked (base checkpoint {base}); {check['images']} verification images "
               f"reproduced their saved research scores (max difference {check['max_abs_diff']:.1e}, no decision changes).")
    demo = build_ui(det, startup)
    demo.queue(default_concurrency_limit=1)  # one GPU job at a time
    # Behind the Colab proxy Gradio cannot infer its public URL; the launcher sets HIT401_ROOT_URL.
    demo.launch(share=os.environ.get("HIT401_SHARE") == "1", debug=False, root_path=os.environ.get("HIT401_ROOT_URL") or None,
                allowed_paths=[str(WORK_DIR), str(VERIFY_IMAGES_DIR)], server_port=int(os.environ.get("HIT401_PORT", "7860")))
    return det


if __name__ == "__main__":
    DET = main()
