"""Experiment 4 - FLUX.1-schnell image generation (pilot / main / replacement).

Usage (Colab terminal, GPU runtime, HF_TOKEN set in the environment):
  python3 exp4_generate.py --mode pilot --strategy nf4_fp16
  python3 exp4_generate.py --mode main  --strategy <strategy chosen in pilot>
  python3 exp4_generate.py --mode replace --strategy <same>      # after visual QC, fills QC failures with next unused seeds

No detector is ever run by this script. Resumable: images already in the manifest are skipped.
"""
import argparse, csv, datetime, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exp4_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["pilot", "main", "replace", "determinism"], required=True)
ap.add_argument("--strategy", choices=["nf4_fp16", "nf4_bf16", "offload_bf16", "offload_fp16"], required=True)
ap.add_argument("--limit", type=int, default=0, help="stop after N new images (0 = no limit)")
a = ap.parse_args()

import numpy as np, torch
import diffusers, transformers
from PIL import Image

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
def log(msg, f=None):
    line = f"[{now()}] {msg}"; print(line, flush=True)
    with open(META / "generation_log.txt", "a") as fh: fh.write(line + "\n")

META.mkdir(parents=True, exist_ok=True)
dtype = torch.bfloat16 if a.strategy.endswith("bf16") else torch.float16
token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set (FLUX.1-schnell is gated on Hugging Face)"

# ---------------- load pipeline ----------------
t0 = time.time()
from diffusers import FluxPipeline
if a.strategy.startswith("nf4"):
    from diffusers import FluxTransformer2DModel, BitsAndBytesConfig as DiffBnb
    from transformers import T5EncoderModel, BitsAndBytesConfig as TfBnb
    import bitsandbytes
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", revision=MODEL_REVISION, token=token, torch_dtype=dtype,
        quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", revision=MODEL_REVISION, token=token, torch_dtype=dtype,
        quantization_config=TfBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))
    pipe = FluxPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token, torch_dtype=dtype,
                                        transformer=transformer, text_encoder_2=text_encoder_2)
    pipe.enable_model_cpu_offload()
    quant = "transformer + T5 text encoder: bitsandbytes 4-bit NF4 (weights), compute dtype " + str(dtype)
else:
    pipe = FluxPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token, torch_dtype=dtype)
    pipe.enable_sequential_cpu_offload()
    quant = "none (full weights, sequential CPU offload), dtype " + str(dtype)
pipe.set_progress_bar_config(disable=True)
load_s = time.time() - t0

def ver(m):
    try: return __import__(m).__version__
    except Exception: return "n/a"
env = {"recorded_utc": now(), "mode": a.mode, "strategy": a.strategy, "quantisation": quant,
       "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "generation_settings": GEN,
       "generator_device": GENERATOR_DEVICE, "jpeg_quality": JPEG_QUALITY, "negative_prompt": "none (FLUX.1-schnell has no negative prompt / CFG)",
       "torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
       "bitsandbytes": ver("bitsandbytes"), "accelerate": ver("accelerate"), "python": sys.version.split()[0],
       "gpu": torch.cuda.get_device_name(0), "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
       "pipeline_load_seconds": round(load_s, 1)}
with open(META / f"generation_env_{a.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json", "w") as fh: json.dump(env, fh, indent=1)
log(f"Pipeline loaded in {load_s:.0f}s | {quant} | torch {torch.__version__} diffusers {diffusers.__version__} transformers {transformers.__version__}")

def generate(cat, sty, seed):
    prompt = prompt_for(cat, sty, seed)
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(GENERATOR_DEVICE).manual_seed(seed)
    t = time.time()
    img = pipe(prompt=prompt, generator=g, **GEN).images[0]
    dt = time.time() - t
    arr = np.asarray(img)
    return img, prompt, dt, torch.cuda.max_memory_allocated() / 2**30, arr

FIELDS = ["image_id", "filename", "batch", "category", "style", "animal_description", "n", "seed", "prompt",
          "model_id", "model_revision", "strategy", "num_inference_steps", "guidance_scale", "max_sequence_length",
          "width", "height", "generator_device", "jpeg_quality", "pixel_mean", "pixel_std", "has_nan_or_black",
          "sha256_jpeg", "gen_seconds", "peak_vram_gib", "created_utc"]

def run(plan, out_dir, manifest_path, batch):
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest_path.exists():
        with open(manifest_path) as fh: done = {r["image_id"] for r in csv.DictReader(fh)}
    new = not manifest_path.exists()
    fh = open(manifest_path, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()
    made = 0
    for cat, sty, n, seed in plan:
        iid = image_id(cat, sty, seed)
        if iid in done: continue
        img, prompt, dt, vram, arr = generate(cat, sty, seed)
        fn = f"{iid}.jpg"; path = out_dir / fn
        img.save(path, format="JPEG", quality=JPEG_QUALITY)
        bad = bool(np.isnan(arr).any() or arr.std() < 2.0)
        w.writerow({"image_id": iid, "filename": fn, "batch": batch, "category": cat, "style": sty,
                    "animal_description": LOGIC["ANIMAL_NAMES"][cat], "n": n, "seed": seed, "prompt": prompt,
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "strategy": a.strategy,
                    "num_inference_steps": GEN["num_inference_steps"], "guidance_scale": GEN["guidance_scale"],
                    "max_sequence_length": GEN["max_sequence_length"], "width": GEN["width"], "height": GEN["height"],
                    "generator_device": GENERATOR_DEVICE, "jpeg_quality": JPEG_QUALITY,
                    "pixel_mean": round(float(arr.mean()), 2), "pixel_std": round(float(arr.std()), 2), "has_nan_or_black": bad,
                    "sha256_jpeg": hashlib.sha256(path.read_bytes()).hexdigest(), "gen_seconds": round(dt, 1),
                    "peak_vram_gib": round(vram, 2), "created_utc": now()})
        fh.flush(); os.fsync(fh.fileno()); made += 1
        log(f"{batch} {iid} {dt:.1f}s peakVRAM {vram:.2f}GiB std {arr.std():.1f}{' BLACK/NaN!' if bad else ''}")
        if a.limit and made >= a.limit: break
    fh.close()
    return made

if a.mode == "pilot":
    plan = cell_plan(PILOT_BASE_SEED, lambda s: [0])           # 7 species x 3 styles = 21
    made = run(plan, PILOT_DIR, META / "pilot_manifest.csv", "pilot")
    log(f"Pilot: {made} new images")

elif a.mode == "determinism":
    # regenerate two pilot seeds and compare pixels with the saved pilot images (decoded arrays, before JPEG)
    res = []
    for cat, sty, n, seed in cell_plan(PILOT_BASE_SEED, lambda s: [0])[:2]:
        _, _, _, _, arr1 = generate(cat, sty, seed)
        _, _, _, _, arr2 = generate(cat, sty, seed)
        saved = np.asarray(Image.open(PILOT_DIR / f"{image_id(cat, sty, seed)}.jpg").convert("RGB")).astype(int)
        d_rep = int(np.abs(arr1.astype(int) - arr2.astype(int)).max())
        d_saved = float(np.abs(saved - arr1.astype(int)).mean())
        res.append({"image_id": image_id(cat, sty, seed), "max_abs_pixel_diff_between_two_runs": d_rep,
                    "mean_abs_diff_vs_saved_jpeg": round(d_saved, 3)})
        log(f"determinism {image_id(cat, sty, seed)}: run-to-run max|diff|={d_rep}, vs saved JPEG mean|diff|={d_saved:.2f}")
    json.dump(res, open(META / "pilot_determinism.json", "w"), indent=1)

elif a.mode == "main":
    plan = cell_plan(MAIN_BASE_SEED, lambda s: range(QUOTA[s]))   # predefined seeds n = 0..quota-1
    made = run(plan, CAND_DIR, META / "generation_manifest.csv", "main")
    log(f"Main: {made} new images")

elif a.mode == "replace":
    # For each species x style cell: replacements needed = quota - accepted. Next unused seeds, in order.
    import pandas as pd
    man = pd.read_csv(META / "generation_manifest.csv")
    qc = pd.read_csv(META / "qc_decisions.csv")
    assert set(qc.image_id) == set(man.image_id), "Every generated image needs a QC decision before replacement"
    m = man.merge(qc[["image_id", "qc_decision"]], on="image_id")
    plan = []
    for ai, (cat, _) in enumerate(ANIMALS):
        for si, sty in enumerate(STYLES):
            c = m[(m.category == cat) & (m["style"] == sty)]
            # pending NEEDS_USER images are not replaced until the user has decided them
            need = QUOTA[sty] - int((c.qc_decision == "ACCEPT").sum()) - int((c.qc_decision == "NEEDS_USER").sum())
            nxt = int(c.n.max()) + 1
            cap = 5 * QUOTA[sty]   # user stop rule (QC_OPERATIONAL_NOTES.md, 2026-09-24)
            if need > 0 and len(c) + need > cap:
                # user decision 2026-09-25: generate up to the cap, never beyond; then STOP that cell and report
                allowed = max(cap - len(c), 0)
                log(f"STOP-RULE {cat}/{sty}: {len(c)} attempts, {int((c.qc_decision == 'ACCEPT').sum())} accepted, "
                    f"{need} still needed; generating only {allowed} more to reach cap {cap}; report to user")
                need = allowed
            for k in range(max(need, 0)):
                plan.append((cat, sty, nxt + k, seed_for(MAIN_BASE_SEED, ai, si, nxt + k)))
    log(f"Replacement plan: {len(plan)} images")
    made = run(plan, CAND_DIR, META / "generation_manifest.csv", "replacement")
    log(f"Replacements: {made} new images")
