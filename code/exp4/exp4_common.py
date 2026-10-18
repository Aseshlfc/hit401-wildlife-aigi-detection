"""Experiment 4 (held-out-from-project-adaptation generator) - shared configuration.

Protocol v1.0 approved by the user on 2026-09-25 (see PROJECT_RECOVERY_STATE/EXPERIMENT_4_PLANNING.md).
Generator: FLUX.1-schnell, held out from project adaptation (Exp 3 used SDXL only).
It is NOT claimed to be unseen by OmniAID (the Flux.1 family is named in OmniAID's Mirage-Train).
"""
import json, random
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/HIT401_Wildlife_Deepfake")
EXP4 = DRIVE / "exp4_heldout_generator"
CODE = EXP4 / "code"
PILOT_DIR = EXP4 / "pilot"
CAND_DIR = EXP4 / "candidates"           # every generated image (accepted and rejected) - never deleted
FINAL_DIR = EXP4 / "final" / "1_fake"    # frozen, QC-accepted images only
META = EXP4 / "metadata"
SHEETS = EXP4 / "contact_sheets"
EVAL = EXP4 / "evaluation"

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"   # HF commit, lastModified 2024-08-16

# Official FLUX.1-schnell inference settings (model card / diffusers FluxPipeline example):
# timestep-distilled, 1-4 steps, no classifier-free guidance, T5 max_sequence_length 256, no negative prompt.
GEN = dict(num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256, height=768, width=768)
# Resolution 768x768 and JPEG quality 95 match the Dataset v1 SDXL images, so only the generator changes.
JPEG_QUALITY = 95
GENERATOR_DEVICE = "cpu"   # torch.Generator("cpu").manual_seed(seed), as in the official example (device-independent noise)

LOGIC = json.load(open(CODE / "dataset_v1_prompt_logic.json"))
ANIMALS = list(LOGIC["ANIMAL_NAMES"].items())          # 7 species, Dataset v1 order
STYLES = ["community", "camera_day", "camera_night_ir"]   # Dataset v1 order
assert STYLES == list(LOGIC["STYLE_TARGETS"])

QUOTA = {"community": 24, "camera_day": 12, "camera_night_ir": 12}   # per species -> 48; x7 = 336
MAIN_BASE_SEED = 600000    # Dataset v1 used 500000; pilot uses 690000 (pilot images are never part of the dataset)
PILOT_BASE_SEED = 690000


def seed_for(base, animal_index, style_index, n):
    """Same seed formula as Dataset v1 (only the base differs). n = 0, 1, 2, ... is the predefined sequence."""
    return base + animal_index * 10000 + style_index * 2000 + n


def prompt_for(category, style, seed):
    """Dataset v1 semantic prompt logic, reproduced exactly (deterministic per seed)."""
    desc = LOGIC["ANIMAL_NAMES"][category]
    rng = random.Random(seed)
    habitat = rng.choice(LOGIC["HABITATS"])
    pose = rng.choice(LOGIC["POSES"])
    distance = rng.choice(LOGIC["DISTANCES"])
    t = LOGIC["STYLE_TEMPLATES"][style]
    if style == "community":
        return t.format(animal=desc, pose=pose, habitat=habitat, lighting=rng.choice(LOGIC["COMMUNITY_LIGHTING"]), distance=distance)
    return t.format(animal=desc, pose=pose, habitat=habitat, distance=distance)


def image_id(category, style, seed):
    return f"flux_{category}_{style}_{seed}"


def cell_plan(base, n_values):
    """All (category, style, n, seed) combinations for the given n values, in Dataset v1 order."""
    out = []
    for ai, (cat, _) in enumerate(ANIMALS):
        for si, sty in enumerate(STYLES):
            for n in n_values(sty):
                out.append((cat, sty, n, seed_for(base, ai, si, n)))
    return out
