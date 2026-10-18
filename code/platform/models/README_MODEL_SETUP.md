# Model setup

The platform needs three model-related files. Only the small project config is bundled.

| File | Where it comes from | Bundled? | Integrity check |
|---|---|---|---|
| `config_omniaid_v2_recovered.json` | Project file: the architecture settings used in Experiments 3 and 4 and in Prototype v1 | **Yes** (this folder) | SHA-256 `213ae1d675f490812c5b6cba202d3894978c7b7cd1bd67c4c9a3a2a1f5d6ff4a`, checked at every start |
| `ckpt/checkpoint_omniaid_v2.pth` (official OmniAID v2 base checkpoint, 3.24 GB) | Hugging Face `Yunncheng/OmniAID`, revision `279cae7398ac6636f46fc4668f755f11210b36bf` | **No**. Downloaded automatically on first start | SHA-256 `919338928cab88cef1e6e4b85777749c74bdff6a1796fcd7e43a91422327f906`, checked at start |
| `adapted_epoch_1.pt` (the project's frozen Experiment 3 parameter file = selected epoch 2, 6,357,297 bytes) | The HIT401 project team, **privately** | **No** (see below) | SHA-256 `733ab455a3d1f677451c9368846eb44e36236d492571d7960d075da641a3cd66`, checked at every start |

The official OmniAID code is also downloaded on first start: `git clone https://github.com/yunncheng/OmniAID.git`, pinned
to commit `40749406fbcd8893c11a160edf4a72a2d4dc7056`. The platform checks that the checkout is exactly that commit
and unmodified. At model construction, the OmniAID code also fetches the configuration of
`openai/clip-vit-large-patch14-336` from Hugging Face. All 2,804 model weights are then replaced by the base checkpoint.

## 1. Base checkpoint: automatic
- **Nothing to do.** On first start the platform downloads the exact revision above from the official Hugging Face
  repository and verifies its SHA-256.
- **No login needed.** No Hugging Face account, token or terms acceptance is needed: the repository is public and
  not gated. This was confirmed in the verified Colab run of 2026-09-29, where the download succeeded while the session
  was explicitly not authenticated.
- **Colab secret prompt:** if Colab asks whether the notebook may access an `HF_TOKEN` secret, you can click
  **Cancel**.
- **Offline use:** set `HIT401_BASE_CKPT=/path/to/checkpoint_omniaid_v2.pth` to use a local copy of the same file. It is
  still hash-checked.

## 2. Adapted parameter file: provided privately
`adapted_epoch_1.pt` is the frozen output of the project's parameter-efficient fine-tuning of OmniAID v2 (Experiment 3).
It contains 1,734 tensors (1,445,959 trained parameters: the low-rank experts, the router and the output layer) that
are applied on top of the official checkpoint.

**Why it is not in the ZIP.** The official OmniAID weights are labelled "MIT" only in repository metadata (the Hugging
Face model card tag and a README badge). Neither the Hugging Face repository nor the GitHub repository contains a
licence file or copyright notice, and the CLIP backbone's Hugging Face repository declares no licence at all. Because
the redistribution terms for weights derived from these models are therefore not fully explicit, the project chose
the conservative option: the derived file is **not published** in the distributable package. See
`../THIRD_PARTY_LICENSES.md`.

**How to obtain it.** Ask the HIT401 project team. They share the single file privately with the reviewer, for
example as a Google Drive file shared with the reviewer's account. The file must not be posted publicly.

**Where to put it.**
- **Colab launcher:** leave `ADAPTED_PATH` empty and upload the file when the upload button appears, or give its path in
  your mounted Google Drive.
- **Local Python:** copy it to `models/adapted_epoch_1.pt`, or set `HIT401_ADAPTED_FILE=/path/to/adapted_epoch_1.pt`.

The platform refuses to start if the file's SHA-256 differs from `733ab455…cd66`.

## Environment variables (optional)
| Variable | Default | Purpose |
|---|---|---|
| `HIT401_ADAPTED_FILE` | `models/adapted_epoch_1.pt` | Adapted parameter file |
| `HIT401_BASE_CKPT` | (download from Hugging Face) | Local copy of the base checkpoint |
| `HIT401_MODELS_DIR` | `models/` | Folder holding the config (and, by default, the adapted file) |
| `HIT401_OMNIAID_REPO` | `_omniaid_repo/` | Where the pinned OmniAID code is cloned |
| `HIT401_SKIP_HASH` | unset | `1` skips re-hashing the 3.24 GB base checkpoint (the small files are always hashed) |
