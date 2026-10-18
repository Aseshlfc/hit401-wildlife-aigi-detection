# Attributions

## Models and code used at runtime (not bundled)
- **OmniAID**: Guo, Y., Ye, J., Zhang, C., Kang, H., Fu, H., He, C. and Li, W. (2026). *OmniAID: decoupling semantics
  and artifacts for universal AI-generated image detection in the wild*. ICML 2026, arXiv:2511.08423.
  Code: https://github.com/yunncheng/OmniAID (commit 40749406). Weights: https://huggingface.co/Yunncheng/OmniAID
  (revision 279cae73).
- **CLIP**: Radford, A. et al. (2021). *Learning transferable visual models from natural language supervision*.
  ICML 2021, PMLR 139, pp. 8748–8763. The ViT-L/14 @ 336 px configuration is fetched from
  https://huggingface.co/openai/clip-vit-large-patch14-336.
- **Gradio, Hugging Face transformers / huggingface_hub, timm, PyTorch, Pillow, pandas**: installed from PyPI under
  their own licences.

## Bundled verification images
The per-file details are in `verification_images/ATTRIBUTION.csv`.
- **Caltech Camera Traps** (2 images): Beery, S., Van Horn, G., Perona, P. (2018). *Recognition in Terra Incognita*.
  ECCV 2018. Distributed by LILA BC (https://lila.science/datasets/caltech-camera-traps) under the Community Data
  License Agreement – Permissive 1.0. The licence text is in `licenses/CDLA-Permissive-1.0.txt`. The files are
  unmodified.
- **iNaturalist** (3 photos, CC0, no rights reserved). Courtesy credit to the observers:
  - observation 1628425 / photo 1997958, by iNaturalist user *deejay*;
  - observation 3277526 / photo 3789805, by *samo_nps*;
  - observation 4310670 / photo 5178747, by *loarie*.
- **AI-generated images** (7), made by the HIT401 project:
  - 3 with Stable Diffusion XL base 1.0 (Stability AI; CreativeML Open RAIL++-M);
  - 4 with FLUX.1-schnell (Black Forest Labs; Apache-2.0).
  They are clearly labelled as AI-generated and are provided for detector-evaluation purposes only.
