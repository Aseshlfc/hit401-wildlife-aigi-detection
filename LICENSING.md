# Licensing

This repository mixes our own project work with material from other sources. Each part keeps its own terms.

## Our own work (code, notebooks, documentation, manifests, results)

Written by HIT401 Group 24 (Ashesh Luitel, Mahezabeen Momtarin Arpi, Christian Jay Guarino Borja, Shahriar Khandakar
Seam) for the HIT401 capstone at Charles Darwin University. **The group has not yet chosen an open-source licence**, so
no `LICENSE` file is included and normal copyright applies. The repository is public so that the work can be assessed
and checked.

Recommendation for the group: the code we wrote does not contain copied upstream source code (OmniAID is cloned and
called at run time), so it could be released under the **MIT License**, and the documentation, manifests and result
tables under **CC BY 4.0**. This needs the agreement of all four members (and should be checked against any CDU rules on
student intellectual property) before a `LICENSE` file is added.

## Third-party material

| Material | Terms | In this repository? |
|---|---|---|
| OmniAID code (github.com/yunncheng/OmniAID) | README states MIT, but the repository has no licence file | No. Cloned at the pinned commit when the code runs |
| OmniAID v2 checkpoint (Hugging Face `Yunncheng/OmniAID`) | Model card says MIT; no licence file or copyright notice | No. Downloaded and checked by SHA-256 |
| CLIP ViT-L/14-336 weights (`openai/clip-vit-large-patch14-336`) | No licence declared on the model card | No |
| Our adapted parameter file `adapted_epoch_1.pt` | Derived from the OmniAID checkpoint (itself derived from CLIP); upstream terms not explicit | **No.** Hash and configuration documented; available to assessors from the team |
| Caltech Camera Traps (LILA BC) | CDLA-Permissive-1.0 | Image IDs and metadata in manifests; 2 verification images in `code/platform/verification_images/` with the licence text in `code/platform/licenses/` |
| iNaturalist photographs | Per image: CC0 or CC-BY (attribution required for CC-BY) | Observation/photo IDs, URLs, licence and attribution in `manifests/dataset_v1/inaturalist_candidate_manifest.csv`; 3 CC0 verification images bundled; no CC-BY image files are published |
| SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) | CreativeML Open RAIL++-M; use restrictions apply to outputs | Prompts and seeds; 3 generated verification images |
| FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`) | Apache-2.0 | Prompts and seeds; 4 generated verification images |
| Report figures in `figures/` | Our work; `fig_examples.png` shows two Caltech images (CDLA-Permissive, licence text in `code/platform/licenses/`) and two CC0 iNaturalist images (observations 2516178 and 35508) | Yes |
| Python packages | Their own licences (Apache-2.0, BSD, MIT and others) | No; installed with pip |

The generated images are labelled as AI-generated and are only for detector research; they must not be used to deceive
anyone. Details for the platform files are in `code/platform/THIRD_PARTY_LICENSES.md` and
`code/platform/verification_images/ATTRIBUTION.csv`.
