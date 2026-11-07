# Image2USD: Scalable 3D Scene Reconstruction from a Single Image via Structured Scene Tokenization

**Atandra Mahalder, Chase Walker, Sumit Kumar Jha, Rickard Ewetz**
University of Florida

Accepted at **WACV 2027**.

[Paper](#citation) · [Appendix](https://github.com/<GITHUB_USER>/Image2USD/releases) · [3DR-Set assets (Zenodo)](https://zenodo.org/records/22151734?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijg2OWU0NjhhLTBkZDUtNGE4MS1iNDQ5LTgyMWNkMTAxYTg5OSIsImRhdGEiOnt9LCJyYW5kb20iOiIyY2ZhNmE5OTVjZTNkM2U4MTc2NGFhNTk4YzYxNmY2NSJ9.Jdt5eQ3YDw6z5KrkK55_q8l4qGTWjM-fNfsIkit2h3WW7MR2QnJas-EGtX8m9RzgVu37jhQqw1wmPr92wjmB0Q)

<!-- TODO at release: replace the Appendix link with the uploaded Release asset URL. -->

![Image2USD overview](docs/overview.png)

Image2USD translates a **single RGB image** into a complete **Universal Scene Description (USD)** file — object layout, camera pose, and scene lighting — drawn from a predefined asset pack. A Gemma-4 vision-language model, fine-tuned with LoRA, emits a compact token encoding of the scene; a ResNet-50 CNN predicts illumination; grammar-constrained decoding guarantees syntactically valid output. The result renders directly in NVIDIA Isaac Sim and scales to scenes of **up to 80 objects**, where prior single-image methods stall at 7–8.

## Highlights

- **Compact USD encoding** — prim types and numeric values are mapped to custom tokens (FP15 floating-point scheme), reducing tokens per prim from 209 to 13 (16× shorter sequences).
- **Object prim sorting** — prims are ordered by scene depth so every scene maps to a unique target sequence.
- **Grammar-constrained decoding** — an EBNF grammar (`fp15_grammar.ebnf`) enforces USD structure at inference time.
- **Dedicated light estimation** — a ResNet-50 regresses light color and intensity, which VLMs fail to predict.
- **3DR-Set** — a procedurally generated dataset of 96,000 training and 9,600 evaluation scenes spanning 1–80 objects, 3 backgrounds (two warehouses, one residential room), 4 camera configurations, and a 48-asset pack, rendered with path tracing in Isaac Sim.

### Results on 3DR-Set

| Method | MSE ↓ | LPIPS ↓ | Chamfer ↓ | F-Score ↑ | Time (s) ↓ | Tokens ↓ |
|---|---|---|---|---|---|---|
| MIDI | 0.0327 | 0.4179 | 0.100 | 0.978 | 459 | – |
| SceneGen | 0.0181 | 0.2712 | 0.100 | 0.979 | 275 | – |
| Gen3DSR | 0.0099 | 0.1982 | 0.045 | 0.989 | 307 | – |
| IG-LLM | 0.0318 | 0.3528 | 1.441 | 0.973 | 575 | 2062 |
| DRP (detection + retrieval + pose) | 0.0121 | 0.1723 | 0.008 | 0.995 | <1 | – |
| **Image2USD (ours)** | **0.0019** | **0.0370** | **0.008** | **0.996** | 138 | **591** |

Image2USD improves on the diffusion and VLM baselines by 87.09% in LPIPS and 91.42% in Chamfer distance on average, and on the task-specific DRP baseline by 78.53% in LPIPS. Full details, ablations (prim sorting, grammar-constrained decoding, encoding schemes), object-level metrics, and out-of-distribution experiments are in the paper and appendix.

## Repository layout

```
.
├── create_3dr_set.py              # end-to-end 3DR-Set generation driver
├── generate_train_data.py         # procedural scene sampler (training split)
├── generate_test_data.py          # procedural scene sampler (evaluation split, semantic constraints)
├── render_train_images.py         # Isaac Sim path-traced rendering of generated scenes
├── render_test_images.py          # Isaac Sim rendering of Image2USD predictions
├── prepare_data.py                # builds the HF dataset (sorted + encoded prims) for finetuning
├── finetune_gemma.py              # LoRA finetuning of the Gemma-4 VLM
├── finetune_resnet.py             # ResNet-50 light estimator training
├── inference.py                   # grammar-constrained generation → USD + camera
├── compute_geometric_metrics.py   # Chamfer distance, F-score (kaolin)
├── compute_visual_metrics.py      # MSE, LPIPS on rendered outputs
├── compute_object_metrics.py      # object detection / pose errors vs. ground truth
├── physical_validity.py           # out-of-bounds / compatibility / intersection checks
├── fp15_grammar.ebnf              # EBNF grammar for constrained decoding
├── symmask.json                   # per-asset rotational symmetry orders
├── environment.yml                # conda environment (training, inference, metrics)
└── utils/
    ├── asset_directory.py         # asset pack catalogue, scene bounds, placement rules
    ├── simpler_polygon_floor_object_generator.py
    ├── utils.py                   # USD parsing, prim encoding/decoding, sorting
    └── utils_classes.py
```

## Installation

Two environments are needed: one conda environment for training, inference, and metrics, and NVIDIA Isaac Sim for data generation and rendering.

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate image2usd
```

`environment.yml` pins `torch==2.8.0` because `kaolin==0.18.0` ships prebuilt wheels for torch 2.7/2.8 only. Install torch, torchvision, and kaolin with the CUDA build matching your GPU (see the comments in `environment.yml`).

### 2. NVIDIA Isaac Sim

Rendering and data generation (`render_*.py`, `generate_*.py`) run inside Isaac Sim's bundled Python, which provides `pxr`, `omni.*`, `isaacsim`, and `carb`. Install Isaac Sim following the [official workstation instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) to `~/isaacsim`, then install the extra dependencies into its Python:

```bash
~/isaacsim/python.sh -m pip install shapely trimesh natsort
```

### 3. Asset pack

Download `assets_bundle.zip` from the [Zenodo record](https://zenodo.org/records/22151734?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijg2OWU0NjhhLTBkZDUtNGE4MS1iNDQ5LTgyMWNkMTAxYTg5OSIsImRhdGEiOnt9LCJyYW5kb20iOiIyY2ZhNmE5OTVjZTNkM2U4MTc2NGFhNTk4YzYxNmY2NSJ9.Jdt5eQ3YDw6z5KrkK55_q8l4qGTWjM-fNfsIkit2h3WW7MR2QnJas-EGtX8m9RzgVu37jhQqw1wmPr92wjmB0Q) and extract it in the repository root:

```bash
unzip assets_bundle.zip -d assets
```

## Usage

### Create 3DR-Set

```bash
python create_3dr_set.py
```

This runs the scene samplers for the training and evaluation splits, renders every scene with Isaac Sim, and prepares the training set for fine-tuning. Outputs land in `usd_data/` (raw training USD + camera), `train_data/` (prepared HF dataset), and `test_data/` (evaluation split). The individual steps can be run separately — see `create_3dr_set.py` for the exact calls.

### Fine-tune

```bash
accelerate launch finetune_gemma.py      # VLM (LoRA), writes to finetuned_gemma/
python finetune_resnet.py                # light estimator, writes to finetuned_resnet/
```

`finetune_gemma.py --encoding_scheme {verbose,p10,p1000,b1999,fp15}` reproduces the encoding ablation; `fp15` is the default used in the paper.

### Inference

`finetune_resnet.py` reports the best checkpoint; pass it to inference:

```bash
python inference.py --cnn_model_dir finetuned_resnet/checkpoint-<best>
```

This writes predicted USD code and camera viewpoints next to each sample in `test_data/`. Render the predictions with:

```bash
~/isaacsim/python.sh render_test_images.py
```

### Evaluation

```bash
python compute_geometric_metrics.py   # Chamfer distance, F-score — needs predicted USD only
python compute_object_metrics.py      # object-level detection and pose errors
python compute_visual_metrics.py      # MSE, LPIPS — needs rendered predictions
python physical_validity.py           # out-of-bounds / background-compatibility / intersection rates
```

All scripts accept `--eval_dataset` and `--suffix` to point at a different split or prediction set (`--suffix code` evaluates ground truth).

## Citation

```bibtex
@inproceedings{mahalder2027image2usd,
  title     = {Image2USD: Scalable 3D Scene Reconstruction from a Single Image via Structured Scene Tokenization},
  author    = {Mahalder, Atandra and Walker, Chase and Jha, Sumit Kumar and Ewetz, Rickard},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgments

This work was supported in part by UF startup funds, the Defense Advanced Research Projects Agency (DARPA) under Cooperative Agreement FA8750-23-2-0501 and the U.S. Department of Energy under Grant DE-SC0024428; the views, opinions, and findings expressed herein are those of the authors and do not necessarily reflect the official views or policies of DARPA, the U.S. Department of Energy, or the U.S. Government.

## License

Code is released under the [MIT License](LICENSE). The 3DR-Set asset pack is distributed separately via Zenodo under its own terms.
