# Cardiac U-Net Segmentation

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BLHmarwane/cardiac-mri-unet-segmentation/main?urlpath=proxy/8501/)

Reproducible TensorFlow/Keras pipeline for binary cardiac MRI segmentation. The V1 focuses on the academic M2 dataset already available locally in `Projet#2`, with separate endocardium and epicardium models, quantitative evaluation, qualitative overlays, a CLI, and a Streamlit demo.

## Why This Project

The original work lived mostly in notebooks. This version turns it into a portfolio-ready project:

- explicit dataset loading and image-mask matching;
- reproducible Keras U-Net training;
- compatibility with the historical `.h5` checkpoints through a legacy U-Net architecture;
- Dice, IoU, precision, recall, Hausdorff distance, and ASSD;
- saved visual panels for interpretation;
- CLI and Streamlit demo for interviews.

The local dataset is not committed because its redistribution rights are not established. The pretrained Keras checkpoints are included so the public Streamlit demo can run with an uploaded image or the synthetic demo sample. Public benchmark support, such as Sunnybrook or ACDC, is planned as a V2.

## Live Demo

Click the Binder badge above to launch the Streamlit app in a browser. The first launch can take a few minutes while the environment builds. Once open, use **Run demo sample** or upload a grayscale cardiac image.

The app is also ready for Streamlit Community Cloud with:

```text
app/streamlit_app.py
requirements.txt
```

Suggested Streamlit Cloud URL after deployment:

```text
https://cardiac-mri-unet-segmentation.streamlit.app
```

## Project Layout

```text
src/segmed/          Python package: data, model, metrics, pipeline, CLI
configs/            Endocardium and epicardium YAML configs
app/                Streamlit demo
tests/              Unit tests for deterministic pieces
Projet#2/           Local legacy project and dataset, ignored by Git
outputs/            Generated metrics, panels, and checkpoints, ignored by Git
models/             Public demo checkpoints for endo and epi inference
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI Usage

In this local preparation folder, prefer `python -m segmed.cli` because the parent path contains `#`, which can break venv console-script shebangs on macOS. In a normal cloned GitHub path, the installed `segmed` command works as expected.

Evaluate an existing endocardium checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m segmed.cli evaluate \
  --config configs/endo.yaml \
  --weights models/endo_legacy_unet.h5 \
  --split Val
```

Train from scratch:

```bash
PYTHONPATH=src .venv/bin/python -m segmed.cli train --config configs/endo.yaml
PYTHONPATH=src .venv/bin/python -m segmed.cli train --config configs/epi.yaml
```

Predict one image:

```bash
PYTHONPATH=src .venv/bin/python -m segmed.cli predict \
  --config configs/endo.yaml \
  --weights outputs/endo/checkpoints/final.weights.h5 \
  --image "Projet#2/ProjetDL2024/Dataset/frames/Val/example.png"
```

## Streamlit Demo

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

The app lets you choose the target, load compatible weights, upload a cardiac MRI image, and optionally upload the ground-truth mask to compute metrics.

## Current Validation Snapshot

Using the legacy checkpoints on the local validation split:

| Target | Dice | IoU | Precision | Recall | Hausdorff | ASSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Endocardium | 0.9374 | 0.8849 | 0.9442 | 0.9369 | 7.35 | 2.29 |
| Epicardium | 0.9198 | 0.8639 | 0.9406 | 0.9187 | 20.66 | 4.58 |

## Notes for Portfolio Presentation

This project can be presented as a cleanup and industrialization of an academic segmentation project: the value is not only the U-Net itself, but the full reproducible workflow around data handling, validation, interpretability, and demo packaging.
