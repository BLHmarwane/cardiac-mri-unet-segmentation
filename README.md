# Cardiac MRI U-Net Segmentation

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BLHmarwane/cardiac-mri-unet-segmentation/main?urlpath=proxy/8501/)

**A reproducible deep-learning case study for cardiac MRI contour segmentation, focused on endocardium and epicardium masks, quantitative validation, visual interpretation, and deployable inference.**

This project revisits an academic medical-imaging segmentation work and turns it into a portfolio-grade pipeline: not only a trained U-Net, but a documented workflow where data pairing, preprocessing, metrics, qualitative inspection, CLI inference, and Streamlit deployment are all explicit.

> This is a research and portfolio demonstration, not a medical device or clinical decision-support tool.

![Workflow](docs/assets/architecture_pipeline.png)

## Live Demo

Launch the interactive Streamlit demo from Binder:

[![Open Streamlit Demo](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BLHmarwane/cardiac-mri-unet-segmentation/main?urlpath=proxy/8501/)

The demo includes pretrained Keras checkpoints for both targets. Visitors can either upload a grayscale cardiac image or click **Run demo sample** to test the inference interface without access to the original dataset.

## Clinical and Technical Motivation

Cardiac contour segmentation is a central step in quantitative cardiac MRI analysis. Delineating the **endocardium** and **epicardium** makes it possible to estimate ventricular cavity and myocardial wall regions, which are used downstream for functional measurements and anatomical interpretation.

From a machine-learning point of view, this is a useful problem because high overlap scores alone are not enough. A segmentation can have good Dice while still failing locally on anatomical boundaries. For that reason, this project evaluates both:

- **Region overlap**: Dice, IoU, precision, recall.
- **Boundary quality**: Hausdorff distance and ASSD.

This distinction is important in medical imaging: a model should not only "fill roughly the right area", it should also preserve contours in a way that can be inspected and trusted.

## Dataset and Task

The local dataset used for validation contains grayscale cardiac MRI frames with paired binary masks:

| Split | Frames | Endocardium masks | Epicardium masks |
| --- | ---: | ---: | ---: |
| Train | 300 | 300 | 300 |
| Validation | 30 | 30 | 30 |

Two independent binary segmentation tasks are modeled:

- **Endocardium**: inner cardiac contour / cavity boundary.
- **Epicardium**: outer myocardial contour.

The dataset is not redistributed as a full dataset in this repository because it does not ship with a formal public license. Selected validation panels are included only as demonstrative project evidence, and the repository includes pretrained checkpoints so the public app can run.

## Methodology

The core model is a U-Net encoder-decoder architecture with skip connections. The repository contains:

- a **legacy U-Net** compatible with the original Keras `.h5` checkpoints;
- a **modern configurable U-Net** variant with batch normalization/dropout options;
- deterministic image-mask pairing by filename;
- grayscale normalization and resize to `256 x 256`;
- binary mask thresholding at `0.5`;
- reproducible evaluation exports in CSV/JSON.

The implementation is intentionally separated from notebooks. The goal is to make the experiment auditable: data loading, model construction, evaluation, visualization and deployment each live in their own module.

## Results and Interpretation

Validation was performed on 30 held-out frames using the legacy U-Net checkpoints.

![Metrics summary](docs/assets/metrics_summary.png)

| Target | Dice | IoU | Precision | Recall | Hausdorff | ASSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Endocardium | 0.9374 | 0.8849 | 0.9442 | 0.9369 | 7.35 | 2.29 |
| Epicardium | 0.9198 | 0.8639 | 0.9406 | 0.9187 | 20.66 | 4.58 |

The endocardium model is the most stable component of the pipeline: its Dice score is close to `0.94`, and its boundary metrics remain comparatively low. This suggests that the learned contour is usually close to the reference mask, not only overlapping it globally.

The epicardium model also reaches strong overlap, with Dice around `0.92`, but the higher Hausdorff distance indicates more boundary variability. This is coherent with the task: the outer myocardial contour is often harder to localize because the transition between myocardium and surrounding tissue can be less contrasted than the inner cavity boundary.

High precision for both targets suggests that the models do not massively over-segment the anatomy. The more informative weakness appears in recall and boundary-sensitive metrics, where local misses or contour shifts become visible.

## Qualitative Analysis

Each panel shows the input frame, ground-truth mask, predicted mask and overlay. In the overlay, agreement appears visually as the model contour aligns with the annotated structure.

### Endocardium

![Endocardium sample 1](docs/assets/qualitative_endo_01.png)

![Endocardium sample 2](docs/assets/qualitative_endo_02.png)

### Epicardium

![Epicardium sample 1](docs/assets/qualitative_epi_01.png)

![Epicardium sample 2](docs/assets/qualitative_epi_02.png)

These examples make the numerical results easier to interpret. Good Dice scores are visible as strong spatial overlap, while the boundary metrics explain why the epicardium task remains more sensitive to local contour deviations.

## Engineering Value

This repository is also an engineering refactor of a notebook-based academic project. The added value is the transition from "model that was trained once" to a small reproducible product-like workflow:

- `src/segmed/`: package for dataset loading, U-Net models, metrics, evaluation and visualization;
- `configs/`: endocardium/epicardium experiment configuration;
- `segmed` CLI: train, evaluate and predict from the terminal;
- `app/streamlit_app.py`: interactive inference demo;
- `docs/results/`: public validation summaries used by this README;
- `tests/`: unit tests for deterministic metric/config behavior.

This structure makes the project inspectable for people who do not want to run the full training pipeline, while still keeping the codebase reproducible for technical review.

## Limitations and Next Steps

The current validation set is small (`30` frames), so the results should be interpreted as a controlled project validation rather than a general benchmark. The next scientific step is external validation on a public benchmark such as Sunnybrook or ACDC, with patient-level splits and stronger reporting of outliers.

Other planned improvements:

- compare the legacy U-Net against the modern U-Net configuration;
- add uncertainty or confidence maps for visual quality control;
- report per-image worst cases and failure modes;
- add a public benchmark loader once licensing and format assumptions are explicit.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Evaluate the public endocardium checkpoint:

```bash
segmed evaluate \
  --config configs/endo.yaml \
  --weights models/endo_legacy_unet.h5 \
  --split Val
```

Run the local Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

For this local preparation folder only, if the path contains special characters such as `#`, use:

```bash
PYTHONPATH=src python -m segmed.cli evaluate \
  --config configs/endo.yaml \
  --weights models/endo_legacy_unet.h5 \
  --split Val
```

## Repository Notes

Full local datasets, training outputs, TensorBoard logs and scratch artifacts are ignored by Git. The included model checkpoints and selected visual panels are used to make the repository understandable and demo-ready without redistributing the full dataset.
