<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/sunwei925/LEIQ-Assessor)](https://github.com/sunwei925/LEIQ-Assessor)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-brightgreen?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/LEIQ-Assessor)
[![arXiv](https://img.shields.io/badge/arXiv-2606.29752-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2606.29752)

**🥈 2nd Place Solution for [QoMEX 2026 Grand Challenge on Low-light Enhanced Image Quality Assessment](https://github.com/CQUPT-HuBo90/MLEDataset)**

*Official Implementation of "LEIQ-Assessor: Multi-dimensional Quality Assessment of Low-light Enhanced Images via Multi-task Learning"*

[📖 Paper](https://arxiv.org/abs/2606.29752) | [📊 Challenge](https://github.com/CQUPT-HuBo90/MLEDataset)

</div>

---

## 📋 Table of Contents

- [🎯 Introduction](#-introduction)
- [🏆 Challenge Results](#-challenge-results)
- [📦 Installation](#-installation)
- [📊 Dataset](#-dataset)
- [🧪 Testing](#-testing)
- [📚 Citation](#-citation)

---

## 🎯 Introduction

**Low-light Image Enhancement Algorithms (LIEAs)** aim to improve the visibility of images captured under poor illumination. However, the enhancement process often introduces artifacts such as noise amplification, color shift, structural damage, and over-exposure, which degrade the perceptual quality of the enhanced images. A reliable **Image Quality Assessment (IQA)** metric for evaluating enhancement effects is therefore of great importance for both the development of LIEAs and their practical applications.

This repository presents **LEIQ-Assessor**, a multi-dimensional quality assessment model for low-light image enhancement, developed for the QoMEX 2026 Grand Challenge on Low-light Enhanced Image Quality Assessment. LEIQ-Assessor simultaneously predicts the overall **Mean Opinion Score (MOS)** together with six perceptual sub-attributes:

- **Lightness** — brightness performance
- **Color Fidelity** — color vividness and faithfulness
- **Noise** — noise suppression level
- **Exposure** — exposure balance
- **Naturalness** — natural appearance
- **Content Recovery** — detail restoration in dark areas

### 🔑 Key Features

- **🔬 Multi-task Learning** — Jointly predicts the overall MOS and six fine-grained dimensional scores in a single forward pass, providing both an accurate global assessment and interpretable attribute-level diagnostics.
- **🧠 SigLIP2 Backbone** — Adopts a pre-trained SigLIP2 Vision Transformer as the feature extractor, whose sigmoid-based vision–language contrastive pre-training yields semantically rich and perceptually discriminative representations.
- **📐 PLCC-based Loss** — All seven regression heads are jointly optimized with a composite Pearson Linear Correlation Coefficient (PLCC) loss that directly maximizes the linear agreement between predictions and ground-truth annotations across all quality dimensions, enabling cross-attribute knowledge transfer.

### 🏆 Key Achievements

- **🥈 2nd Place** in QoMEX 2026 Grand Challenge on Low-light Enhanced Image Quality Assessment

---

## 🏆 Challenge Results

Comparison with representative no-reference IQA methods on the MLE dataset. Mean SRCC / PLCC over 10 random splits are reported. **Bold** indicates the best result.

| Task | BRISQUE | NIQE | MANIQA | StairIQA | CLIP-IQA+ | LIQE | **LEIQ-Assessor (Ours)** |
|------|:-------:|:----:|:------:|:--------:|:---------:|:----:|:------------------------:|
| MOS | 0.365 / 0.402 | 0.365 / 0.406 | 0.274 / 0.315 | 0.425 / 0.442 | 0.489 / 0.501 | 0.595 / 0.603 | **0.886 / 0.892** |
| Light | 0.117 / 0.189 | 0.286 / 0.408 | 0.178 / 0.262 | 0.341 / 0.412 | 0.173 / 0.256 | 0.339 / 0.382 | **0.796 / 0.863** |
| Color | 0.279 / 0.329 | 0.335 / 0.383 | 0.231 / 0.271 | 0.384 / 0.374 | 0.412 / 0.434 | 0.515 / 0.526 | **0.821 / 0.830** |
| Noise | 0.259 / 0.301 | 0.297 / 0.335 | 0.392 / 0.412 | 0.458 / 0.460 | 0.552 / 0.571 | 0.655 / 0.662 | **0.831 / 0.850** |
| Exposure | 0.336 / 0.382 | 0.195 / 0.217 | 0.160 / 0.128 | 0.282 / 0.278 | 0.389 / 0.363 | 0.439 / 0.393 | **0.817 / 0.881** |
| Nature | 0.369 / 0.409 | 0.318 / 0.342 | 0.282 / 0.285 | 0.401 / 0.423 | 0.480 / 0.483 | 0.586 / 0.586 | **0.888 / 0.906** |
| Content Rec. | 0.306 / 0.351 | 0.380 / 0.437 | 0.311 / 0.330 | 0.478 / 0.474 | 0.488 / 0.498 | 0.610 / 0.611 | **0.863 / 0.870** |
| **Average** | 0.290 / 0.338 | 0.311 / 0.361 | 0.261 / 0.286 | 0.396 / 0.409 | 0.426 / 0.444 | 0.534 / 0.538 | **0.843 / 0.870** |

---

## 📦 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.4
- CUDA >= 11.0 (for GPU inference)

### Environment Setup

```bash
conda create -n LEIQ python=3.8
conda activate LEIQ

pip install -r requirements.txt
```

Core dependencies:

| Package | Version |
|---------|---------|
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| open_clip_torch | 2.32.0 |
| numpy | 1.24.3 |
| scipy | 1.10.1 |
| pillow | 10.4.0 |

---

## 📊 Dataset

LEIQ-Assessor is trained and evaluated on the **MLE (Multi-annotated and multimodal Low-light image Enhancement)** dataset from the QoMEX 2026 Grand Challenge, which contains 1,000 low-light enhanced images annotated with seven-dimensional quality labels (MOS + 6 sub-attributes) and textual descriptions.

Please refer to the [challenge repository](https://github.com/CQUPT-HuBo90/MLEDataset) for data download and format details.

For testing, prepare a directory of images together with a `metadata.csv` listing two columns (`id`, `file_name`):

```
test_images/
├── metadata.csv          # id,file_name
├── xxx.png
└── ...
```

If `metadata.csv` is missing, all supported images (`png/jpg/jpeg/bmp/webp`) in `test_images/` are used and the file stem is taken as the `id`.

---

## 🧪 Testing

This repository provides three testing scripts.

| Script | Usage | Input | Output |
|--------|-------|-------|--------|
| `test_ensemble.py` | Ensemble test | Test directory | MOS averaged over both models (10 splits each), then averaged again |
| `test_single.py` | Single-checkpoint batch test | One trained `.pth` + test directory | MOS from that single checkpoint for every image |
| `test_one_image.py` | Single-image inference | One trained `.pth` + one image | MOS for that image (multi-task model also prints 6 attribute scores) |

Two checkpoints are released:

| Model | Backbone | Input Size | Task | Download |
|-------|----------|:---------:|------|:--------:|
| `SigLIP2_384_Image` | ViT-B/16-SigLIP2 | 384×384 | Single-task (MOS) | [Baidu Yun](https://pan.baidu.com/s/19u6uNDDWKARMRVcAunQEuw) (code: `rpkv`) |
| `SigLIP2_ViTG_384_Image_10splits` | ViT-SO400M/14-SigLIP2 | 378×378 | Multi-task (MOS + 6 attributes) | [Baidu Yun](https://pan.baidu.com/s/1n35i2y0aY5moLbtyVobX0w) (code: `rs6d`) |

After downloading, extract each archive into the repository root so the layout matches the structure shown at the end of this section.

Each split ships a pair of files:

- **`.pth`** — model weights, named `{model}_MLE_v{split}_ep{epoch}_SRCC{value}.pth`
- **`.mat`** — training-set predictions/labels used to fit a logistic mapping that maps the raw output onto the MOS scale

### 1. Ensemble Test (`test_ensemble.py`)

Averages the two models' 10-split means.

```bash
bash test.sh
```

Or with custom arguments:

```bash
python test_ensemble.py \
    --test_dir test_images \
    --metadata_csv test_images/metadata.csv \
    --st_ckpt_dir SigLIP2_384_Image \
    --st_ckpt_name SigLIP2_384_Image \
    --mt_ckpt_dir SigLIP2_ViTG_384_Image_10splits \
    --mt_ckpt_name SigLIP2_ViTG_384_Image_MT \
    --n_splits 10 \
    --batch_size 16 \
    --num_workers 4 \
    --output_json result_ensemble.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--test_dir` | `test_images` | Directory of test images |
| `--metadata_csv` | `test_images/metadata.csv` | Metadata CSV with `id`, `file_name` columns |
| `--st_ckpt_dir` | `SigLIP2_384_Image` | Single-task checkpoint directory |
| `--st_ckpt_name` | `SigLIP2_384_Image` | Single-task checkpoint filename prefix |
| `--mt_ckpt_dir` | `SigLIP2_ViTG_384_Image_10splits` | Multi-task checkpoint directory |
| `--mt_ckpt_name` | `SigLIP2_ViTG_384_Image_MT` | Multi-task checkpoint filename prefix |
| `--n_splits` | `10` | Expected number of splits |
| `--batch_size` | `16` | Inference batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--output_json` | `result.json` | Output JSON path |
| `--round` | `4` | Decimal places for MOS |
| `--clamp_to_range` | `False` | Clamp MOS to `[min_mos, max_mos]` |
| `--min_mos` | `0.0` | MOS lower bound (requires `--clamp_to_range`) |
| `--max_mos` | `10.0` | MOS upper bound (requires `--clamp_to_range`) |

### 2. Single-Checkpoint Batch Test (`test_single.py`)

Loads **one trained `.pth`** and scores every image in the test directory (no 10-split averaging).

```bash
# Single-task model
python test_single.py \
    --model st \
    --ckpt_path SigLIP2_384_Image/SigLIP2_384_Image_MLE_v0_ep30_SRCC0.92.pth \
    --test_dir test_images \
    --metadata_csv test_images/metadata.csv \
    --output_json result_single_st.json

# Multi-task model
python test_single.py \
    --model mt \
    --ckpt_path SigLIP2_ViTG_384_Image_10splits/SigLIP2_ViTG_384_Image_MT_MLE_v0_ep30_SRCC0.93.pth \
    --output_json result_single_mt.json

# Explicitly specify the .mat (defaults to the .pth stem with .mat)
python test_single.py \
    --model st \
    --ckpt_path path/to/xxx.pth \
    --mat_path path/to/xxx.mat \
    --output_json result_single.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `st` | Architecture: `st` (single-task, 384) / `mt` (multi-task, 378) |
| `--ckpt_path` | *(required)* | Path to a single trained `.pth` checkpoint |
| `--mat_path` | `.pth` stem + `.mat` | Path to the `.mat` for logistic mapping |
| `--test_dir` | `test_images` | Directory of test images |
| `--metadata_csv` | `test_images/metadata.csv` | Metadata CSV with `id`, `file_name` columns |
| `--output_json` | `result_single.json` | Output JSON path |
| `--batch_size` | `16` | Inference batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--round` | `4` | Decimal places for MOS |
| `--clamp_to_range` | `False` | Clamp MOS to `[min_mos, max_mos]` |
| `--min_mos` | `0.0` | MOS lower bound (requires `--clamp_to_range`) |
| `--max_mos` | `10.0` | MOS upper bound (requires `--clamp_to_range`) |

### 3. Single-Image Inference (`test_one_image.py`)

Loads **one trained `.pth`** and scores **one image**, printing the result to the terminal. The multi-task model additionally prints the 6 attribute scores.

```bash
# Single-task model
python test_one_image.py \
    --image_path test_images/xxx.png \
    --model st \
    --ckpt_path SigLIP2_384_Image/SigLIP2_384_Image_MLE_v0_ep30_SRCC0.92.pth

# Multi-task model (also prints light/color/noise/exposure/nature/content_recovery)
python test_one_image.py \
    --image_path test_images/xxx.png \
    --model mt \
    --ckpt_path SigLIP2_ViTG_384_Image_10splits/SigLIP2_ViTG_384_Image_MT_MLE_v0_ep30_SRCC0.93.pth

# Explicitly specify the .mat and clamp the output range
python test_one_image.py \
    --image_path xxx.png \
    --model st \
    --ckpt_path path/to/xxx.pth \
    --mat_path path/to/xxx.mat \
    --clamp_to_range --min_mos 0 --max_mos 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--image_path` | *(required)* | Path to a single image |
| `--model` | `st` | Architecture: `st` (single-task, 384) / `mt` (multi-task, 378) |
| `--ckpt_path` | *(required)* | Path to a single trained `.pth` checkpoint |
| `--mat_path` | `.pth` stem + `.mat` | Path to the `.mat` for logistic mapping |
| `--round` | `4` | Decimal places for MOS / attribute scores |
| `--clamp_to_range` | `False` | Clamp MOS to `[min_mos, max_mos]` |
| `--min_mos` | `0.0` | MOS lower bound (requires `--clamp_to_range`) |
| `--max_mos` | `10.0` | MOS upper bound (requires `--clamp_to_range`) |

### Directory Structure

```
LEIQ-Assessor/
├── README.md
├── requirements.txt
├── test_ensemble.py          # Ensemble test (two models × 10 splits, averaged)
├── test_single.py            # Single-checkpoint batch test (one .pth over a directory)
├── test_one_image.py         # Single-image inference (one .pth over one image)
├── IQAModels.py              # Model definitions
├── test.sh                   # Quick-run script for the ensemble test
├── test_images/              # Test image directory
│   ├── metadata.csv          # id,file_name columns
│   ├── xxx.png
│   └── ...
├── SigLIP2_384_Image/        # Single-task checkpoint directory
│   ├── SigLIP2_384_Image_MLE_v0_ep*.pth
│   ├── SigLIP2_384_Image_MLE_v0_ep*.mat
│   └── ... (v0–v9, 10 splits)
└── SigLIP2_ViTG_384_Image_10splits/   # Multi-task checkpoint directory
    ├── SigLIP2_ViTG_384_Image_MT_MLE_v0_ep*.pth
    ├── SigLIP2_ViTG_384_Image_MT_MLE_v0_ep*.mat
    └── ... (v0–v9, 10 splits)
```

---

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{sun2026leiq,
  title={LEIQ-Assessor: Multi-dimensional Quality Assessment of Low-light Enhanced Images via Multi-task Learning},
  author={Sun, Wei and Jiang, Yanwei and Zhu, Dandan and Sang, Jinqiu and Xu, Jikai and Zhang, Weixia and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE International Conference on Quality of Multimedia Experience (QoMEX)},
  year={2026}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
