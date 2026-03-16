# ABAW-10 Violence Detection

Frame-level violence detection on the [DVD dataset](https://arxiv.org/abs/2506.05372) for the **Fine-Grained Violence Detection Challenge** at the [10th ABAW Workshop & Competition](https://affective-behavior-analysis-in-the-wild.github.io/10th/) (CVPR 2026).

Given a video, the model predicts for every frame whether it depicts a **violent** (1) or **non-violent** (0) event. Performance is measured by **macro F1** across the two classes.

---

## Overview

The pipeline supports a wide family of architectures — from simple 2D CNN baselines to multimodal two-stream fusion networks — all sharing a unified `[B, C, T, H, W] → [B, num_classes, T]` interface for per-frame prediction.

**Key features:**

- 30+ backbone variants in a single model registry (`models.py`)
- Multiple temporal heads: BiLSTM, Transformer, TCN, CRF (Viterbi decoding)
- Multimodal streams: RGB, optical flow (Farneback), skeleton (MediaPipe Pose)
- Fusion strategies: concatenation, gated fusion, cross-attention
- Configurable training via JSON presets (augmentation, sampling, loss, optimizer)
- Boundary-aware loss for transition frames between violent and non-violent segments
- Precomputation scripts for skeleton and optical flow features
- Full-validation evaluation and test-set submission generation

---

## Repository Structure

```
.
├── models.py                        # All model architectures + build_model() factory
├── precompute_features.py           # Skeleton (MediaPipe) & optical flow (Farneback) extraction
├── run_exp_backbones_train_v2.py    # Training script (preset-driven)
├── run_full_val_eval_v2.py          # Full validation evaluation
├── exp_backbones_v2.ipynb           # Main experiment notebook (dataset, training loop, helpers)
├── fullval.ipynb                    # Quick full-validation notebook
├── single_video_inference_plot.ipynb # Per-video inference with visualisation
├── improved_presets.json            # Training presets (conservative / aggressive configs)
├── two_stream_presets.json          # Presets for RGB + optical flow models
├── skeleton_attention_boost_presets.json  # Presets for skeleton fusion models
├── to_tune_more_presets.json        # Extra presets for hyperparameter search
├── requirements.txt                 # Pinned dependencies (CUDA 11.8)
└── test_set_processing/
    ├── README.md                    # Test inference docs
    ├── generate_submissions.py      # Test-set inference → submission CSV/ZIP
    ├── run_all.sh                   # Launcher wrapper
    └── models_suite.*.json          # Model suite configs for batch inference
```

---

## Supported Architectures

### 2D Backbones + Temporal Heads

| Backbone | Temporal Head | Model Key |
|---|---|---|
| ResNet-18 | Conv1d | `resnet18` |
| EfficientNet-B0 | BiLSTM | `efficientnet_b0_bilstm` |
| ConvNeXt-Tiny | BiLSTM | `convnext_tiny_bilstm` |
| ConvNeXt-Tiny | Transformer | `convnext_tiny_transformer` |
| ConvNeXt-Tiny | TCN | `convnext_tiny_tcn` |
| ConvNeXt-Tiny | BiLSTM + CRF | `convnext_tiny_crf` |
| ConvNeXt-Tiny | Multi-Scale + BiLSTM | `convnext_tiny_multiscale` |
| ConvNeXt-Small | BiLSTM | `convnext_small_bilstm` |
| ConvNeXt-Base | BiLSTM | `convnext_base_bilstm` |

### 3D / Video Backbones

| Backbone | Temporal Head | Model Key |
|---|---|---|
| R3D-18 | Conv1d / BiLSTM | `r3d_18_temporal` / `r3d_18_temporal_v2` |
| R(2+1)D-18 | Conv1d / BiLSTM | `r2plus1d_18_temporal` / `r2plus1d_18_temporal_v2` |
| S3D | Conv1d / BiLSTM | `s3d_temporal` / `s3d_temporal_v2` |
| SlowFast R50 | BiLSTM | `slowfast_r50_v2` |
| I3D R50 | BiLSTM | `i3d_r50_v2` |
| Video Swin-T | — | `video_swin_tiny` |
| VideoMAE Small | — | `videomae_small` |
| VideoMAE Base | BiLSTM | `videomae_base_bilstm` |
| VideoMAEv2 Base | BiLSTM | `videomaev2_base_bilstm` |

### Multimodal Fusion

| Modalities | Fusion | Temporal Head | Model Key |
|---|---|---|---|
| RGB + Skeleton | Attention / Gated / Concat | BiLSTM | `skeleton_attention_bilstm`, `skeleton_gated_bilstm`, ... |
| RGB + Skeleton | Attention / Gated / Concat | TCN | `skeleton_attention_tcn`, `skeleton_gated_tcn`, ... |
| RGB + Optical Flow | Cross-Attention | BiLSTM | `twostream_attention_bilstm` |
| RGB + Optical Flow | Gated | BiLSTM / TCN | `twostream_gated_bilstm`, `twostream_gated_tcn` |
| RGB + Optical Flow | Concat | BiLSTM | `twostream_concat_bilstm` |

---

## Installation

```bash
# Clone
git clone https://github.com/kdtsypliakova/ABAW-10-Violence-Detection.git
cd ABAW-10-Violence-Detection

# Create environment (Python 3.10+ recommended)
pip install -r requirements.txt
```

> **Note:** `requirements.txt` is pinned for CUDA 11.8. If you use a different CUDA version, adjust the `--extra-index-url` for PyTorch accordingly.

---

## Data Preparation

### DVD Dataset

Register for the 10th ABAW Competition following the [official instructions](https://affective-behavior-analysis-in-the-wild.github.io/10th/) to obtain access to the DVD database. Extract frames from the provided videos into the following layout:

```
<project_root>/
├── frames/
│   ├── Training/
│   │   ├── 0001/
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── Validation/
│       └── ...
├── labels/
│   ├── Training/
│   └── Validation/
```

### Precompute Skeleton Features

```bash
python precompute_features.py --mode skeleton \
    --frames-root /path/to/frames \
    --output-dir /path/to/skeleton_features
```

This extracts MediaPipe Pose landmarks (top-2 people, 33 keypoints × 3 coords + velocities + pairwise features = 406-dim) and saves `.npy` files per video.

### Precompute Optical Flow

```bash
python precompute_features.py --mode flow \
    --frames-root /path/to/frames \
    --output-dir /path/to/flow_frames
```

This computes Farneback optical flow between consecutive frames and saves flow visualisations as `.jpg` (dx, dy, magnitude → 3 channels), compatible with the standard frame-based data loader.

---

## Training

Training is driven by JSON presets that specify the backbone, augmentation strategy, loss function, optimizer, and all hyperparameters.

```bash
python run_exp_backbones_train_v2.py \
    --preset convnext_tiny_bilstm_conservative \
    --presets-path improved_presets.json
```

Available preset files:

| File | Description |
|---|---|
| `improved_presets.json` | Main configs for single-modality backbones |
| `two_stream_presets.json` | RGB + optical flow two-stream models |
| `skeleton_attention_boost_presets.json` | RGB + skeleton fusion models |
| `to_tune_more_presets.json` | Extra configs for hyperparameter search |

Preset configs control augmentation (TrivialAugWide, strong augmentation, temporal-coherent augmentation), sampling strategies, learning rate schedules, label smoothing, boundary-aware loss weights, and more.

---

## Evaluation

### Full Validation

```bash
python run_full_val_eval_v2.py \
    --run-dir runs/convnext_tiny_clip32_0.78 \
    --notebook exp_backbones_v2.ipynb \
    --device cuda
```

Results are saved to `<run_dir>/full_val_result.json` with per-video and aggregate macro F1 scores.

### Test Set Submission

Generate submission CSV files for the competition:

```bash
python test_set_processing/generate_submissions.py \
    --project-root "$(pwd)" \
    --notebook exp_backbones_v2.ipynb \
    --models-suite test_set_processing/models_suite.convnext_bilstm_unimodal.json \
    --test-frames-root /path/to/DVD/Test/frames \
    --out-root test_set_processing/submissions_generated \
    --device cuda \
    --zip
```

Or use the launcher:

```bash
bash test_set_processing/run_all.sh \
    --models-suite test_set_processing/models_suite.convnext_bilstm_unimodal.json \
    --test-frames-root /path/to/DVD/Test/frames
```

Each model produces a folder of per-video `.csv` files (columns: `Frame_Number`, `Predicted_Label`) and optionally a `.zip` ready for upload.

---

## Inference on a Single Video

Use `single_video_inference_plot.ipynb` to run a trained model on one video and visualise the per-frame predictions alongside ground truth.

---

## License

Please refer to the EULA signed during competition registration for the DVD dataset usage terms. Code in this repository is provided for research purposes.
