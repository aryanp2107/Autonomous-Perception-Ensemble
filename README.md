# Autonomous Perception Ensemble

Multi-model sensor fusion for autonomous driving scene understanding.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project combines three perception models into a unified scene understanding system:

```
┌─────────────────────────────────────────────────────────────┐
│                      CAMERA INPUT                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   YOLOv8      │ │    U-Net      │ │    MiDaS      │
│  Detection    │ │ Segmentation  │ │    Depth      │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                ┌─────────────────┐
                │     FUSION      │
                └─────────────────┘
```

## Models

| Model | Task | Dataset | Output |
|-------|------|---------|--------|
| YOLOv8-nano | Object Detection | BDD100K | Bounding boxes |
| U-Net | Drivable Area | BDD100K | Segmentation mask |
| MiDaS | Depth Estimation | Pretrained | Depth map |

## Results

| Component | Metric | Value |
|-----------|--------|-------|
| Detection | mAP@50 | ~0.50 |
| Segmentation | IoU | ~0.85 |
| Depth | Relative | Pretrained |
| Ensemble | FPS | ~15-20 |

## Project Structure

```
├── notebooks/
│   ├── 01_Detection.ipynb      # YOLOv8 fine-tuning
│   ├── 02_Segmentation.ipynb   # U-Net training
│   └── 03_Fusion_ONNX.ipynb    # Ensemble + export
├── models/                      # ONNX models (see HuggingFace)
├── src/
│   └── inference.py            # Inference pipeline
├── demo/
│   └── app.py                  # Gradio demo
└── assets/
    └── sample_output.png
```

## Quick Start

### Installation

```bash
git clone https://github.com/aryanp2107/Autonomous-Perception-Ensemble.git
cd Autonomous-Perception-Ensemble
pip install -r requirements.txt
```

### Inference

```python
from src.inference import PerceptionEnsemble

# Load ensemble
ensemble = PerceptionEnsemble(
    detection_model="models/yolov8n_bdd100k.onnx",
    segmentation_model="models/unet_drivable.onnx",
    depth_model="models/midas_small.onnx"
)

# Run on image
result = ensemble.predict("path/to/dashcam.jpg")

# Visualize
ensemble.visualize(result, save_path="output.png")
```

### Demo

```bash
cd demo
python app.py
# Opens Gradio interface at localhost:7860
```

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| 01_Detection | Fine-tune YOLOv8 on BDD100K | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| 02_Segmentation | Train U-Net for drivable area | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| 03_Fusion_ONNX | Combine models + ONNX export | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |

## Models Download

ONNX models hosted on HuggingFace:

```bash
# Download all models
huggingface-cli download aryanp2107/autonomous-perception-ensemble --local-dir models/
```

Or download individually:
- [yolov8n_bdd100k.onnx](https://huggingface.co/aryanp2107/autonomous-perception-ensemble)
- [unet_drivable.onnx](https://huggingface.co/aryanp2107/autonomous-perception-ensemble)
- [midas_small.onnx](https://huggingface.co/aryanp2107/autonomous-perception-ensemble)

## Dataset

[BDD100K](https://www.bdd100k.com/) — Berkeley DeepDrive 100K

We use a ~10K image subset via Roboflow for training.

## Tech Stack

- **Detection:** Ultralytics YOLOv8
- **Segmentation:** PyTorch U-Net
- **Depth:** Intel MiDaS
- **Export:** ONNX Runtime
- **Demo:** Gradio

## Deployment

Deployed on [Arxelos](https://arxelos.com/perception) (coming soon)

## License

MIT

## Author

**Aryan Patel**  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/aryanp2107)
