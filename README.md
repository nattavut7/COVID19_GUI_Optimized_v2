# COVID19_GUI_Optimized_v2

> A GPU-Optimized PyTorch Framework with an Interactive GUI for COVID-19 Chest X-ray Classification

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)
![GPU](https://img.shields.io/badge/GPU-RTX%205070Ti-success)

---

## Overview

COVID19_GUI_Optimized_v2 is a modern PyTorch-based research framework developed for automatic COVID-19 chest X-ray image classification. The framework integrates GPU-accelerated deep learning training, an intuitive graphical user interface (GUI), model pruning, ONNX export, TensorBoard visualization, and real-time prediction into a single research platform.

The project has been optimized for NVIDIA RTX 50-series GPUs and Intel Core Ultra processors to maximize training efficiency while maintaining high classification performance.

---

## Features

- GPU-Accelerated PyTorch Training
- Automatic Mixed Precision (AMP)
- CUDA Tensor Core Optimization
- TensorBoard Support
- Interactive Prediction GUI
- Model Pruning
- ONNX Export
- Dynamic Quantization
- Automatic Experiment Logging
- Confusion Matrix Generation
- Classification Report Export
- Training History Visualization
- Accuracy & Loss Curves
- Multi-Class Chest X-ray Classification

---
## Dataset

This folder is intentionally left empty.

Please download the dataset before training.

Recommended dataset

COVID-19 Radiography Database

https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

After downloading

1. Extract into

```
data/raw/
```

2. Run

```bash
python scripts/prepare_xray.py --mode 3class
```

The processed dataset will be created automatically under

```
data/prepared/xray_3class
```

## Supported Classes

- COVID-19
- Normal
- Pneumonia

---

## Hardware Optimization

This framework has been optimized for the following hardware:

| Component | Specification |
|------------|--------------|
| GPU | NVIDIA GeForce RTX 5070 Ti 16GB GDDR7 |
| CPU | Intel Core Ultra 9 285K |
| RAM | 32GB DDR5 |
| Storage | WD Black SN850X NVMe SSD |
| CUDA | 12.8 |
| PyTorch | 2.11 |

Implemented optimizations include

- Mixed Precision Training
- Channels Last Memory Format
- cuDNN Benchmark
- TF32 Support
- Optimized DataLoader
- Multi-worker Loading
- Automatic GPU Detection

---

## Project Structure

```
COVID19_GUI_Optimized_v2
│
├── configs/
├── data/
│   ├── raw/
│   └── prepared/
├── outputs/
├── reports/
├── scripts/
├── src/
├── research_gui.py
├── requirements.txt
├── install_windows_conda.bat
├── run_gui.bat
└── README.md
```

---

## Installation

Clone repository

```bash
git clone https://github.com/USERNAME/COVID19_GUI_Optimized_v2.git

cd COVID19_GUI_Optimized_v2
```

Create environment

```bash
install_windows_conda.bat
```

or

```bash
conda env create -f environment.yml

conda activate q1covid-torch-v2
```

---

## Training

```bash
python scripts/train.py --config configs/xray_3class_rtx5070ti.yaml
```

Training automatically generates

- Best Model
- Last Checkpoint
- Accuracy Curve
- Loss Curve
- Classification Report
- Confusion Matrix
- TensorBoard Logs

---

## TensorBoard

```bash
tensorboard --logdir outputs
```

Open

```
http://localhost:6006
```

---

## Prediction

GUI

```bash
python research_gui.py
```

or

```bash
run_gui.bat
```

Command Line

```bash
python scripts/predict.py \
--checkpoint outputs/.../original_model.pt \
--image sample.png
```

---

## Model Pruning

```bash
python scripts/prune_and_export.py \
--config configs/xray_3class_rtx5070ti.yaml \
--checkpoint outputs/.../original_model.pt
```

---

## Export

Supported export formats

- PyTorch (.pt)
- ONNX
- Dynamic Quantized Model

---

## Output Example

```
outputs/

mobilenetv2_bs64_rtx5070ti_20260717_140530/

├── original_model.pt
├── training_history.csv
├── original_results.csv
├── accuracy_curve.png
├── loss_curve.png
├── reports/
├── checkpoints/
└── tensorboard/
```

---

## GUI Features

The interactive GUI provides

- Image Preview
- AI Prediction
- Confidence Score
- Probability Bar
- Inference Latency
- GPU Device Information
- Model Selection
- Real-Time Analysis

---

## Performance

Features include

- CUDA Acceleration
- Mixed Precision
- Tensor Core Utilization
- Multi-worker Data Loading
- Automatic Experiment Management

Designed for modern NVIDIA GPUs with high throughput and low inference latency.

---

## Future Development

Planned features

- Grad-CAM Visualization
- Explainable AI (XAI)
- Vision Transformer Support
- ConvNeXt Support
- EfficientNetV2
- Swin Transformer
- Automatic Hyperparameter Search
- K-Fold Cross Validation
- Multi-GPU Training
- TensorRT Export
- Medical AI Benchmark Suite

---

## Citation

Sriwiboon, N. Efficient and lightweight CNN model for COVID-19 diagnosis from CT and X-ray images using customized pruning and quantization techniques. Neural Computing and Applications, 37, 13059–13078 (2025). https://doi.org/10.1007/s00521-025-11219-0

---

## License

MIT License

---

## Author

**Nattavut Sriwiboon, Ph.D.**

Department of Computer Science and Information Technology, Faculty of Science and Health Technology, Kalasin University, Kalasin, Thailand.

---

## Acknowledgements

The authors would like to convey their thanks and appreciation to the ‘‘Kalasin University’’ for supporting this work.
