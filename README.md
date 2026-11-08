# NeuroFusion AI

> **AI-Powered MRI Enhancement & Intelligent Disease Localization**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![MONAI 1.3](https://img.shields.io/badge/MONAI-1.3-brightgreen.svg)](https://monai.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NeuroFusion AI is a production-ready, research-grade medical imaging framework designed for volumetric analysis of **Brain MRI**. Built for national-level AI medical imaging hackathons, it combines deep learning restoration transformers, volumetric ROI segmentation, automated pathology localization, explainable AI (Grad-CAM), and web-based 3D rendering into a unified platform.

---

## 🌟 Key Capabilities

- **Automatic Dataset Analysis (Module 1)**: Analyzes Brain MRI, identifies modalities (T1, T1c, T2, FLAIR), and generates volumetric statistics (voxel spacing, SNR, CNR, entropy, contrast, sharpness).
- **Advanced Preprocessing (Module 2)**: SimpleITK N4 bias field correction, isotropic resampling, z-score normalization, skull stripping, and Perona-Malik anisotropic diffusion.
- **Deep MRI Enhancement (Module 3)**: Selectable enhancement architectures including **Restormer** (CVPR 2022), **SwinIR** (ICCV 2021), **MIRNet** (ECCV 2020), and CLAHE baseline.
- **Disease Localization & Heatmaps (Module 4 & 7)**: Automated detection of tumors, peritumoral edema, and necrotic core with GradCAM attention heatmaps and clinical decision support findings.
- **ROI Segmentation (Module 5)**: Multi-class segmentation engine supporting **UNet**, **UNet++**, and **Attention UNet**.
- **Comprehensive Evaluation (Module 6)**: Computes PSNR, SSIM, MSE, RMSE, UQI, FSIM, BRISQUE, NIQE, Entropy, Dice, IoU, and Hausdorff Distance.
- **Interactive 3D Medical Viewer (Module 8)**: WebGL volume rendering powered by **NiiVue**, supporting multi-planar reconstruction (axial, coronal, sagittal, 3D render).
- **Clinical Decision Support & Report Generator**: Automatic technical HTML/PDF report generation with diagnostic suggestions.

---

## 🏗️ Architecture

```
NeuroFusion AI/
├── backend/
│   ├── api/                  # REST API Routes, Schemas, WebSocket manager
│   ├── configs/              # System YAML configuration
│   ├── evaluation/           # Quality metrics, explainability (GradCAM), disease localizer
│   ├── models/
│   │   ├── enhancement/      # Restormer, SwinIR, MIRNet, CLAHE models
│   │   └── segmentation/     # UNet, UNet++, Attention UNet
│   ├── preprocessing/        # Dataset analyzer, preprocessor, filters, augmentation
│   ├── utils/                # NIfTI I/O, GPU benchmarking, visualization, report generator
│   └── main.py               # FastAPI application entrypoint
├── frontend/
│   ├── src/
│   │   ├── components/       # Layout, Glassmorphism components
│   │   ├── pages/            # Landing, Dashboard, Upload, Analysis, Enhancement, Segmentation, Viewer, Settings
│   │   ├── api.ts            # API client & TypeScript interfaces
│   │   └── index.css         # Custom dark-theme medical design system
│   ├── index.html
│   └── package.json
├── dataset/                  # BraTS2020 Brain MRI storage
├── docker/                   # Dockerfile & Docker Compose
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- (Optional) CUDA-enabled GPU for accelerated inference

### 2. Backend Setup
```bash
# Clone and navigate to project root
cd "NeuroFusion AI"

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🐳 Docker Deployment

To launch the full stack with Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📊 Dataset & Benchmark Support

Supported Datasets:
- **Brain MRI**: [BraTS 2020 Dataset (Kaggle)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

*Note: Scans are processed in native 3D NIfTI format. Scans are never converted permanently into lossy 2D image formats.*

---

## 📜 Disclaimer
NeuroFusion AI is built for research, academic evaluation, and demonstration purposes. Output predictions are AI-assisted recommendations and must be verified by a certified radiologist before clinical use.

---

## 📄 License
Distributed under the MIT License.
