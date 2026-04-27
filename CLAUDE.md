# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSSG (Multi-scale Speaker Graph Network) is an Active Speaker Detection system from an IEEE Transactions on Multimedia paper. It detects who is speaking in video using audio-visual fusion with multi-scale visual features and graph neural networks. The primary dataset is AVA Active Speaker; Columbia is used for cross-dataset evaluation.

## Build & Run Commands

### Environment Setup
```bash
conda create -n MSSG python=3.7.9 anaconda
conda activate MSSG
pip install -r requirements.txt
# Also requires: torch >= 1.9, torch_geometric >= 2.0, python_speech_features
```

### Three-Stage Pipeline
```bash
# Stage 1: Data preparation (steps 0-8, run individually)
bash data_prep.sh <step>     # 0=download AVA, 1=CSVs, 2-8=multi-scale image extraction

# Stage 2: Graph initialization (steps 0-9, run individually)
bash graph_init.sh <step>    # 0=pretrain encoders (optional), 2-8=extract node embeddings, 9=build graph

# Stage 3: Graph training
bash graph_train.sh          # Trains MSSG GNN model (60 epochs)
```

### Direct Training Commands
```bash
# Pre-train individual encoders (train_type: face, background, face_body, face_large, face_down, face_body_large)
python train.py --dataPathAVA ./data/AVA --train_type face --batchSize 1000 --maxEpoch 30

# Train the graph neural network
python train_graph.py --gpu_id=0 --numv=2000 --batch_size=64 --graph_path=./graphs --lr=0.001

# Real-time inference
python realtime/realtime_main.py
```

### Student Model (proposed/)
```bash
python proposed/train_student.py   # Train distilled student model for faster inference
python proposed/eval_ava.py        # Evaluate on AVA
python proposed/eval_columb.py     # Evaluate on Columbia
```

## Architecture

### Two-Phase Training

**Phase 1 — Encoder Pre-training** (`train.py` → `ASD.py`):
Trains 6 separate visual encoders (3D-Conv, channels `[1,32,64,128]`) and 1 audio encoder (MFCC-based). Each encoder targets a different visual scale of the speaker. Loss = `lossAV` + 0.5 × `lossV`. Pre-trained weights go in `predata/pretrain_model/`.

**Phase 2 — Graph Training** (`train_graph.py` → `model/models_gnn.py`):
The `MSSG` class builds a speaker graph (2000 nodes per clip). Node features are 7 concatenated 128-dim embeddings (896 total) + speaker count + spatial info. Graph layers: `EdgeConv` + `SAGEConv` with face-size-based masking. Loss: `BCELoss` with temperature scaling `r = 1.3 - 0.02*(epoch-1)`. Metric: mAP.

### Seven Multi-Scale Visual Features (Eq. 3-8 from paper)
1. `F_face` — face region (112×112)
2. `F_body` — face + body (224×224)
3. `F_large_face` — expanded face (112×112)
4. `F_small_face` — tight face crop (112×112)
5. `F_background` — full frame (224×224)
6. `F_body_large` — larger face-body (224×224)
7. `F_down_face` — lower half of face / mouth (112×112)

### Key Modules
- `model/Encoder.py` — Visual (3D-Conv) and audio (MFCC) encoders
- `model/Classifier.py` — Bidirectional GRU for temporal modeling
- `model/models_gnn.py` — `MSSG` class: the main GNN model
- `model/Graph.py` — `GraphASD` with MessagePassing layers
- `model/fusion.py` — Multi-feature fusion (DAF, iAFF, AFF)
- `model/faceDetector/s3fd/` — S3FD face detector
- `dataLoader.py` — Data loading for both encoder training and graph training (`GraphDataset`)
- `loss.py` — Loss functions: `lossAV`, `lossV`, `lossContrast`

### Data Flow
- `local/` — All preprocessing scripts: image cropping, feature extraction, graph construction
- `predata/` — Cached pre-trained models and extracted node features per scale
- `graphs/` — Constructed graph structures (PyTorch Geometric `Data` objects)
- `proposed/` — Student model distillation for real-time inference with `OnlineTracker`

### Training Conventions
- Visual normalization: mean=0.4161, std=0.1688 (grayscale)
- Max speakers per frame: 3 (clamped)
- Graph dropout: p=0.9 on feature fusion, p=0.01 on edges during training
- Model checkpoints saved as `.pt` files; supports multi-model averaging (top-7 checkpoints)
- Handles `module.` prefix stripping for DataParallel compatibility
- NaN/Inf gradient checking before optimizer step
- Preprocessing uses `local/utils/run.pl` for parallel job execution across train/val splits
