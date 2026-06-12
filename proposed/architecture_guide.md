# HoloPi Causal Student ASD — Architecture Drawing Guide

Reference: `proposed/student_model.py` (CausalStudentASD) + `proposed/train_student.py`

---

## Overall Layout

Four sections left-to-right. Section D is training-only (dashed border).

| Section | Title | Color |
|---------|-------|-------|
| A | Multimodal Feature Encoding | Blue |
| B | Causal Temporal Modeling | Green |
| C | Teacher Emulation & Spatial Context | Orange/Yellow |
| D | Dual-Objective Knowledge Distillation | Pink/Red (dashed) |

---

## Inputs (3 tensors entering Section A from the left)

| Input | Shape | Description |
|-------|-------|-------------|
| V | [B, S, T, 896] | 7 concatenated 128-dim visual scale features |
| A | [B, S, T, 128] | Audio embedding (MFCC-based) |
| S_cues | [B, S, T, 4] | Face bounding box [cx, cy, w, h] normalized |

B = batch, S = 5 speakers (max), T = 50 frames, D = 128

---

## Section A: Multimodal Feature Encoding

```
V [B,S,T,896]
    |
    |--- Split into 7 slices of 128-dim each
    |
    v
7x [ Linear(128, 64) + ReLU + Dropout(0.1) ]       per-scale projection
    |
    v
Stack -> [B, S, T, 7, 64]                            7 projected scales
    |
    |    S_cues [B,S,T,4]
    |        |
    |        v
    |    Linear(4, 32) + ReLU + Linear(32, 7)
    |        |
    |        v
    |    Softmax(dim=-1) -> attn_weights [B,S,T,7]    spatial attention weights
    |        |
    v        v
Weighted sum: (stack * attn_weights.unsqueeze(-1)).sum(dim=3)
    |
    v
v_fused [B, S, T, 64]                                spatially-weighted visual
    |
    v
Linear(64, 128) + ReLU                               fusion layer
    |
    v
(+) <-------- A [B,S,T,128]                          ELEMENT-WISE ADDITION
    |
    v
X [B, S, T, 128]                                     fused multimodal features
```

### Drawing Notes
- Show V splitting into 7 colored strips (one per visual scale)
- Each strip passes through its own projection block (label "x7" or "N=7")
- S_cues feeds a small "Spatial Attention" sub-block producing 7 softmax weights
- Show the weighted sum merging 7 strips into one
- Audio A shown as a separate block, merged via a (+) circle (element-wise add)
- Output arrow labeled X flows right into Section B

---

## Section B: Causal Temporal Modeling

```
X [B, S, T, 128]
    |
    v
Reshape -> [B*S, T, 128]                  each speaker processed independently
    |
    v
+---------------------------+
| 2-Layer Causal GRU         |
| hidden=128, dropout=0.1    |
| bidirectional=False         |
+---------------------------+
    |
    | gru_out [B*S, T, 128]
    |
    |-----(skip)---------------------------+
    |                                      |
    v                                      |
+---------------------------+              |
| Temporal MHA               |             |
| 4 heads, dropout=0.1       |             |
| causal_mask (upper-tri inf) |             |
+---------------------------+              |
    |                                      |
    | attn_out [B*S, T, 128]               |
    |                                      |
    v                                      v
  (+)  <-----------------------------------+    RESIDUAL: gru_out + attn_out
    |
    v
LayerNorm(128)
    |
    v
seq_features [B*S, T, 128]
```

### Drawing Notes
- Show the reshape step (speakers become independent sequences)
- Draw the causal mask as a small triangular matrix icon beside the MHA block
- Draw a skip arrow from GRU output around MHA to the (+) addition node
- LayerNorm sits after the residual addition

---

## Section C: Teacher Emulation & Spatial Context

Two sub-parts: **C1** (Gated FFN) and **C2** (Cross-Speaker Attention).

### C1: Gated FFN — Future-State Emulation

Mimics the teacher's backward GRU as a causal gated feedforward transform.

```
seq_features [B*S, T, 128]
    |
    |-----(skip)-------------------------------------------+
    |                                                      |
    +---> Linear(128, 128) --> Sigmoid --> gate             |
    |                                                      |
    +---> Linear(128, 128) --> Tanh ----> value             |
    |                                                      |
    |     gate (*) value   (element-wise multiply)         |
    |         |                                            |
    v         v                                            v
        gate * value ------------> (+) <-------------------+
                                    |       RESIDUAL: seq_features + gate*value
                                    v
                                LayerNorm(128)
                                    |
                                    v
                                [B*S, T, 128]
                                    |
                                    v
                              Reshape -> [B, S, T, 128]
                                    |
                                    v
                              (*) <--- mask [B,S,T,1]     GHOST MASKING
                                    |                      zeros out ghost speakers
                                    v
                              seq_features [B, S, T, 128]
                                    |
                        +-----------+-----------+
                        |                       |
                        v                       v
                   To Section C2           To L_MSE (Section D)
                 (Cross-Speaker)         (feature distillation target)
```

### Drawing Notes
- Show two parallel branches: gate (sigmoid) and value (tanh)
- They merge via element-wise multiply
- Draw skip arrow from input around both branches to (+)
- After LayerNorm: reshape back to [B,S,T,128]
- Show ghost masking as an explicit multiply with binary mask
- seq_features BRANCHES here: one path right to C2, one path up/out to L_MSE in Section D

---

### C2: Cross-Speaker Attention

Replaces the teacher's graph neural network. At each timestep, speakers attend to each other.

```
seq_features [B, S, T, 128]
    |
    v
Permute(0,2,1,3) + Reshape -> [B*T, S, 128]     each timestep is a "meeting room"
    |
    |    mask [B, S, T]
    |        |
    |        v
    |    Permute(0,2,1) -> [B*T, S]
    |        |
    |        v
    |    key_padding_mask = (mask < 0.5)          True = ghost = ignore
    |        |
    |        v
    |    Safety: if ALL speakers ghost, unmask all (prevent NaN)
    |        |
    |-----(skip)---------------------------+
    |                                      |
    v                                      |
+-------------------------------+          |
| Cross-Speaker MHA              |         |
| 4 heads, dropout=0.1           |         |
| key_padding_mask (ghost mask)   |         |
| NO causal mask (all speakers    |         |
|   see each other at same time)  |         |
+-------------------------------+          |
    |                                      |
    | attn_out [B*T, S, 128]               |
    |                                      |
    v                                      v
  (+)  <-----------------------------------+    RESIDUAL: input + attn_out
    |
    v
LayerNorm(128)
    |
    v
Reshape back: [B*T, S, 128] -> [B, T, S, 128] -> Permute(0,2,1,3)
    |
    v
graph_features [B, S, T, 128]
```

### Drawing Notes
- Label the reshape as "R(B x S x T -> B*T x S)" — timesteps become batch dim
- Show mask flowing in as key_padding_mask with label "Binary Mask {0,1}"
- Draw skip arrow from input around MHA to (+)
- Note: NO causal mask here (unlike Section B) — speakers attend freely to each other
- Output flows right into Section D (classifier)

---

## Section D: Dual-Objective Knowledge Distillation (Training Only)

Draw entire section with **dashed border** — removed during inference.

### Classifier

```
graph_features [B, S, T, 128]
    |
    v
Dropout(0.1)
    |
    v
Linear(128, 2)                              single linear layer, NOT MLP
    |
    v
logits [B, S, T, 2]  -------> L_CE
```

### Loss Components

| Loss | Weight | Student Source | Teacher/Target | Formula |
|------|--------|--------------|----------------|---------|
| L_CE | 0.4 | logits after classifier | Ground truth labels {0,1} | CrossEntropy with label_smoothing=0.1 |
| L_MSE | 0.6 | seq_features (after C1 GatedFFN, before C2 CrossSpeaker) | Frozen teacher BiGRU features [B,S,T,128] | MSE per-element |

### Total Loss

```
L_total = 0.4 * L_CE + 0.6 * L_MSE
```

### Drawing Notes
- Show "Frozen Teacher Model (Offline)" as a separate box with a snowflake/lock icon
- L_MSE arrow: from seq_features (branching out of C1) up to the frozen teacher box
- L_CE arrow: from logits down to ground truth labels
- The frozen teacher provides target features only — no gradient flows back through it
- Classifier is a single Linear(128, 2) with Dropout before it

---

## Training Configuration (for figure caption or footnote)

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| LR Schedule | CosineAnnealingWarmRestarts (T_0=10, T_mult=1, eta_min=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Batch size (train / val) | 16 / 32 |
| Epochs | 50 |
| Label smoothing | 0.1 |
| Model selection | Best of (0.4 x AVA_mAP + 0.6 x Columbia_F1) |
| Loss mode | CE + MSE |
| Loss weights | alpha=0.4 (CE), 1-alpha=0.6 (MSE) |

---

## All Residual Connections (3 total — draw as skip/bypass arrows)

| Location | From | Bypasses | Into |
|----------|------|----------|------|
| Section B | GRU output | Temporal MHA | (+) then LayerNorm |
| Section C1 | seq_features | gate*value (Gated FFN) | (+) then LayerNorm |
| Section C2 | x_cross input | Cross-Speaker MHA | (+) then LayerNorm |

---

## Parameter Count per Component

| Component | Module | Approx. Params |
|-----------|--------|---------------|
| 7x Scale Projections | `scale_projs` | ~57K |
| Scale Attention | `scale_attn` | ~400 |
| Fusion Layer | `fuse` | ~8K |
| 2-Layer Causal GRU | `gru` | ~198K |
| Temporal MHA + LayerNorm | `mha` + `temporal_norm` | ~66K |
| Gated FFN + LayerNorm | `gate_proj` + `value_proj` + `ffn_norm` | ~33K |
| Cross-Speaker Attn + LayerNorm | `cross_speaker` | ~66K |
| Classifier | `classifier` | ~260 |

Run `comprehensive_eval.py` Section 1 for exact counts.

---

## End-to-End Tensor Shape Flow

```
V [B,5,50,896]  --+
A [B,5,50,128]  --+--> Section A --> X [B,5,50,128]
S [B,5,50,4]    --+                       |
                                          v
                                    Section B --> seq [B*5,50,128]
                                                      |
                                                      v
                                               Section C1 (GatedFFN)
                                                      |
                                          +-----------+-----------+
                                          |                       |
                                     C2 (CrossSpeaker)      seq_features
                                          |               (to L_MSE loss)
                                          v
                                    graph [B,5,50,128]
                                          |
                                          v
                                    Dropout + Linear
                                          |
                                          v
                                    logits [B,5,50,2]
                                     (to L_CE loss)
```
