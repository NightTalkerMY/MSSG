"""
Multi-seed comprehensive evaluation.
Loads the best model from each seed (1-5), runs the same evaluation as
comprehensive_eval.py, then reports mean +/- std across all seeds.

Sections:
  1. Parameter count (same across seeds)
  2. AVA streaming (Teacher vs Student, mAP + latency) — per-seed + aggregated
  3. Columbia streaming (Teacher vs Student, per-speaker F1 + latency) — per-seed + aggregated
  4. Streaming degradation analysis — per-seed + aggregated
  5. Ablation study — per-seed + aggregated

All results saved to a single JSON file.
"""
import os
import sys
import glob
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ASD import ASD
from model.models_gnn import MSSG
from student_model import CausalStudentASD

# ==========================================
# CONFIG
# ==========================================
PRETRAIN_DIR = r"D:\FYP\MSSG\predata\pretrain_model"
MSSG_WEIGHTS = r"D:\FYP\MSSG\exps\graph_training\graph_2000\chckpoint_best.pt"
VAL_CACHE_DIR = r"D:\FYP\MSSG\realtime\val_cached_chunks"
COL_CACHE_DIR = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
FRAME_WIDTH = 1280
LAST_FRAME = 49

SEEDS = [1, 2, 3, 4, 5]

ENCODER_CONFIGS = [
    {'name': 'face',            'path': 'face_model',             'cache_slice': (0, 128),     'mssg_start': 0},
    {'name': 'face_body',       'path': 'body_model',             'cache_slice': (256, 384),   'mssg_start': 128},
    {'name': 'face_large',      'path': 'face_large_model',       'cache_slice': (384, 512),   'mssg_start': 256},
    {'name': 'background',      'path': 'background_model.model', 'cache_slice': (512, 640),   'mssg_start': 384},
    {'name': 'face_small',      'path': 'face_small_model',       'cache_slice': (640, 768),   'mssg_start': 512},
    {'name': 'face_down',       'path': 'face_down_model',        'cache_slice': (768, 896),   'mssg_start': 640},
    {'name': 'face_body_large', 'path': 'body_large_model',       'cache_slice': (128, 256),   'mssg_start': 768},
]


# ==========================================
# DATASET CLASSES
# ==========================================
class ValCacheDataset(IterableDataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "val_cache_part_*.pt")))

    def __iter__(self):
        for f_path in self.files:
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            for idx in range(num_items):
                yield (
                    data['v'][idx], data['a'][idx], data['t'][idx],
                    data['s'][idx], data['l'][idx], data['m'][idx],
                )


class ColumbiaDataset(IterableDataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "col_cache_part_*.pt")))

    def __iter__(self):
        for f_path in self.files:
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            for idx in range(num_items):
                yield (
                    data['v'][idx], data['a'][idx],
                    data['s'][idx], data['l'][idx], data['m'][idx],
                    data['names'][idx],
                )


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def count_all_params(model):
    return sum(p.numel() for p in model.parameters())


def warmup_gpu(model, device):
    dummy_v = torch.randn(1, 5, 50, 896).to(device)
    dummy_a = torch.randn(1, 5, 50, 128).to(device)
    dummy_s = torch.randn(1, 5, 50, 4).to(device)
    dummy_m = torch.ones(1, 5, 50).to(device)
    for _ in range(10):
        _ = model(dummy_v, dummy_a, dummy_s, dummy_m)
    if device.type == 'cuda':
        torch.cuda.synchronize()


def fmt_mean_std(values, fmt=".2f"):
    """Format mean +/- std string."""
    m = np.mean(values)
    s = np.std(values)
    return f"{m:{fmt}} +/- {s:{fmt}}"


# ==========================================
# EVALUATION FUNCTIONS (student-only — teacher runs once)
# ==========================================
def eval_ava_streaming_student(student_model, val_loader, device):
    """AVA streaming: student only, last-frame mAP + latency."""
    student_scores, student_labels = [], []
    student_times = []

    with torch.no_grad():
        for v, a, t_feat, s_feat, labels, masks in val_loader:
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]

            active_at_last = [p for p in range(5) if masks[p, LAST_FRAME] > 0.5]
            if not active_at_last:
                continue

            if device.type == 'cuda':
                torch.cuda.synchronize()
            s_start = time.perf_counter()

            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)
            logits, _ = student_model(v_in, a_in, s_in, m_in)
            s_probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            student_times.append(time.perf_counter() - s_start)

            for p in active_at_last:
                student_scores.append(s_probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                student_labels.append(0.0 if lbl == 2.0 else lbl)

    student_mAP = average_precision_score(student_labels, student_scores) * 100

    return {
        "student_mAP": round(student_mAP, 4),
        "student_latency_ms": round(np.mean(student_times) * 1000, 4),
    }


def eval_ava_streaming_teacher(bigrus, mssg_model, val_loader, device):
    """AVA streaming: teacher only (run once). Returns teacher mAP + latency."""
    teacher_scores, teacher_labels = [], []
    teacher_times = []

    with torch.no_grad():
        for v, a, t_feat, s_feat, labels, masks in tqdm(val_loader, desc="Teacher AVA Streaming"):
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]

            active_at_last = [p for p in range(5) if masks[p, LAST_FRAME] > 0.5]
            if not active_at_last:
                continue

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            valid_speakers = [p for p in range(5) if masks[p].sum() > 0]
            num_valid = len(valid_speakers)
            feat_av_all = torch.zeros(num_valid, 50, 896)
            for config in ENCODER_CONFIGS:
                cs_start, cs_end = config['cache_slice']
                ms = config['mssg_start']
                feat_v_enc = torch.stack([v[p, :, cs_start:cs_end] for p in valid_speakers])
                feat_a_enc = torch.stack([a[p] for p in valid_speakers])
                bigru_input = (feat_v_enc + feat_a_enc).to(device)
                bigru_output = bigrus[config['name']](bigru_input).cpu()
                feat_av_all[:, :, ms:ms + 128] = bigru_output

            node_features, node_labels_t = [], []
            src_v, tgt_v, e_attr = [], [], []
            active_indices = [(si, p) for si, p in enumerate(valid_speakers) if masks[p, LAST_FRAME] > 0.5]
            num_active = len(active_indices)

            for sp_idx, p in active_indices:
                feat_av = feat_av_all[sp_idx, LAST_FRAME]
                spatial = s_feat[p, LAST_FRAME]
                face_size = (spatial[2] * FRAME_WIDTH).unsqueeze(0)
                speaker_num = torch.tensor([float(num_active)])
                node_features.append(torch.cat([feat_av, face_size, spatial, speaker_num]))
                lbl = labels[p, LAST_FRAME].item()
                node_labels_t.append(0.0 if lbl == 2.0 else lbl)

            for j in range(len(node_features)):
                for k in range(len(node_features)):
                    src_v.append(j); tgt_v.append(k)
                    e_attr.append(111 if j == k else 0)

            x = torch.stack(node_features).to(device)
            edge_index = torch.tensor([src_v, tgt_v], dtype=torch.long).to(device)
            edge_attr_t = torch.tensor(e_attr, dtype=torch.float).to(device)
            y = torch.tensor(node_labels_t, dtype=torch.float).unsqueeze(1).to(device)
            graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_t)
            output = mssg_model(graph_data)
            t_scores = F.softmax(output, dim=-1)[:, 1].cpu().tolist()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            teacher_times.append(time.perf_counter() - t_start)
            teacher_scores.extend(t_scores)
            teacher_labels.extend(node_labels_t)

    teacher_mAP = average_precision_score(teacher_labels, teacher_scores) * 100

    return {
        "teacher_mAP": round(teacher_mAP, 4),
        "teacher_latency_ms": round(np.mean(teacher_times) * 1000, 4),
        "num_samples": len(teacher_labels),
    }


def eval_columbia_streaming_student(student_model, col_loader, device):
    """Columbia streaming: student only, per-speaker F1 + latency."""
    student_preds, student_targets = {}, {}
    student_times = []

    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in col_loader:
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]

            active_at_last = [
                p for p in range(5)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            if not active_at_last:
                continue

            if device.type == 'cuda':
                torch.cuda.synchronize()
            s_start = time.perf_counter()

            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)
            logits, _ = student_model(v_in, a_in, s_in, m_in)
            s_probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            student_times.append(time.perf_counter() - s_start)

            for p in active_at_last:
                name = extracted_names[p]
                student_preds.setdefault(name, []).append(s_probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                student_targets.setdefault(name, []).append(0.0 if lbl == 2.0 else lbl)

    per_speaker = {}
    f1_list = []
    for speaker in sorted(student_preds.keys()):
        s_p = np.array(student_preds[speaker])
        s_t = np.array(student_targets[speaker])
        s_prec, s_rec, _ = precision_recall_curve(s_t, s_p)
        s_f1 = np.max(2 * (s_prec * s_rec) / (s_prec + s_rec + 1e-10)) * 100
        f1_list.append(s_f1)
        per_speaker[speaker] = round(s_f1, 4)

    return {
        "avg_student_F1": round(float(np.mean(f1_list)), 4),
        "student_latency_ms": round(np.mean(student_times) * 1000, 4),
        "per_speaker": per_speaker,
    }


def eval_columbia_streaming_teacher(bigrus, mssg_model, col_loader, device):
    """Columbia streaming: teacher only (run once)."""
    teacher_preds, teacher_targets = {}, {}
    teacher_times = []

    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in tqdm(col_loader, desc="Teacher Columbia Streaming"):
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]

            active_at_last = [
                p for p in range(5)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            if not active_at_last:
                continue

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            valid_speakers = [p for p in range(5) if masks[p].sum() > 0]
            num_valid = len(valid_speakers)
            feat_av_all = torch.zeros(num_valid, 50, 896)
            for config in ENCODER_CONFIGS:
                cs_start, cs_end = config['cache_slice']
                ms = config['mssg_start']
                feat_v_enc = torch.stack([v[p, :, cs_start:cs_end] for p in valid_speakers])
                feat_a_enc = torch.stack([a[p] for p in valid_speakers])
                bigru_input = (feat_v_enc + feat_a_enc).to(device)
                bigru_output = bigrus[config['name']](bigru_input).cpu()
                feat_av_all[:, :, ms:ms + 128] = bigru_output

            node_features, node_labels_t, node_names_t = [], [], []
            src_v, tgt_v, e_attr = [], [], []
            active_indices = [
                (si, p) for si, p in enumerate(valid_speakers)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            num_active = len(active_indices)

            for sp_idx, p in active_indices:
                feat_av = feat_av_all[sp_idx, LAST_FRAME]
                spatial = s_feat[p, LAST_FRAME]
                face_size = (spatial[2] * FRAME_WIDTH).unsqueeze(0)
                speaker_num = torch.tensor([float(num_active)])
                node_features.append(torch.cat([feat_av, face_size, spatial, speaker_num]))
                lbl = labels[p, LAST_FRAME].item()
                node_labels_t.append(0.0 if lbl == 2.0 else lbl)
                node_names_t.append(extracted_names[p])

            if not node_features:
                continue

            for j in range(len(node_features)):
                for k in range(len(node_features)):
                    src_v.append(j); tgt_v.append(k)
                    e_attr.append(111 if j == k else 0)

            x = torch.stack(node_features).to(device)
            edge_index = torch.tensor([src_v, tgt_v], dtype=torch.long).to(device)
            edge_attr_t = torch.tensor(e_attr, dtype=torch.float).to(device)
            y = torch.tensor(node_labels_t, dtype=torch.float).unsqueeze(1).to(device)
            graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_t)
            output = mssg_model(graph_data)
            t_scores = F.softmax(output, dim=-1)[:, 1].cpu().tolist()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            teacher_times.append(time.perf_counter() - t_start)

            for i, name in enumerate(node_names_t):
                teacher_preds.setdefault(name, []).append(t_scores[i])
                teacher_targets.setdefault(name, []).append(node_labels_t[i])

    per_speaker = {}
    f1_list = []
    for speaker in sorted(teacher_preds.keys()):
        t_p = np.array(teacher_preds[speaker])
        t_t = np.array(teacher_targets[speaker])
        t_prec, t_rec, _ = precision_recall_curve(t_t, t_p)
        t_f1 = np.max(2 * (t_prec * t_rec) / (t_prec + t_rec + 1e-10)) * 100
        f1_list.append(t_f1)
        per_speaker[speaker] = round(t_f1, 4)

    return {
        "avg_teacher_F1": round(float(np.mean(f1_list)), 4),
        "teacher_latency_ms": round(np.mean(teacher_times) * 1000, 4),
        "per_speaker": per_speaker,
    }


def eval_student_allframes_mAP(student_model, val_loader, device):
    """Student AVA mAP using ALL frames (non-streaming, for degradation)."""
    all_scores, all_labels = [], []
    with torch.no_grad():
        for v, a, t_feat, s, l, m in val_loader:
            v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
            l = l.to(device)
            logits, _ = student_model(v, a, s, m)
            flat_mask = m.view(-1)
            valid = flat_mask > 0.5
            scores = F.softmax(logits.view(-1, 2)[valid], dim=-1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(l.view(-1)[valid].cpu().tolist())
    return round(average_precision_score(all_labels, all_scores) * 100, 4) if all_labels else 0.0


class AblatedStudent(nn.Module):
    """Wraps CausalStudentASD with component knockouts for ablation."""
    def __init__(self, base_model, ablate=None):
        super().__init__()
        self.model = base_model
        self.ablate = ablate or set()

    def forward(self, v, a, s_cues, mask):
        B, S, T, _ = v.shape
        m = self.model

        scale_feats = []
        for i, proj in enumerate(m.scale_projs):
            scale_v = v[:, :, :, i * m.SCALE_DIM:(i + 1) * m.SCALE_DIM]
            scale_feats.append(proj(scale_v))
        scale_stack = torch.stack(scale_feats, dim=3)

        if 'scale_attn' in self.ablate:
            attn_weights = torch.ones(B, S, T, m.NUM_SCALES, device=v.device) / m.NUM_SCALES
        else:
            attn_weights = F.softmax(m.scale_attn(s_cues), dim=-1)
        v_fused = (scale_stack * attn_weights.unsqueeze(-1)).sum(dim=3)

        x = m.fuse(v_fused) + a
        D = x.shape[-1]
        x_seq = x.view(B * S, T, D)
        gru_out, _ = m.gru(x_seq)

        if 'mha' in self.ablate:
            seq_features = m.temporal_norm(gru_out)
        else:
            causal_mask = torch.triu(torch.ones(T, T, device=v.device) * float('-inf'), diagonal=1)
            attn_out, _ = m.mha(gru_out, gru_out, gru_out, attn_mask=causal_mask)
            seq_features = m.temporal_norm(gru_out + attn_out)

        if 'gated_ffn' not in self.ablate:
            gate = torch.sigmoid(m.gate_proj(seq_features))
            value = torch.tanh(m.value_proj(seq_features))
            seq_features = m.ffn_norm(seq_features + gate * value)

        seq_features = seq_features.view(B, S, T, D)
        seq_features = seq_features * mask.unsqueeze(-1)

        if 'cross_speaker' in self.ablate:
            graph_features = seq_features
        else:
            graph_features = m.cross_speaker(seq_features, mask)

        logits = m.classifier(m.feat_dropout(graph_features))
        return logits, seq_features


def eval_ablation_ava(model_variant, val_loader, device):
    model_variant.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for v, a, t_feat, s, l, m in val_loader:
            v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
            l = l.to(device)
            logits, _ = model_variant(v, a, s, m)
            flat_mask = m[:, :, LAST_FRAME].view(-1)
            valid = flat_mask > 0.5
            last_logits = logits[:, :, LAST_FRAME, :].reshape(-1, 2)[valid]
            last_labels = l[:, :, LAST_FRAME].reshape(-1)[valid]
            scores = F.softmax(last_logits, dim=-1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(last_labels.cpu().tolist())
    return round(average_precision_score(all_labels, all_scores) * 100, 4) if all_labels else 0.0


def eval_ablation_columbia(model_variant, col_loader, device):
    model_variant.eval()
    speaker_preds, speaker_targets = {}, {}
    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in col_loader:
            v, a, s_feat = v[0], a[0], s_feat[0]
            labels, masks = labels[0], masks[0]
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]
            active = [p for p in range(5) if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"]
            if not active:
                continue
            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)
            logits, _ = model_variant(v_in, a_in, s_in, m_in)
            probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)
            for p in active:
                name = extracted_names[p]
                speaker_preds.setdefault(name, []).append(probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                speaker_targets.setdefault(name, []).append(0.0 if lbl == 2.0 else lbl)

    f1_list = []
    for speaker in sorted(speaker_preds.keys()):
        preds = np.array(speaker_preds[speaker])
        targets = np.array(speaker_targets[speaker])
        prec, rec, _ = precision_recall_curve(targets, preds)
        f1_list.append(np.max(2 * (prec * rec) / (prec + rec + 1e-10)) * 100)
    return round(float(np.mean(f1_list)), 4) if f1_list else 0.0


# ==========================================
# MAIN
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running multi-seed comprehensive evaluation on {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seeds": SEEDS,
    }

    # ================================================================
    # LOAD TEACHER (once — deterministic, same across all seeds)
    # ================================================================
    print("Loading teacher (7 BiGRUs + MSSG)...")
    bigrus = {}
    for config in ENCODER_CONFIGS:
        full_path = os.path.join(PRETRAIN_DIR, config['path'])
        s = ASD(encoder_struct=[32, 64, 128])
        if os.path.isdir(full_path):
            s.loadParameters_multi(full_path)
        else:
            s.loadParameters(full_path)
        bigrus[config['name']] = s.model.GRU.to(device).eval()

    mssg_model = MSSG().to(device)
    state_dict = torch.load(MSSG_WEIGHTS, map_location=device)
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    mssg_model.load_state_dict(clean_dict)
    mssg_model.eval()

    # ================================================================
    # SECTION 1: PARAMETER COUNT (same across seeds)
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 1: MODEL COMPLEXITY")
    print(f"{'='*70}")

    student_model_tmp = CausalStudentASD()
    single_bigru_params = count_all_params(list(bigrus.values())[0])
    teacher_bigru_total = single_bigru_params * 7
    mssg_params = count_all_params(mssg_model)
    teacher_total = teacher_bigru_total + mssg_params
    student_total = count_all_params(student_model_tmp)

    results["parameters"] = {
        "teacher_total": teacher_total,
        "student_total": student_total,
        "reduction_ratio": round(teacher_total / student_total, 2),
    }

    print(f"  Teacher: {teacher_total:,} params")
    print(f"  Student: {student_total:,} params")
    print(f"  Reduction: {results['parameters']['reduction_ratio']}x smaller")
    del student_model_tmp

    # ================================================================
    # TEACHER EVAL (once)
    # ================================================================
    print(f"\n{'='*70}")
    print("  TEACHER EVALUATION (run once)")
    print(f"{'='*70}\n")

    val_dataset_t = ValCacheDataset(VAL_CACHE_DIR)
    val_loader_t = DataLoader(val_dataset_t, batch_size=1, num_workers=0)
    teacher_ava = eval_ava_streaming_teacher(bigrus, mssg_model, val_loader_t, device)

    col_dataset_t = ColumbiaDataset(COL_CACHE_DIR)
    col_loader_t = DataLoader(col_dataset_t, batch_size=1, num_workers=0)
    teacher_col = eval_columbia_streaming_teacher(bigrus, mssg_model, col_loader_t, device)

    results["teacher"] = {
        "ava_streaming_mAP": teacher_ava["teacher_mAP"],
        "ava_latency_ms": teacher_ava["teacher_latency_ms"],
        "columbia_avg_F1": teacher_col["avg_teacher_F1"],
        "columbia_latency_ms": teacher_col["teacher_latency_ms"],
        "columbia_per_speaker": teacher_col["per_speaker"],
    }

    print(f"  Teacher AVA mAP:     {teacher_ava['teacher_mAP']:.2f}%")
    print(f"  Teacher Columbia F1: {teacher_col['avg_teacher_F1']:.2f}%")

    teacher_offline = 95.5201  # From original MSSG evaluation

    # ================================================================
    # PER-SEED STUDENT EVALUATION
    # ================================================================
    seed_results = {}
    all_ava_mAP = []
    all_ava_latency = []
    all_col_f1 = []
    all_col_latency = []
    all_allframes_mAP = []
    all_col_per_speaker = {}

    # Ablation collectors
    ablation_configs = [
        ('Full Model',             set()),
        ('w/o Scale Attention',    {'scale_attn'}),
        ('w/o Buffer MHA',        {'mha'}),
        ('w/o Gated FFN',         {'gated_ffn'}),
        ('w/o Cross-Speaker Attn', {'cross_speaker'}),
    ]
    all_ablation = {name: {"ava": [], "col": []} for name, _ in ablation_configs}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        weights_path = rf"D:\FYP\MSSG\realtime\realtime_epoch_{seed}\holopi_student_best.pt"
        if not os.path.exists(weights_path):
            print(f"  WARNING: {weights_path} not found, skipping seed {seed}")
            continue

        student_model = CausalStudentASD().to(device)
        student_model.load_state_dict(torch.load(weights_path, map_location=device))
        student_model.eval()
        warmup_gpu(student_model, device)

        # --- AVA Streaming ---
        print(f"  Evaluating AVA streaming...")
        vd = ValCacheDataset(VAL_CACHE_DIR)
        vl = DataLoader(vd, batch_size=1, num_workers=0)
        ava_res = eval_ava_streaming_student(student_model, vl, device)

        all_ava_mAP.append(ava_res["student_mAP"])
        all_ava_latency.append(ava_res["student_latency_ms"])
        print(f"  AVA mAP: {ava_res['student_mAP']:.2f}%")

        # --- Columbia Streaming ---
        print(f"  Evaluating Columbia streaming...")
        cd = ColumbiaDataset(COL_CACHE_DIR)
        cl = DataLoader(cd, batch_size=1, num_workers=0)
        col_res = eval_columbia_streaming_student(student_model, cl, device)

        all_col_f1.append(col_res["avg_student_F1"])
        all_col_latency.append(col_res["student_latency_ms"])
        for speaker, f1 in col_res["per_speaker"].items():
            all_col_per_speaker.setdefault(speaker, []).append(f1)
        print(f"  Columbia F1: {col_res['avg_student_F1']:.2f}%")

        # --- All-frames (degradation) ---
        print(f"  Evaluating all-frames mAP...")
        vd2 = ValCacheDataset(VAL_CACHE_DIR)
        vl2 = DataLoader(vd2, batch_size=32, num_workers=0)
        allframes = eval_student_allframes_mAP(student_model, vl2, device)
        all_allframes_mAP.append(allframes)
        print(f"  All-frames mAP: {allframes:.2f}%")

        # --- Ablation ---
        print(f"  Running ablation study...")
        seed_ablation = {}
        for abl_name, ablate_set in ablation_configs:
            ablated = AblatedStudent(student_model, ablate=ablate_set).to(device)
            ablated.eval()

            vd3 = ValCacheDataset(VAL_CACHE_DIR)
            vl3 = DataLoader(vd3, batch_size=32, num_workers=0)
            abl_ava = eval_ablation_ava(ablated, vl3, device)

            cd3 = ColumbiaDataset(COL_CACHE_DIR)
            cl3 = DataLoader(cd3, batch_size=1, num_workers=0)
            abl_col = eval_ablation_columbia(ablated, cl3, device)

            seed_ablation[abl_name] = {"ava_mAP": abl_ava, "col_F1": abl_col}
            all_ablation[abl_name]["ava"].append(abl_ava)
            all_ablation[abl_name]["col"].append(abl_col)

        seed_results[seed] = {
            "ava_mAP": ava_res["student_mAP"],
            "ava_latency_ms": ava_res["student_latency_ms"],
            "col_F1": col_res["avg_student_F1"],
            "col_latency_ms": col_res["student_latency_ms"],
            "col_per_speaker": col_res["per_speaker"],
            "allframes_mAP": allframes,
            "ablation": seed_ablation,
        }

        del student_model
        torch.cuda.empty_cache()

    # ================================================================
    # AGGREGATED RESULTS
    # ================================================================
    num_seeds = len(all_ava_mAP)
    if num_seeds == 0:
        print("\nERROR: No seed models found. Train first with train_student_seeded.py")
        return

    results["per_seed"] = seed_results

    # --- AVA Summary ---
    print(f"\n{'='*70}")
    print("  AGGREGATED RESULTS (mean +/- std)")
    print(f"{'='*70}")

    print(f"\n  AVA Streaming (n={num_seeds} seeds):")
    print(f"  {'Metric':<25} | {'Teacher':>15} | {'Student':>20}")
    print(f"  {'-'*65}")
    print(f"  {'mAP (%)':<25} | {teacher_ava['teacher_mAP']:>14.2f}% | {fmt_mean_std(all_ava_mAP)}%")
    print(f"  {'Latency (ms/sample)':<25} | {teacher_ava['teacher_latency_ms']:>15.2f} | {fmt_mean_std(all_ava_latency)}")

    results["aggregated_ava"] = {
        "teacher_mAP": teacher_ava["teacher_mAP"],
        "student_mAP_mean": round(np.mean(all_ava_mAP), 4),
        "student_mAP_std": round(np.std(all_ava_mAP), 4),
        "student_latency_mean": round(np.mean(all_ava_latency), 4),
        "student_latency_std": round(np.std(all_ava_latency), 4),
    }

    # --- Columbia Summary ---
    print(f"\n  Columbia Streaming (n={num_seeds} seeds):")
    print(f"  {'Speaker':<15} | {'Teacher F1':>12} | {'Student F1 (mean +/- std)':>25}")
    print(f"  {'-'*58}")
    for speaker in sorted(all_col_per_speaker.keys()):
        t_f1 = teacher_col["per_speaker"][speaker]
        s_vals = all_col_per_speaker[speaker]
        print(f"  {speaker:<15} | {t_f1:>11.2f}% | {fmt_mean_std(s_vals)}%")
    print(f"  {'-'*58}")
    print(f"  {'AVERAGE':<15} | {teacher_col['avg_teacher_F1']:>11.2f}% | {fmt_mean_std(all_col_f1)}%")
    print(f"\n  {'Latency (ms/sample)':<25} | {teacher_col['teacher_latency_ms']:>15.2f} | {fmt_mean_std(all_col_latency)}")

    col_per_speaker_agg = {}
    for speaker in sorted(all_col_per_speaker.keys()):
        vals = all_col_per_speaker[speaker]
        col_per_speaker_agg[speaker] = {
            "teacher_F1": teacher_col["per_speaker"][speaker],
            "student_F1_mean": round(np.mean(vals), 4),
            "student_F1_std": round(np.std(vals), 4),
        }

    results["aggregated_columbia"] = {
        "teacher_avg_F1": teacher_col["avg_teacher_F1"],
        "student_avg_F1_mean": round(np.mean(all_col_f1), 4),
        "student_avg_F1_std": round(np.std(all_col_f1), 4),
        "student_latency_mean": round(np.mean(all_col_latency), 4),
        "student_latency_std": round(np.std(all_col_latency), 4),
        "per_speaker": col_per_speaker_agg,
    }

    # --- Degradation Summary ---
    print(f"\n  Streaming Degradation (n={num_seeds} seeds):")
    print(f"  {'Configuration':<45} | {'AVA mAP':>20}")
    print(f"  {'-'*70}")
    print(f"  {'Teacher: Offline (2000-node, all frames)':<45} | {teacher_offline:>19.2f}%")
    print(f"  {'Teacher: Streaming (last-frame only)':<45} | {teacher_ava['teacher_mAP']:>19.2f}%")
    print(f"  {'Student: All frames':<45} | {fmt_mean_std(all_allframes_mAP):>20}%")
    print(f"  {'Student: Streaming (last-frame only)':<45} | {fmt_mean_std(all_ava_mAP):>20}%")

    results["aggregated_degradation"] = {
        "teacher_offline_mAP": teacher_offline,
        "teacher_streaming_mAP": teacher_ava["teacher_mAP"],
        "student_allframes_mean": round(np.mean(all_allframes_mAP), 4),
        "student_allframes_std": round(np.std(all_allframes_mAP), 4),
        "student_streaming_mean": round(np.mean(all_ava_mAP), 4),
        "student_streaming_std": round(np.std(all_ava_mAP), 4),
    }

    # --- Ablation Summary ---
    print(f"\n  Ablation Study (n={num_seeds} seeds):")
    print(f"  {'Configuration':<25} | {'AVA mAP (mean +/- std)':>25} | {'Col F1 (mean +/- std)':>25}")
    print(f"  {'-'*80}")

    ablation_agg = []
    for abl_name, _ in ablation_configs:
        ava_vals = all_ablation[abl_name]["ava"]
        col_vals = all_ablation[abl_name]["col"]
        print(f"  {abl_name:<25} | {fmt_mean_std(ava_vals):>24}% | {fmt_mean_std(col_vals):>24}%")
        ablation_agg.append({
            "name": abl_name,
            "ava_mAP_mean": round(np.mean(ava_vals), 4),
            "ava_mAP_std": round(np.std(ava_vals), 4),
            "col_F1_mean": round(np.mean(col_vals), 4),
            "col_F1_std": round(np.std(col_vals), 4),
        })

    results["aggregated_ablation"] = ablation_agg

    # ================================================================
    # SAVE
    # ================================================================
    result_path = r"D:\FYP\MSSG\realtime\comprehensive_eval_seeded_result.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  All results saved to: {result_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
