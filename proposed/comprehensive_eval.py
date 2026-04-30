"""
Comprehensive evaluation super-script for FYP report.
Runs ALL evaluations from scratch — no hardcoded values.

Sections:
  1. Parameter count comparison (Teacher vs Student)
  2. AVA streaming evaluation (Teacher vs Student, mAP + latency)
  3. Columbia streaming evaluation (Teacher vs Student, per-speaker F1 + latency)
  4. Streaming degradation analysis
  5. Ablation study (component knockout)
  6. CPU latency benchmark

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
STUDENT_WEIGHTS = r"D:\FYP\MSSG\realtime\realtime_epoch\holopi_student_best.pt"
VAL_CACHE_DIR = r"D:\FYP\MSSG\realtime\val_cached_chunks"
COL_CACHE_DIR = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
FRAME_WIDTH = 1280
LAST_FRAME = 49

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
        print(f"Found {len(self.files)} val cache files.")

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
        print(f"Found {len(self.files)} Columbia cache files.")

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


def warmup_gpu(student_model, device):
    dummy_v = torch.randn(1, 5, 50, 896).to(device)
    dummy_a = torch.randn(1, 5, 50, 128).to(device)
    dummy_s = torch.randn(1, 5, 50, 4).to(device)
    dummy_m = torch.ones(1, 5, 50).to(device)
    for _ in range(10):
        _ = student_model(dummy_v, dummy_a, dummy_s, dummy_m)
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ==========================================
# EVALUATION FUNCTIONS
# ==========================================
def eval_ava_streaming(bigrus, mssg_model, student_model, val_loader, device):
    """AVA streaming evaluation — teacher + student, mAP + latency."""
    teacher_scores, teacher_labels = [], []
    student_scores, student_labels = [], []
    teacher_times, student_times = [], []

    with torch.no_grad():
        for v, a, t_feat, s_feat, labels, masks in tqdm(val_loader, desc="AVA Streaming"):
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]

            active_at_last = [p for p in range(5) if masks[p, LAST_FRAME] > 0.5]
            if not active_at_last:
                continue

            # --- TEACHER ---
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

            # --- STUDENT ---
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

    teacher_mAP = average_precision_score(teacher_labels, teacher_scores) * 100
    student_mAP = average_precision_score(student_labels, student_scores) * 100

    return {
        "teacher_mAP": round(teacher_mAP, 4),
        "student_mAP": round(student_mAP, 4),
        "diff_mAP": round(student_mAP - teacher_mAP, 4),
        "teacher_latency_ms": round(np.mean(teacher_times) * 1000, 4),
        "student_latency_ms": round(np.mean(student_times) * 1000, 4),
        "speedup": round(np.mean(teacher_times) / np.mean(student_times), 2),
        "num_samples": len(teacher_labels),
    }


def eval_columbia_streaming(bigrus, mssg_model, student_model, col_loader, device):
    """Columbia streaming evaluation — teacher + student, per-speaker F1 + latency."""
    teacher_preds, teacher_targets = {}, {}
    student_preds, student_targets = {}, {}
    teacher_times, student_times = [], []

    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in tqdm(col_loader, desc="Columbia Streaming"):
            v, a, s_feat, labels, masks = v[0], a[0], s_feat[0], labels[0], masks[0]
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]

            active_at_last = [
                p for p in range(5)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            if not active_at_last:
                continue

            # --- TEACHER ---
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

            # --- STUDENT ---
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

    # Compute per-speaker F1
    all_speakers = sorted(set(list(teacher_preds.keys()) + list(student_preds.keys())))
    per_speaker = {}
    teacher_f1_list, student_f1_list = [], []

    for speaker in all_speakers:
        t_p, t_t = np.array(teacher_preds[speaker]), np.array(teacher_targets[speaker])
        t_prec, t_rec, _ = precision_recall_curve(t_t, t_p)
        t_f1 = np.max(2 * (t_prec * t_rec) / (t_prec + t_rec + 1e-10)) * 100

        s_p, s_t = np.array(student_preds[speaker]), np.array(student_targets[speaker])
        s_prec, s_rec, _ = precision_recall_curve(s_t, s_p)
        s_f1 = np.max(2 * (s_prec * s_rec) / (s_prec + s_rec + 1e-10)) * 100

        teacher_f1_list.append(t_f1)
        student_f1_list.append(s_f1)
        per_speaker[speaker] = {
            "teacher_F1": round(t_f1, 4),
            "student_F1": round(s_f1, 4),
            "diff_F1": round(s_f1 - t_f1, 4),
            "num_samples": len(teacher_preds[speaker]),
        }

    teacher_avg = float(np.mean(teacher_f1_list))
    student_avg = float(np.mean(student_f1_list))

    return {
        "avg_teacher_F1": round(teacher_avg, 4),
        "avg_student_F1": round(student_avg, 4),
        "diff_F1": round(student_avg - teacher_avg, 4),
        "teacher_latency_ms": round(np.mean(teacher_times) * 1000, 4),
        "student_latency_ms": round(np.mean(student_times) * 1000, 4),
        "speedup": round(np.mean(teacher_times) / np.mean(student_times), 2),
        "per_speaker": per_speaker,
    }


def eval_student_allframes_mAP(student_model, val_loader, device):
    """Student AVA mAP using ALL frames (non-streaming, for degradation comparison)."""
    all_scores, all_labels = [], []
    with torch.no_grad():
        for v, a, t_feat, s, l, m in tqdm(val_loader, desc="Student All-Frames"):
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
    """Evaluate a model variant on AVA streaming (last-frame mAP)."""
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
    """Evaluate a model variant on Columbia streaming F1."""
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
    print(f"Running comprehensive evaluation on {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "device": str(device)}

    # ================================================================
    # LOAD MODELS
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

    print("Loading student...")
    student_model = CausalStudentASD().to(device)
    student_model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device))
    student_model.eval()

    # Load data
    val_dataset = ValCacheDataset(VAL_CACHE_DIR)
    val_loader_b1 = DataLoader(val_dataset, batch_size=1, num_workers=0)

    col_dataset = ColumbiaDataset(COL_CACHE_DIR)
    col_loader = DataLoader(col_dataset, batch_size=1, num_workers=0)

    warmup_gpu(student_model, device)

    # ================================================================
    # SECTION 1: PARAMETER COUNT
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 1: MODEL COMPLEXITY")
    print(f"{'='*70}")

    single_bigru_params = count_all_params(list(bigrus.values())[0])
    teacher_bigru_total = single_bigru_params * 7
    mssg_params = count_all_params(mssg_model)
    teacher_total = teacher_bigru_total + mssg_params
    student_total = count_all_params(student_model)

    student_breakdown = {
        "scale_projections_7x": sum(count_all_params(p) for p in student_model.scale_projs),
        "scale_attention": count_all_params(student_model.scale_attn),
        "fusion_layer": count_all_params(student_model.fuse),
        "causal_gru_2layer": count_all_params(student_model.gru),
        "buffer_mha": count_all_params(student_model.mha) + count_all_params(student_model.temporal_norm),
        "gated_ffn": count_all_params(student_model.gate_proj) + count_all_params(student_model.value_proj) + count_all_params(student_model.ffn_norm),
        "cross_speaker_attn": count_all_params(student_model.cross_speaker),
        "classifier": count_all_params(student_model.classifier),
    }

    results["parameters"] = {
        "teacher_7_bigru": teacher_bigru_total,
        "teacher_mssg": mssg_params,
        "teacher_total": teacher_total,
        "student_total": student_total,
        "student_breakdown": student_breakdown,
        "reduction_ratio": round(teacher_total / student_total, 2),
    }

    print(f"\n  {'Component':<30} | {'Parameters':>12}")
    print(f"  {'-'*48}")
    print(f"  {'Teacher: 7x BiGRU':<30} | {teacher_bigru_total:>12,}")
    print(f"  {'Teacher: MSSG graph':<30} | {mssg_params:>12,}")
    print(f"  {'Teacher TOTAL':<30} | {teacher_total:>12,}")
    print(f"  {'-'*48}")
    for name, count in student_breakdown.items():
        print(f"  {'  ' + name:<30} | {count:>12,}")
    print(f"  {'Student TOTAL':<30} | {student_total:>12,}")
    print(f"  {'-'*48}")
    print(f"  Reduction: {results['parameters']['reduction_ratio']}x smaller")

    # ================================================================
    # SECTION 2: AVA STREAMING
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 2: AVA STREAMING EVALUATION")
    print(f"{'='*70}\n")

    ava_result = eval_ava_streaming(bigrus, mssg_model, student_model, val_loader_b1, device)
    results["ava_streaming"] = ava_result

    print(f"\n  {'Metric':<25} | {'Teacher':>12} | {'Student':>12} | {'Diff':>12}")
    print(f"  {'-'*65}")
    print(f"  {'mAP (%)':<25} | {ava_result['teacher_mAP']:>11.2f}% | {ava_result['student_mAP']:>11.2f}% | {ava_result['diff_mAP']:>+11.2f}%")
    print(f"  {'Latency (ms/sample)':<25} | {ava_result['teacher_latency_ms']:>12.2f} | {ava_result['student_latency_ms']:>12.2f} | {ava_result['speedup']}x faster")

    # ================================================================
    # SECTION 3: COLUMBIA STREAMING
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 3: COLUMBIA STREAMING EVALUATION")
    print(f"{'='*70}\n")

    col_result = eval_columbia_streaming(bigrus, mssg_model, student_model, col_loader, device)
    results["columbia_streaming"] = col_result

    print(f"  {'Speaker':<15} | {'Teacher F1':>12} | {'Student F1':>12} | {'Diff':>10}")
    print(f"  {'-'*55}")
    for speaker, data in col_result['per_speaker'].items():
        print(f"  {speaker:<15} | {data['teacher_F1']:>11.2f}% | {data['student_F1']:>11.2f}% | {data['diff_F1']:>+9.2f}%")
    print(f"  {'-'*55}")
    print(f"  {'AVERAGE':<15} | {col_result['avg_teacher_F1']:>11.2f}% | {col_result['avg_student_F1']:>11.2f}% | {col_result['diff_F1']:>+9.2f}%")
    print(f"\n  Latency: Teacher {col_result['teacher_latency_ms']:.2f}ms | Student {col_result['student_latency_ms']:.2f}ms | {col_result['speedup']}x faster")

    # ================================================================
    # SECTION 4: STREAMING DEGRADATION
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 4: STREAMING DEGRADATION ANALYSIS")
    print(f"{'='*70}")

    # Student all-frames mAP
    print("\n  Computing student all-frames mAP...")
    val_dataset2 = ValCacheDataset(VAL_CACHE_DIR)
    val_loader_b32 = DataLoader(val_dataset2, batch_size=32, num_workers=0)
    student_allframes = eval_student_allframes_mAP(student_model, val_loader_b32, device)

    # Teacher offline is from the original MSSG paper/training (2000-node graph, all frames)
    # We compute teacher streaming live above
    teacher_offline = 95.5201  # From original MSSG evaluation — the only hardcoded reference value

    teacher_degradation = round(ava_result['teacher_mAP'] - teacher_offline, 4)
    student_recovery = round(ava_result['student_mAP'] - ava_result['teacher_mAP'], 4)

    results["degradation"] = {
        "teacher_offline_mAP": teacher_offline,
        "teacher_streaming_mAP": ava_result['teacher_mAP'],
        "teacher_degradation": teacher_degradation,
        "student_allframes_mAP": student_allframes,
        "student_streaming_mAP": ava_result['student_mAP'],
        "student_vs_teacher_streaming": student_recovery,
    }

    print(f"\n  {'Configuration':<45} | {'AVA mAP':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Teacher: Offline (2000-node, all frames)':<45} | {teacher_offline:>9.2f}%")
    print(f"  {'Teacher: Streaming (last-frame only)':<45} | {ava_result['teacher_mAP']:>9.2f}%")
    print(f"  {'Student: All frames':<45} | {student_allframes:>9.2f}%")
    print(f"  {'Student: Streaming (last-frame only)':<45} | {ava_result['student_mAP']:>9.2f}%")
    print(f"  {'-'*60}")
    print(f"  {'Teacher offline → streaming degradation':<45} | {teacher_degradation:>+9.2f}%")
    print(f"  {'Student advantage over teacher (streaming)':<45} | {student_recovery:>+9.2f}%")

    # ================================================================
    # SECTION 5: ABLATION STUDY
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 5: ABLATION STUDY (Component Knockout)")
    print(f"{'='*70}")
    print("  Removing one component at a time from the trained model.\n")

    ablation_configs = [
        ('Full Model',             set()),
        ('w/o Scale Attention',    {'scale_attn'}),
        ('w/o Buffer MHA',        {'mha'}),
        ('w/o Gated FFN',         {'gated_ffn'}),
        ('w/o Cross-Speaker Attn', {'cross_speaker'}),
    ]

    # Reload data for ablation
    val_dataset3 = ValCacheDataset(VAL_CACHE_DIR)
    val_loader_abl = DataLoader(val_dataset3, batch_size=32, num_workers=0)
    col_dataset2 = ColumbiaDataset(COL_CACHE_DIR)
    col_loader2 = DataLoader(col_dataset2, batch_size=1, num_workers=0)

    ablation_results = []
    for name, ablate_set in ablation_configs:
        print(f"  Evaluating: {name}...")
        ablated = AblatedStudent(student_model, ablate=ablate_set).to(device)
        ablated.eval()

        # Reload datasets for each ablation (IterableDataset exhausts)
        vd = ValCacheDataset(VAL_CACHE_DIR)
        vl = DataLoader(vd, batch_size=32, num_workers=0)
        cd = ColumbiaDataset(COL_CACHE_DIR)
        cl = DataLoader(cd, batch_size=1, num_workers=0)

        ava_mAP = eval_ablation_ava(ablated, vl, device)
        col_f1 = eval_ablation_columbia(ablated, cl, device)
        ablation_results.append({"name": name, "ava_mAP": ava_mAP, "col_F1": col_f1})

    results["ablation"] = ablation_results

    baseline_ava = ablation_results[0]['ava_mAP']
    baseline_col = ablation_results[0]['col_F1']

    print(f"\n  {'Configuration':<25} | {'AVA mAP':>10} | {'Δ AVA':>8} | {'Col F1':>10} | {'Δ Col':>8}")
    print(f"  {'-'*70}")
    for r in ablation_results:
        d_ava = r['ava_mAP'] - baseline_ava
        d_col = r['col_F1'] - baseline_col
        if r['name'] == 'Full Model':
            print(f"  {r['name']:<25} | {r['ava_mAP']:>9.2f}% | {'---':>8} | {r['col_F1']:>9.2f}% | {'---':>8}")
        else:
            print(f"  {r['name']:<25} | {r['ava_mAP']:>9.2f}% | {d_ava:>+7.2f}% | {r['col_F1']:>9.2f}% | {d_col:>+7.2f}%")

    # ================================================================
    # SECTION 6: CPU LATENCY
    # ================================================================
    print(f"\n{'='*70}")
    print("  SECTION 6: CPU LATENCY BENCHMARK")
    print(f"{'='*70}")

    print("\n  Measuring student CPU inference latency...")
    student_cpu = CausalStudentASD()
    student_cpu.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location='cpu'))
    student_cpu.eval()

    dummy = (torch.randn(1, 5, 50, 896), torch.randn(1, 5, 50, 128),
             torch.randn(1, 5, 50, 4), torch.ones(1, 5, 50))
    for _ in range(5):
        _ = student_cpu(*dummy)

    cpu_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = student_cpu(*dummy)
            cpu_times.append(time.perf_counter() - t0)
    student_cpu_ms = round(np.mean(cpu_times) * 1000, 4)

    results["latency"] = {
        "teacher_gpu_ms": ava_result['teacher_latency_ms'],
        "student_gpu_ms": ava_result['student_latency_ms'],
        "student_cpu_ms": student_cpu_ms,
        "gpu_speedup": ava_result['speedup'],
        "note": "Teacher cannot run on CPU without torch_geometric. Latency measures temporal+graph inference only.",
    }

    print(f"\n  {'Platform':<20} | {'Teacher (ms)':>15} | {'Student (ms)':>15} | {'Speedup':>10}")
    print(f"  {'-'*65}")
    print(f"  {'GPU':<20} | {ava_result['teacher_latency_ms']:>14.2f} | {ava_result['student_latency_ms']:>14.2f} | {ava_result['speedup']}x")
    print(f"  {'CPU':<20} | {'N/A (needs PyG)':>15} | {student_cpu_ms:>14.2f} | {'':>10}")

    # ================================================================
    # SAVE ALL RESULTS
    # ================================================================
    result_path = r"D:\FYP\MSSG\realtime\realtime_epoch\comprehensive_eval_result.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  All results saved to: {result_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
