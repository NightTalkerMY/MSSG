"""
Evaluate Teacher MSSG vs Student under TRUE online streaming — AVA dataset.

Protocol:
  - Both models receive the same 50-frame cached chunks
  - Only the LAST frame (frame 49) prediction is used
  - Teacher BiGRU runs FULLY INTACT (forward + backward)
  - Reports mAP comparison + per-sample latency
"""
import os
import sys
import glob
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score
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
FRAME_WIDTH = 1280

ENCODER_CONFIGS = [
    {'name': 'face',            'path': 'face_model',             'cache_slice': (0, 128),     'mssg_start': 0},
    {'name': 'face_body',       'path': 'body_model',             'cache_slice': (256, 384),   'mssg_start': 128},
    {'name': 'face_large',      'path': 'face_large_model',       'cache_slice': (384, 512),   'mssg_start': 256},
    {'name': 'background',      'path': 'background_model.model', 'cache_slice': (512, 640),   'mssg_start': 384},
    {'name': 'face_small',      'path': 'face_small_model',       'cache_slice': (640, 768),   'mssg_start': 512},
    {'name': 'face_down',       'path': 'face_down_model',        'cache_slice': (768, 896),   'mssg_start': 640},
    {'name': 'face_body_large', 'path': 'body_large_model',       'cache_slice': (128, 256),   'mssg_start': 768},
]


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
                    data['v'][idx],
                    data['a'][idx],
                    data['s'][idx],
                    data['l'][idx],
                    data['m'][idx],
                )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- TRUE Online Streaming Evaluation (AVA) on {device} ---")

    # 1. Load teacher: 7 BiGRUs + MSSG
    print("Loading 7 BiGRUs from pre-trained encoders...")
    bigrus = {}
    for config in ENCODER_CONFIGS:
        full_path = os.path.join(PRETRAIN_DIR, config['path'])
        s = ASD(encoder_struct=[32, 64, 128])
        if os.path.isdir(full_path):
            s.loadParameters_multi(full_path)
        else:
            s.loadParameters(full_path)
        bigrus[config['name']] = s.model.GRU.to(device).eval()

    print("Loading MSSG graph model...")
    mssg_model = MSSG().to(device)
    state_dict = torch.load(MSSG_WEIGHTS, map_location=device)
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    mssg_model.load_state_dict(clean_dict)
    mssg_model.eval()

    # 2. Load student model
    print("Loading Student model...")
    student_model = CausalStudentASD().to(device)
    student_model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device))
    student_model.eval()

    # 3. Load validation data
    dataset = ValCacheDataset(VAL_CACHE_DIR)
    val_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    teacher_scores, teacher_labels = [], []
    student_scores, student_labels = [], []
    teacher_times, student_times = [], []

    LAST_FRAME = 49

    # Warmup GPU
    print("Warming up GPU...")
    dummy_v = torch.randn(1, 5, 50, 896).to(device)
    dummy_a = torch.randn(1, 5, 50, 128).to(device)
    dummy_s = torch.randn(1, 5, 50, 4).to(device)
    dummy_m = torch.ones(1, 5, 50).to(device)
    for _ in range(10):
        _ = student_model(dummy_v, dummy_a, dummy_s, dummy_m)
    torch.cuda.synchronize()

    print("Evaluating both models — LAST FRAME ONLY (true streaming)...")
    with torch.no_grad():
        for v, a, s_feat, labels, masks in tqdm(val_loader):
            v = v[0]
            a = a[0]
            s_feat = s_feat[0]
            labels = labels[0]
            masks = masks[0]

            active_at_last = [p for p in range(5) if masks[p, LAST_FRAME] > 0.5]
            if len(active_at_last) == 0:
                continue

            # ===== TEACHER =====
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            valid_speakers = [p for p in range(5) if masks[p].sum() > 0]
            num_valid = len(valid_speakers)

            feat_av_all = torch.zeros(num_valid, 50, 896)
            for config in ENCODER_CONFIGS:
                cs_start, cs_end = config['cache_slice']
                ms = config['mssg_start']
                bigru = bigrus[config['name']]
                feat_v_enc = torch.stack([v[p, :, cs_start:cs_end] for p in valid_speakers])
                feat_a_enc = torch.stack([a[p] for p in valid_speakers])
                bigru_input = (feat_v_enc + feat_a_enc).to(device)
                bigru_output = bigru(bigru_input).cpu()
                feat_av_all[:, :, ms:ms+128] = bigru_output

            node_features = []
            node_labels_t = []
            src_v, tgt_v, e_attr = [], [], []
            node_idx = 0

            active_indices = []
            for sp_idx, p in enumerate(valid_speakers):
                if masks[p, LAST_FRAME] > 0.5:
                    active_indices.append((sp_idx, p))

            num_active = len(active_indices)
            for sp_idx, p in active_indices:
                feat_av = feat_av_all[sp_idx, LAST_FRAME]
                spatial = s_feat[p, LAST_FRAME]
                face_size = (spatial[2] * FRAME_WIDTH).unsqueeze(0)
                speaker_num = torch.tensor([float(num_active)])
                node_feat = torch.cat([feat_av, face_size, spatial, speaker_num])
                node_features.append(node_feat)
                lbl = labels[p, LAST_FRAME].item()
                if lbl == 2.0:
                    lbl = 0.0
                node_labels_t.append(lbl)
                node_idx += 1

            for j in range(node_idx):
                for k in range(node_idx):
                    src_v.append(j)
                    tgt_v.append(k)
                    e_attr.append(111 if j == k else 0)

            x = torch.stack(node_features).to(device)
            edge_index = torch.tensor([src_v, tgt_v], dtype=torch.long).to(device)
            edge_attr_t = torch.tensor(e_attr, dtype=torch.float).to(device)
            y = torch.tensor(node_labels_t, dtype=torch.float).unsqueeze(1).to(device)
            graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_t)
            output = mssg_model(graph_data)
            t_scores_batch = F.softmax(output, dim=-1)[:, 1].cpu().tolist()

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            teacher_times.append(t_end - t_start)

            teacher_scores.extend(t_scores_batch)
            teacher_labels.extend(node_labels_t)

            # ===== STUDENT =====
            torch.cuda.synchronize()
            s_start = time.perf_counter()

            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)
            logits, _ = student_model(v_in, a_in, s_in, m_in)
            s_probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)

            torch.cuda.synchronize()
            s_end = time.perf_counter()
            student_times.append(s_end - s_start)

            for p in active_at_last:
                student_scores.append(s_probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                if lbl == 2.0:
                    lbl = 0.0
                student_labels.append(lbl)

    # 4. Results
    teacher_mAP = average_precision_score(teacher_labels, teacher_scores) * 100
    student_mAP = average_precision_score(student_labels, student_scores) * 100
    diff_mAP = student_mAP - teacher_mAP

    teacher_avg_ms = np.mean(teacher_times) * 1000
    student_avg_ms = np.mean(student_times) * 1000
    speedup = teacher_avg_ms / student_avg_ms

    print(f"\n{'='*70}")
    print(f"  TRUE ONLINE STREAMING — AVA (last-frame prediction only)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<25} | {'Teacher':<12} | {'Student':<12} | {'Diff':<12}")
    print(f"  {'-'*65}")
    print(f"  {'mAP (%)':<25} | {teacher_mAP:<12.2f} | {student_mAP:<12.2f} | {diff_mAP:+.2f}%")
    print(f"  {'Latency (ms/sample)':<25} | {teacher_avg_ms:<12.2f} | {student_avg_ms:<12.2f} | {speedup:.1f}x faster")
    print(f"  {'Samples evaluated':<25} | {len(teacher_labels):<12} | {len(student_labels):<12} |")
    print(f"{'='*70}\n")

    result = {
        "dataset": "AVA",
        "experiment": "true_online_streaming",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "last_frame_only (frame 49 of 50-frame buffer)",
        "teacher_mAP": round(teacher_mAP, 4),
        "student_mAP": round(student_mAP, 4),
        "diff_mAP": round(diff_mAP, 4),
        "teacher_latency_ms": round(teacher_avg_ms, 4),
        "student_latency_ms": round(student_avg_ms, 4),
        "speedup": round(speedup, 2),
        "num_teacher_samples": len(teacher_labels),
        "num_student_samples": len(student_labels),
    }
    result_path = r"D:\FYP\MSSG\realtime\realtime_epoch\eval_streaming_result.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to: {result_path}")


if __name__ == '__main__':
    main()
