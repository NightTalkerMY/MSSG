"""
Evaluate Teacher MSSG vs Student under TRUE online streaming — Columbia dataset.

Same protocol as eval_teacher_causal.py (AVA version):
  - Both models receive the same 50-frame cached chunks
  - Only the LAST frame (frame 49) prediction is used
  - Teacher BiGRU runs FULLY INTACT (forward + backward)
  - Backward GRU starts cold at frame 49 → zero future context

Reports per-speaker mAP and F1 (matching Columbia evaluation convention).
"""
import os
import sys
import glob
import json
import torch
import numpy as np
import torch.nn.functional as F
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
COL_CACHE_DIR = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
FRAME_WIDTH = 1280  # Columbia video is 1280x720, size_dic uses width (1280)

ENCODER_CONFIGS = [
    {'name': 'face',            'path': 'face_model',             'cache_slice': (0, 128),     'mssg_start': 0},
    {'name': 'face_body',       'path': 'body_model',             'cache_slice': (256, 384),   'mssg_start': 128},
    {'name': 'face_large',      'path': 'face_large_model',       'cache_slice': (384, 512),   'mssg_start': 256},
    {'name': 'background',      'path': 'background_model.model', 'cache_slice': (512, 640),   'mssg_start': 384},
    {'name': 'face_small',      'path': 'face_small_model',       'cache_slice': (640, 768),   'mssg_start': 512},
    {'name': 'face_down',       'path': 'face_down_model',        'cache_slice': (768, 896),   'mssg_start': 640},
    {'name': 'face_body_large', 'path': 'body_large_model',       'cache_slice': (128, 256),   'mssg_start': 768},
]


class ColumbiaIterableDataset(IterableDataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "col_cache_part_*.pt")))
        print(f"Found {len(self.files)} Columbia cache files.")

    def __iter__(self):
        for f_path in self.files:
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            for idx in range(num_items):
                yield (
                    data['v'][idx],      # [5, 50, 896]
                    data['a'][idx],      # [5, 50, 128]
                    data['s'][idx],      # [5, 50, 4]
                    data['l'][idx],      # [5, 50]
                    data['m'][idx],      # [5, 50]
                    data['names'][idx],  # list of 5 strings
                )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- TRUE Online Streaming Evaluation (Columbia) on {device} ---")

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

    # 3. Load Columbia cached data
    dataset = ColumbiaIterableDataset(COL_CACHE_DIR)
    val_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Per-speaker tracking
    teacher_speaker_preds = {}
    teacher_speaker_targets = {}
    student_speaker_preds = {}
    student_speaker_targets = {}

    LAST_FRAME = 49

    print("Evaluating both models — LAST FRAME ONLY (true streaming)...")
    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in tqdm(val_loader):
            v = v[0]            # [5, 50, 896]
            a = a[0]            # [5, 50, 128]
            s_feat = s_feat[0]  # [5, 50, 4]
            labels = labels[0]  # [5, 50]
            masks = masks[0]    # [5, 50]

            # Extract names (DataLoader wraps strings in tuples)
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]

            # Only evaluate speakers active at the LAST frame (and not Ghost)
            active_at_last = [
                p for p in range(5)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            if len(active_at_last) == 0:
                continue

            # ===== TEACHER: full BiGRU (unchanged) → last frame only =====
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

            # Build graph for LAST FRAME ONLY
            node_features = []
            node_labels_t = []
            node_names_t = []
            src_v, tgt_v, e_attr = [], [], []
            node_idx = 0

            active_indices = []
            for sp_idx, p in enumerate(valid_speakers):
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost":
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
                node_names_t.append(extracted_names[p])
                node_idx += 1

            # Edges: all-to-all at same frame
            for j in range(node_idx):
                for k in range(node_idx):
                    src_v.append(j)
                    tgt_v.append(k)
                    e_attr.append(111 if j == k else 0)

            if len(node_features) == 0:
                continue

            x = torch.stack(node_features).to(device)
            edge_index = torch.tensor([src_v, tgt_v], dtype=torch.long).to(device)
            edge_attr_t = torch.tensor(e_attr, dtype=torch.float).to(device)
            y = torch.tensor(node_labels_t, dtype=torch.float).unsqueeze(1).to(device)

            graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_t)
            output = mssg_model(graph_data)
            t_scores = F.softmax(output, dim=-1)[:, 1].cpu().tolist()

            for i, name in enumerate(node_names_t):
                if name not in teacher_speaker_preds:
                    teacher_speaker_preds[name] = []
                    teacher_speaker_targets[name] = []
                teacher_speaker_preds[name].append(t_scores[i])
                teacher_speaker_targets[name].append(node_labels_t[i])

            # ===== STUDENT: causal model → last frame only =====
            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)

            logits, _ = student_model(v_in, a_in, s_in, m_in)
            s_probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)

            for p in active_at_last:
                name = extracted_names[p]
                if name not in student_speaker_preds:
                    student_speaker_preds[name] = []
                    student_speaker_targets[name] = []
                student_speaker_preds[name].append(s_probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                if lbl == 2.0:
                    lbl = 0.0
                student_speaker_targets[name].append(lbl)

    # 4. Per-speaker results (F1 comparison)
    print(f"\n{'='*75}")
    print(f"  === TRUE ONLINE STREAMING — Columbia (last-frame prediction only) ===")
    print(f"{'='*75}")
    print(f"\n{'Speaker':<15} | {'Teacher F1':<15} | {'Student F1':<15} | {'Diff':<10}")
    print("-" * 60)

    teacher_f1_list, student_f1_list = [], []
    all_speakers = sorted(set(list(teacher_speaker_preds.keys()) + list(student_speaker_preds.keys())))

    for speaker in all_speakers:
        # Teacher F1
        t_preds = np.array(teacher_speaker_preds.get(speaker, []))
        t_targets = np.array(teacher_speaker_targets.get(speaker, []))
        t_prec, t_rec, _ = precision_recall_curve(t_targets, t_preds)
        t_f1_scores = 2 * (t_prec * t_rec) / (t_prec + t_rec + 1e-10)
        t_best_f1 = np.max(t_f1_scores) * 100
        teacher_f1_list.append(t_best_f1)

        # Student F1
        s_preds = np.array(student_speaker_preds.get(speaker, []))
        s_targets = np.array(student_speaker_targets.get(speaker, []))
        s_prec, s_rec, _ = precision_recall_curve(s_targets, s_preds)
        s_f1_scores = 2 * (s_prec * s_rec) / (s_prec + s_rec + 1e-10)
        s_best_f1 = np.max(s_f1_scores) * 100
        student_f1_list.append(s_best_f1)

        diff = s_best_f1 - t_best_f1
        print(f"{speaker:<15} | {t_best_f1:<15.2f} | {s_best_f1:<15.2f} | {diff:+.2f}")

    teacher_avg_f1 = float(np.mean(teacher_f1_list))
    student_avg_f1 = float(np.mean(student_f1_list))

    print("-" * 60)
    print(f"{'AVERAGE':<15} | {teacher_avg_f1:<15.2f} | {student_avg_f1:<15.2f} | {student_avg_f1 - teacher_avg_f1:+.2f}")
    print(f"\n  Student advantage (F1): {student_avg_f1 - teacher_avg_f1:+.4f}%")
    print(f"{'='*75}\n")

    # 5. Save result
    result = {
        "dataset": "Columbia",
        "experiment": "true_online_streaming",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "last_frame_only (frame 49 of 50-frame buffer)",
        "avg_teacher_F1": round(teacher_avg_f1, 4),
        "avg_student_F1": round(student_avg_f1, 4),
        "student_advantage_F1": round(student_avg_f1 - teacher_avg_f1, 4),
        "per_speaker": {},
    }
    for i, speaker in enumerate(all_speakers):
        result["per_speaker"][speaker] = {
            "teacher_F1": round(teacher_f1_list[i], 4),
            "student_F1": round(student_f1_list[i], 4),
            "num_teacher_samples": len(teacher_speaker_preds.get(speaker, [])),
            "num_student_samples": len(student_speaker_preds.get(speaker, [])),
        }

    result_path = r"D:\FYP\MSSG\realtime\realtime_epoch\eval_streaming_columbia_result.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to: {result_path}")


if __name__ == '__main__':
    main()
