"""
Average the top-K checkpoints by combined metric, then evaluate on AVA + Columbia.
"""
import os
import sys
import csv
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from student_model import CausalStudentASD
from train_student import ValChunkedIterableDataset, ColumbiaValDataset, compute_columbia_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = r"D:\FYP\MSSG\realtime\realtime_epoch"
    log_path = os.path.join(save_dir, "training_log.csv")

    # 1. Parse training log to find top-K epochs by combined metric
    epochs = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append({
                'epoch': int(row['epoch']),
                'combined': float(row['combined']),
                'mAP': float(row['val_mAP']),
                'col_F1': float(row['col_F1']),
            })

    epochs.sort(key=lambda x: x['combined'], reverse=True)
    TOP_K = 5
    top_epochs = epochs[:TOP_K]

    print(f"Top {TOP_K} epochs by combined metric:")
    for e in top_epochs:
        print(f"  Epoch {e['epoch']:>2}: Combined={e['combined']:.4f}  AVA={e['mAP']:.4f}  Col={e['col_F1']:.4f}")

    # 2. Average their weights
    print(f"\nAveraging weights...")
    avg_state = None
    for e in top_epochs:
        path = os.path.join(save_dir, f"holopi_student_epoch_{e['epoch']}.pt")
        state = torch.load(path, map_location='cpu')
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    for k in avg_state:
        avg_state[k] /= TOP_K

    # 3. Load averaged model
    model = CausalStudentASD().to(device)
    model.load_state_dict(avg_state)
    model.eval()

    # 4. Evaluate on AVA val
    val_cache_dir = r"D:\FYP\MSSG\realtime\val_cached_chunks"
    val_dataset = ValChunkedIterableDataset(val_cache_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for v, a, t, s, l, m in tqdm(val_loader, desc="AVA Val"):
            v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
            l = l.to(device)
            logits, _ = model(v, a, s, m)
            flat_mask = m.view(-1)
            valid = flat_mask > 0.5
            valid_logits = logits.view(-1, 2)[valid]
            valid_labels = l.view(-1)[valid]
            scores = F.softmax(valid_logits, dim=-1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(valid_labels.cpu().tolist())

    mAP = average_precision_score(all_labels, all_scores) * 100
    print(f"\n  AVA Val mAP (averaged): {mAP:.2f}%")

    # 5. Evaluate on Columbia
    col_cache_dir = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
    col_dataset = ColumbiaValDataset(col_cache_dir)
    col_loader = DataLoader(col_dataset, batch_size=1, num_workers=0)

    col_f1 = compute_columbia_f1(model, col_loader, device)
    print(f"  Columbia F1 (averaged):  {col_f1:.2f}%")

    combined = 0.5 * mAP + 0.5 * col_f1
    print(f"  Combined (averaged):     {combined:.2f}%")

    # Compare with best single epoch
    best_single = top_epochs[0]
    print(f"\n  Best single epoch ({best_single['epoch']}): Combined={best_single['combined']:.2f}  AVA={best_single['mAP']:.2f}  Col={best_single['col_F1']:.2f}")
    print(f"  Averaged top-{TOP_K}:            Combined={combined:.2f}  AVA={mAP:.2f}  Col={col_f1:.2f}")

    # Save averaged model
    avg_path = os.path.join(save_dir, "holopi_student_avg_top5.pt")
    torch.save(avg_state, avg_path)
    print(f"\n  Averaged model saved to: {avg_path}")


if __name__ == '__main__':
    main()
