"""
Multi-seed training script for statistical validation.
Trains 5 independent models (seeds 1-5) sequentially.

Usage:
    python proposed/train_student_seeded.py

Each seed saves to: D:\FYP\MSSG\realtime\realtime_epoch_{seed}\
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from student_model import CausalStudentASD


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All seeds set to: {seed}")


class ChunkedHoloPiIterableDataset(IterableDataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "train_cache_part_*.pt")))
        print(f"Found {len(self.files)} cache files.")

    def __iter__(self):
        random.shuffle(self.files)
        for f_path in self.files:
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            indices = list(range(num_items))
            random.shuffle(indices)
            for idx in indices:
                yield (
                    data['v'][idx], data['a'][idx], data['t'][idx],
                    data['s'][idx], data['l'][idx], data['m'][idx],
                )


class KDLoss(nn.Module):
    def __init__(self, mode='ce_mse', alpha=0.4, beta=0.3, temperature=3.0):
        super(KDLoss, self).__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, s_logits, s_features, t_features, labels, mask):
        flat_mask = mask.view(-1)
        valid_indices = flat_mask > 0.5

        flat_s_logits = s_logits.view(-1, 2)[valid_indices]
        flat_labels = labels.view(-1)[valid_indices]
        loss_ce = self.ce_loss(flat_s_logits, flat_labels).mean()

        flat_s_feat = s_features.view(-1, 128)[valid_indices]
        flat_t_feat = t_features.view(-1, 128)[valid_indices]
        loss_mse = self.mse_loss(flat_s_feat, flat_t_feat).mean()

        total_loss = (self.alpha * loss_ce) + ((1.0 - self.alpha) * loss_mse)
        return total_loss, loss_ce, loss_mse


class ValChunkedIterableDataset(IterableDataset):
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


class ColumbiaValDataset(IterableDataset):
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


def compute_columbia_f1(model, col_loader, device):
    LAST_FRAME = 49
    speaker_preds = {}
    speaker_targets = {}

    with torch.no_grad():
        for v, a, s_feat, labels, masks, names in col_loader:
            v = v[0]; a = a[0]; s_feat = s_feat[0]
            labels = labels[0]; masks = masks[0]
            extracted_names = [n[0] if isinstance(n, (list, tuple)) else n for n in names]

            active_at_last = [
                p for p in range(5)
                if masks[p, LAST_FRAME] > 0.5 and extracted_names[p] != "Ghost"
            ]
            if len(active_at_last) == 0:
                continue

            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            s_in = s_feat.unsqueeze(0).to(device)
            m_in = masks.unsqueeze(0).to(device)

            logits, _ = model(v_in, a_in, s_in, m_in)
            probs = F.softmax(logits[0, :, LAST_FRAME, :], dim=-1)

            for p in active_at_last:
                name = extracted_names[p]
                if name not in speaker_preds:
                    speaker_preds[name] = []
                    speaker_targets[name] = []
                speaker_preds[name].append(probs[p, 1].item())
                lbl = labels[p, LAST_FRAME].item()
                if lbl == 2.0:
                    lbl = 0.0
                speaker_targets[name].append(lbl)

    f1_list = []
    for speaker in sorted(speaker_preds.keys()):
        preds = np.array(speaker_preds[speaker])
        targets = np.array(speaker_targets[speaker])
        prec, rec, _ = precision_recall_curve(targets, preds)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
        f1_list.append(np.max(f1_scores) * 100)

    return float(np.mean(f1_list)) if f1_list else 0.0


SEEDS = [1, 2, 3, 4, 5]


def train_one_seed(seed):
    set_all_seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device} | Seed: {seed}")

    cache_dir = r"D:\FYP\MSSG\realtime\cached_chunks"
    val_cache_dir = r"D:\FYP\MSSG\realtime\val_cached_chunks"
    col_cache_dir = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"

    dataset = ChunkedHoloPiIterableDataset(cache_dir)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=0)

    val_dataset = ValChunkedIterableDataset(val_cache_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    col_dataset = ColumbiaValDataset(col_cache_dir)
    col_loader = DataLoader(col_dataset, batch_size=1, num_workers=0)

    model = CausalStudentASD().to(device)

    criterion = KDLoss(mode='ce_mse', alpha=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    num_epochs = 50
    save_dir = rf"D:\FYP\MSSG\realtime\realtime_epoch_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    best_combined = 0.0

    log_path = os.path.join(save_dir, "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,avg_loss,avg_ce,avg_mse,lr,val_mAP,col_F1,combined\n")

    for epoch in range(num_epochs):
        # ========== TRAIN ==========
        model.train()
        total_loss, total_ce, total_mse = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"[Seed {seed}] Epoch {epoch+1}/{num_epochs}")

        for v, a, t, s, l, m in pbar:
            v, a, t = v.to(device), a.to(device), t.to(device)
            s, l, m = s.to(device), l.to(device), m.to(device)

            optimizer.zero_grad()
            student_logits, student_features = model(v, a, s, m)

            loss, loss_ce, loss_mse = criterion(
                student_logits, student_features, t, l, m
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            num_batches += 1

            pbar.set_postfix({
                'Tot': f"{loss.item():.3f}",
                'CE': f"{loss_ce.item():.3f}",
                'MSE': f"{loss_mse.item():.3f}",
            })

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"[Seed {seed}] Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # ========== VALIDATE (mAP) ==========
        model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for v, a, t, s, l, m in tqdm(val_loader, desc="Validating"):
                v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
                l = l.to(device)
                student_logits, _ = model(v, a, s, m)
                flat_mask = m.view(-1)
                valid = flat_mask > 0.5
                valid_logits = student_logits.view(-1, 2)[valid]
                valid_labels = l.view(-1)[valid]
                scores = F.softmax(valid_logits, dim=-1)[:, 1]
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(valid_labels.cpu().tolist())

        mAP = average_precision_score(all_labels, all_scores) * 100 if all_labels else 0.0
        print(f"  AVA Val mAP: {mAP:.2f}%")

        # ========== COLUMBIA VALIDATION ==========
        col_f1 = compute_columbia_f1(model, col_loader, device)
        print(f"  Columbia F1:  {col_f1:.2f}%")

        combined = 0.4 * mAP + 0.6 * col_f1
        print(f"  Combined:     {combined:.2f}%")

        # Log
        avg_ce = total_ce / num_batches
        avg_mse = total_mse / num_batches
        lr = scheduler.get_last_lr()[0]
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_ce:.6f},{avg_mse:.6f},{lr:.8f},{mAP:.4f},{col_f1:.4f},{combined:.4f}\n")

        # Save every epoch
        save_path = os.path.join(save_dir, f"holopi_student_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)

        # Track best
        if combined > best_combined:
            best_combined = combined
            best_path = os.path.join(save_dir, "holopi_student_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best! Combined={best_combined:.2f}% (AVA={mAP:.2f}%, Col={col_f1:.2f}%)")

    print(f"\n[Seed {seed}] Training complete. Best combined: {best_combined:.2f}%")
    print(f"Models saved to: {save_dir}")
    print(f"Training log: {log_path}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()


def main():
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  STARTING SEED {seed} / {len(SEEDS)}")
        print(f"{'='*70}")
        train_one_seed(seed)

    print(f"\n{'='*70}")
    print(f"  ALL {len(SEEDS)} SEEDS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
