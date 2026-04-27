import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from student_model import CausalStudentASD

class ChunkedHoloPiIterableDataset(IterableDataset):
    """
    Loads one chunk file at a time, shuffles internally, and feeds the GPU.
    Completely eliminates SSD thrashing.
    """
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "train_cache_part_*.pt")))
        print(f"Found {len(self.files)} cache files.")

    def __iter__(self):
        # 1. Shuffle the order of the files every epoch
        random.shuffle(self.files)
        
        for f_path in self.files:
            # 2. Load the single 2GB file into RAM
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            
            # 3. Shuffle the 2000 items inside this specific chunk
            indices = list(range(num_items))
            random.shuffle(indices)
            
            # 4. Yield them one by one to form the batch
            for idx in indices:
                yield (
                    data['v'][idx],
                    data['a'][idx],
                    data['t'][idx],
                    data['s'][idx],
                    data['l'][idx],
                    data['m'][idx]
                )

# --- 2. DUAL-MODE DISTILLATION LOSS ---
class KDLoss(nn.Module):
    def __init__(self, mode='ce_mse_kl', alpha=0.5, beta=0.3, temperature=3.0):
        super(KDLoss, self).__init__()
        self.mode = mode
        self.alpha = alpha     # Weight for CE Loss
        self.beta = beta       # Weight for MSE Loss (KL takes the remainder)
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(self, s_logits, s_features, t_features, t_logits, labels, mask):
        # Flatten and filter out the Ghost Padding
        flat_mask = mask.view(-1)
        valid_indices = flat_mask > 0.5 
        
        # 1. Hard Classification Loss (Cross Entropy)
        flat_s_logits = s_logits.view(-1, 2)[valid_indices]
        flat_labels = labels.view(-1)[valid_indices]
        loss_ce = self.ce_loss(flat_s_logits, flat_labels).mean()
        
        # 2. Feature Hint Loss (MSE)
        flat_s_feat = s_features.view(-1, 128)[valid_indices]
        flat_t_feat = t_features.view(-1, 128)[valid_indices]
        loss_mse = self.mse_loss(flat_s_feat, flat_t_feat).mean()
        
        # 3. Soft Logit Loss (KL Divergence)
        if self.mode == 'ce_mse_kl' and t_logits is not None:
            flat_t_logits = t_logits.view(-1, 2)[valid_indices]
            
            # PyTorch KLDiv requires log_softmax for the student, and regular softmax for the teacher
            s_log_probs = F.log_softmax(flat_s_logits / self.temperature, dim=-1)
            t_probs = F.softmax(flat_t_logits / self.temperature, dim=-1)
            
            # Calculate KL and multiply by T^2 to scale gradients properly
            loss_kl = self.kl_loss(s_log_probs, t_probs).sum(dim=-1).mean() * (self.temperature ** 2)
            
            # Weighted sum of all three
            gamma = 1.0 - (self.alpha + self.beta)
            total_loss = (self.alpha * loss_ce) + (self.beta * loss_mse) + (gamma * loss_kl)
            
            return total_loss, loss_ce, loss_mse, loss_kl
            
        else:
            # Fallback to pure CE + MSE mode
            total_loss = (self.alpha * loss_ce) + ((1.0 - self.alpha) * loss_mse)
            return total_loss, loss_ce, loss_mse, torch.tensor(0.0)

# --- 3. TEACHER LOGIT SIMULATOR (Temporary) ---
class TeacherLogitSimulator(nn.Module):
    """
    Simulates the Teacher's PyG classifier so the KL math can run.
    Replace this with the real Teacher logits if extracted offline later.
    """
    def __init__(self):
        super().__init__()
        self.mock_classifier = nn.Linear(128, 2)
        # Freeze it so it acts like a static teacher
        for param in self.mock_classifier.parameters():
            param.requires_grad = False
            
    def forward(self, t_features):
        return self.mock_classifier(t_features)

# --- 4. VALIDATION DATASET (mirrors train but reads val_cache files without shuffling) ---
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
                    data['s'][idx], data['l'][idx], data['m'][idx]
                )

# --- 5. THE TRAINING LOOP ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # --- TOGGLE YOUR MODE HERE ---
    TRAINING_MODE = 'ce_mse' # KL disabled: TeacherLogitSimulator is a random Linear, not a real teacher

    cache_dir = r"D:\FYP\MSSG\realtime\cached_chunks"
    val_cache_dir = r"D:\FYP\MSSG\realtime\val_cached_chunks"

    dataset = ChunkedHoloPiIterableDataset(cache_dir)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=0)

    val_dataset = ValChunkedIterableDataset(val_cache_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    model = CausalStudentASD().to(device)
    teacher_sim = TeacherLogitSimulator().to(device)

    criterion = KDLoss(mode=TRAINING_MODE, alpha=0.4, beta=0.3, temperature=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30
    save_dir = r"D:\FYP\MSSG\realtime\realtime_epoch"
    os.makedirs(save_dir, exist_ok=True)

    best_mAP = 0.0

    # --- Result logging ---
    log_path = os.path.join(save_dir, "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,avg_loss,avg_ce,avg_mse,avg_kl,lr,val_mAP\n")

    for epoch in range(num_epochs):
        # ========== TRAIN ==========
        model.train()
        total_loss, total_ce, total_mse, total_kl = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for v, a, t, s, l, m in pbar:
            v, a, t = v.to(device), a.to(device), t.to(device)
            s, l, m = s.to(device), l.to(device), m.to(device)

            optimizer.zero_grad()

            t_logits = teacher_sim(t) if TRAINING_MODE == 'ce_mse_kl' else None
            student_logits, student_features = model(v, a, s, m)

            loss, loss_ce, loss_mse, loss_kl = criterion(
                student_logits, student_features, t, t_logits, l, m
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            total_kl += loss_kl.item()
            num_batches += 1

            metrics = {
                'Tot': f"{loss.item():.3f}",
                'CE': f"{loss_ce.item():.3f}",
                'MSE': f"{loss_mse.item():.3f}"
            }
            if TRAINING_MODE == 'ce_mse_kl':
                metrics['KL'] = f"{loss_kl.item():.3f}"
            pbar.set_postfix(metrics)

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

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

        if len(all_labels) > 0:
            mAP = average_precision_score(all_labels, all_scores) * 100
            print(f"  Val mAP: {mAP:.2f}%")
        else:
            mAP = 0.0
            print("  WARNING: No valid val samples found.")

        # Log metrics to CSV
        avg_ce = total_ce / num_batches
        avg_mse = total_mse / num_batches
        avg_kl = total_kl / num_batches
        lr = scheduler.get_last_lr()[0]
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_ce:.6f},{avg_mse:.6f},{avg_kl:.6f},{lr:.8f},{mAP:.4f}\n")

        # Save every epoch
        save_path = os.path.join(save_dir, f"holopi_student_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)

        # Track best model
        if mAP > best_mAP:
            best_mAP = mAP
            best_path = os.path.join(save_dir, "holopi_student_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best model! mAP={best_mAP:.2f}% saved to {best_path}")

    print(f"\nTraining complete. Best val mAP: {best_mAP:.2f}%")
    print(f"Training log saved to: {log_path}")

if __name__ == '__main__':
    main()