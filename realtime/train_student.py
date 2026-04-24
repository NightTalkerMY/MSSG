import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import our custom Student model from Phase 2
from student_model import CausalStudentASD

# --- 1. THE CHUNKED DATALOADER ---
class ChunkedHoloPiDataset(Dataset):
    """
    Dynamically loads the 2000-chunk .pt files to protect your 64GB RAM.
    """
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "train_cache_part_*.pt")))
        print(f"Found {len(self.files)} cache files.")
        
        self.data_map = []
        # Quickly map exactly which chunk lives in which file
        for file_idx, f_path in enumerate(self.files):
            # Load just the metadata/shape, not the whole file
            v_shape = torch.load(f_path, map_location='cpu')['l'].shape[0] 
            for local_idx in range(v_shape):
                self.data_map.append((f_path, local_idx))
                
        # Cache the currently opened file so we don't constantly hit the SSD
        self.current_file_path = None
        self.current_data = None

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        file_path, local_idx = self.data_map[idx]
        
        # Load the file into RAM only if it's not already loaded
        if self.current_file_path != file_path:
            self.current_data = torch.load(file_path, map_location='cpu')
            self.current_file_path = file_path
            
        return (
            self.current_data['v'][local_idx],
            self.current_data['a'][local_idx],
            self.current_data['t'][local_idx], # The Cached Teacher!
            self.current_data['s'][local_idx],
            self.current_data['l'][local_idx],
            self.current_data['m'][local_idx]
        )

# --- 2. THE DISTILLATION LOSS ---
class KDLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        # Standard classification loss for 0/1 Speaking labels
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        # Feature matching loss to mimic the Teacher
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, student_logits, student_features, teacher_features, labels, mask):
        # Flatten everything to [Batch * Speakers * Frames, ...]
        flat_mask = mask.view(-1)
        valid_indices = flat_mask > 0.5 # Ignore the Ghost padding!
        
        # 1. Hard Classification Loss (Cross Entropy)
        flat_logits = student_logits.view(-1, 2)[valid_indices]
        flat_labels = labels.view(-1)[valid_indices]
        loss_ce = self.ce_loss(flat_logits, flat_labels).mean()
        
        # 2. Feature Distillation Loss (MSE)
        flat_s_feat = student_features.view(-1, 128)[valid_indices]
        flat_t_feat = teacher_features.view(-1, 128)[valid_indices]
        loss_mse = self.mse_loss(flat_s_feat, flat_t_feat).mean()
        
        # 3. Combine them
        total_loss = (self.alpha * loss_ce) + ((1 - self.alpha) * loss_mse)
        return total_loss, loss_ce, loss_mse

# --- 3. THE TRAINING LOOP ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Initialize Dataset and Dataloader
    cache_dir = r"D:\FYP\MSSG\realtime\cached_chunks"
    dataset = ChunkedHoloPiDataset(cache_dir)
    
    # num_workers=0 is still recommended on Windows to avoid pickling the loaded files
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Initialize Model, Loss, and Optimizer
    model = CausalStudentASD().to(device)
    criterion = KDLoss(alpha=0.5) # 50% CE, 50% Teacher Mimicry
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    num_epochs = 30
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_mse = 0.0
        
        # TQDM progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for v, a, t, s, l, m in pbar:
            # Move to RTX 4080 Super
            v, a, t = v.to(device), a.to(device), t.to(device)
            s, l, m = s.to(device), l.to(device), m.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (Student)
            student_logits, student_features = model(v, a, s, m)
            
            # Calculate Distillation Loss
            loss, loss_ce, loss_mse = criterion(student_logits, student_features, t, l, m)
            
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            
            pbar.set_postfix({'Total': f"{loss.item():.3f}", 'CE': f"{loss_ce.item():.3f}", 'MSE': f"{loss_mse.item():.3f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Student Checkpoint
        save_path = f"D:\\FYP\\MSSG\\holopi_student_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

if __name__ == '__main__':
    main()