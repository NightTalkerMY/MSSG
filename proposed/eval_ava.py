import os
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from datetime import datetime

from student_model import CausalStudentASD

# Define a specific Validation Loader so we don't accidentally import the "train_" hardcode
class ValChunkedIterableDataset(IterableDataset):
    def __init__(self, cache_dir):
        # Explicitly look for the val_cache files
        self.files = sorted(glob.glob(os.path.join(cache_dir, "val_cache_part_*.pt")))
        print(f"Found {len(self.files)} cache files.")

    def __iter__(self):
        # We don't shuffle validation data, just read it straight through
        for f_path in self.files:
            data = torch.load(f_path, map_location='cpu')
            num_items = data['l'].shape[0]
            
            for idx in range(num_items):
                yield (
                    data['v'][idx],
                    data['a'][idx],
                    data['t'][idx],
                    data['s'][idx],
                    data['l'][idx],
                    data['m'][idx]
                )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Booting AVA Evaluation on {device} ---")
    
    # 1. Load the Validation Cache
    val_cache_dir = r"D:\FYP\MSSG\realtime\val_cached_chunks"
    dataset = ValChunkedIterableDataset(val_cache_dir)
    val_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    # 2. Initialize Model & Load Epoch 30 Weights
    model = CausalStudentASD().to(device)
    weights_path = r"D:\FYP\MSSG\realtime\realtime_epoch\holopi_student_best.pt"
    print(f"Loading weights from: {weights_path}")
    
    # Use weights_only=True for security and future compatibility
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() 
    
    target_total = []
    soft_total = []
    
    print("Evaluating Model...")
    with torch.no_grad(): 
        for v, a, t, s, l, m in tqdm(val_loader):
            v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
            l = l.to(device)
            
            # Forward pass
            student_logits, _ = model(v, a, s, m)
            
            # Flatten to 1D arrays and filter out Ghost Padding
            flat_mask = m.view(-1)
            valid_indices = flat_mask > 0.5 
            
            valid_logits = student_logits.view(-1, 2)[valid_indices]
            valid_labels = l.view(-1)[valid_indices]
            
            # Apply Softmax to get probability of Class 1 (Speaking)
            scores = F.softmax(valid_logits, dim=-1)[:, 1]
            
            # Move back to CPU for Sklearn metrics
            soft_total.extend(scores.cpu().tolist())
            target_total.extend(valid_labels.cpu().tolist())

    # 3. Calculate Final AVA mAP
    print("\n" + "="*50)
    if len(target_total) == 0:
        print("ERROR: No valid targets found. Check your mask/padding logic.")
    else:
        mAP = average_precision_score(target_total, soft_total) * 100
        print(f"FINAL AVA mAP SCORE: {mAP:.3f}%")

        # Save result
        result = {
            "dataset": "AVA",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weights": weights_path,
            "mAP": round(mAP, 4),
            "num_samples": len(target_total),
        }
        result_path = r"D:\FYP\MSSG\realtime\realtime_epoch\eval_ava_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to: {result_path}")
    print("="*50)

if __name__ == '__main__':
    main()