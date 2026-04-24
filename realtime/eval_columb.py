import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from student_model import CausalStudentASD

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
                    data['v'][idx], data['a'][idx], data['s'][idx],
                    data['l'][idx], data['m'][idx], data['names'][idx] 
                )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Booting Columbia Evaluation on {device} ---")
    
    col_cache_dir = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
    dataset = ColumbiaIterableDataset(col_cache_dir)
    # Batch size 1 keeps name matching strictly aligned
    val_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    model = CausalStudentASD().to(device)
    weights_path = r"D:\FYP\MSSG\realtime\realtime_epoch_mse_ce\holopi_student_epoch_30.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() 
    
    speaker_preds = {}
    speaker_targets = {}
    
    print("Evaluating Columbia Model...")
    with torch.no_grad(): 
        for v, a, s, l, m, names in tqdm(val_loader):
            v, a, s, m = v.to(device), a.to(device), s.to(device), m.to(device)
            l = l.to(device)
            
            student_logits, _ = model(v, a, s, m)
            scores = F.softmax(student_logits, dim=-1)[:, :, :, 1] # [Batch, Speakers, Frames]
            
            # Since batch_size is 1, index [0]
            scores = scores[0] 
            labels = l[0]
            mask = m[0]
            
            # The 'names' object is a tuple of 5 string tuples due to DataLoader quirks
            # E.g. (('Abbas',), ('Bell',), ('Ghost',), ('Ghost',), ('Ghost',))
            extracted_names = [n[0] for n in names]
            
            # Map predictions to speakers
            for p_idx, name in enumerate(extracted_names):
                if name == "Ghost": continue
                
                if name not in speaker_preds:
                    speaker_preds[name] = []
                    speaker_targets[name] = []
                    
                for t_idx in range(50):
                    if mask[p_idx, t_idx] > 0.5: # If person exists in this frame
                        speaker_preds[name].append(scores[p_idx, t_idx].item())
                        speaker_targets[name].append(labels[p_idx, t_idx].item())

    # Calculate Per-Speaker F1
    print("\n" + "="*50)
    print(f"{'Speaker':<15} | {'mAP (%)':<15} | {'Best F1 (%)':<15}")
    print("-" * 50)
    
    f1_list, map_list = [], []
    
    for speaker in sorted(speaker_preds.keys()):
        sp_soft = np.array(speaker_preds[speaker])
        sp_target = np.array(speaker_targets[speaker])
        
        # mAP
        try:
            sp_map = average_precision_score(sp_target, sp_soft) * 100
        except:
            sp_map = 0.0
        map_list.append(sp_map)
        
        # F1 Score
        precision, recall, _ = precision_recall_curve(sp_target, sp_soft)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores) * 100
        f1_list.append(best_f1)
        
        print(f"{speaker:<15} | {sp_map:<15.2f} | {best_f1:<15.2f}")
        
    print("-" * 50)
    print(f"{'AVERAGE':<15} | {np.mean(map_list):<15.2f} | {np.mean(f1_list):<15.2f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()