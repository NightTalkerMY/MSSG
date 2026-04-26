import torch
import os
import gc
from HoloPiStudentDataset import HoloPiStudentDataset 

if __name__ == '__main__':
    print("Initializing SCENE-BASED Data Pipeline...")
    
    # Process the Training Data (This metadata loading takes a minute but uses low RAM)
    train_dataset = HoloPiStudentDataset(
        predata_base_dir=r"D:\FYP\MSSG\predata", 
        label_dir=r"D:\FYP\MSSG\data\AVA\ava_activespeaker_train_v1.0", 
        mode='train',
        buffer_size=50, 
        stride=25,
        max_speakers=5 
    )
    
    output_dir = r"D:\FYP\MSSG\realtime\cached_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_LIMIT = 2000 # Save a new file every 2000 windows
    file_index = 0
    
    all_v, all_a, all_t, all_s, all_l, all_m = [], [], [], [], [], []
    
    total_items = len(train_dataset)
    print(f"Total SCENE windows to process: {total_items}")
    print("Extracting and saving in chunks to protect RAM...")

    for i in range(total_items):
        v, a, t, s, l, m = train_dataset[i]
        all_v.append(v)
        all_a.append(a)
        all_t.append(t)
        all_s.append(s)
        all_l.append(l)
        all_m.append(m)
        
        # When we hit the limit (or the very last item), save to disk and clear RAM
        if (i + 1) % CHUNK_LIMIT == 0 or (i + 1) == total_items:
            save_path = os.path.join(output_dir, f"train_cache_part_{file_index}.pt")
            
            torch.save({
                'v': torch.stack(all_v),
                'a': torch.stack(all_a),
                't': torch.stack(all_t),
                's': torch.stack(all_s),
                'l': torch.stack(all_l),
                'm': torch.stack(all_m)
            }, save_path)
            
            print(f"Saved {save_path} (Processed {i + 1}/{total_items})")
            
            # CLEAR THE RAM
            all_v, all_a, all_t, all_s, all_l, all_m = [], [], [], [], [], []
            file_index += 1
            gc.collect() # Force Python to release memory back to Windows
            
    print("Phase 1 Complete! All data safely cached.")