import torch
import os
import gc
from HoloPiStudentDataset import HoloPiStudentDataset 

if __name__ == '__main__':
    print("--- Extracting AVA Validation Features ---")
    
    # 1. Initialize Dataset for VAL
    val_dataset = HoloPiStudentDataset(
        predata_base_dir=r"D:\FYP\MSSG\predata", 
        # UPDATE THIS PATH if your validation labels are in a different folder:
        label_dir=r"D:\FYP\MSSG\data\AVA\ava_activespeaker_test_v1.0", 
        mode='val', # SWITCHED TO VAL
        buffer_size=50, 
        stride=25,
        max_speakers=5 
    )
    
    # 2. Separate output directory for validation chunks
    output_dir = r"D:\FYP\MSSG\realtime\val_cached_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_LIMIT = 2000
    file_index = 0
    all_v, all_a, all_t, all_s, all_l, all_m = [], [], [], [], [], []
    
    total_items = len(val_dataset)
    print(f"Total VAL SCENE windows to process: {total_items}")

    for i in range(total_items):
        v, a, t, s, l, m = val_dataset[i]
        all_v.append(v)
        all_a.append(a)
        all_t.append(t)
        all_s.append(s)
        all_l.append(l)
        all_m.append(m)
        
        if (i + 1) % CHUNK_LIMIT == 0 or (i + 1) == total_items:
            save_path = os.path.join(output_dir, f"val_cache_part_{file_index}.pt")
            torch.save({
                'v': torch.stack(all_v), 'a': torch.stack(all_a),
                't': torch.stack(all_t), 's': torch.stack(all_s),
                'l': torch.stack(all_l), 'm': torch.stack(all_m)
            }, save_path)
            
            print(f"Saved {save_path} (Processed {i + 1}/{total_items})")
            
            all_v, all_a, all_t, all_s, all_l, all_m = [], [], [], [], [], []
            file_index += 1
            gc.collect() 
            
    print("AVA Validation Cache Complete!")