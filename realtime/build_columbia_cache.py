import os
import csv
import ast
import torch
import numpy as np
from tqdm import tqdm

def decode_feature(feature_str):
    feature_str = feature_str.strip("[]")
    return np.array([float(x) for x in feature_str.split(',')], dtype=np.float32)

def load_columbia_base_features(csv_path):
    """Loads face_feats_info which contains Teacher, Face Visual, and Audio."""
    feat_dict = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Could not find base features at {csv_path}")
        return feat_dict
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4: continue
            key = row[0] # "abbas_tracks_...:3750.614"
            feat_dict[key] = {
                't': decode_feature(row[1]), # Teacher AV
                'v': decode_feature(row[2]), # Face V
                'a': decode_feature(row[3])  # Raw Audio
            }
    return feat_dict

def load_columbia_aux_features(csv_path):
    """Loads the other 6 visual features which only have one array."""
    feat_dict = {}
    if not os.path.exists(csv_path): return feat_dict
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            key = row[0]
            feat_dict[key] = decode_feature(row[1])
    return feat_dict

def main():
    print("--- Booting Columbia Cache Builder ---")
    
    # 1. Load the Labels & Bounding Boxes
    label_csv = r"D:\FYP\MSSG\data\Columbia\predata\csv_split\train\col.csv"
    timestamps = {}
    
    print("Parsing Ground Truth Labels...")
    with open(label_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        # for row in reader:
        #     _, ts, x1, y1, x2, y2, label, entity_id, _, _ = row
        #     ts_float = float(ts)
            
        #     if ts_float not in timestamps: timestamps[ts_float] = []
                
        #     timestamps[ts_float].append({
        #         'entity_id': entity_id,
        #         'name': entity_id.split('_')[0].capitalize(),
        #         'label': int(label),
        #         'box': [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)] # Convert to cx, cy, w, h format
        #     })
        for row in reader:
            # We explicitly unpack the real_label_id now
            _, ts, x1, y1, x2, y2, dummy_label, entity_id, real_label_id, _ = row
            ts_float = float(ts)
            
            if ts_float not in timestamps: timestamps[ts_float] = []
                
            timestamps[ts_float].append({
                'entity_id': entity_id,
                'name': entity_id.split('_')[0].capitalize(),
                'label': int(real_label_id), # NOW GRABBING THE ACTUAL GROUND TRUTH!
                'box': [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)] 
            })

    sorted_ts = sorted(timestamps.keys())
    print(f"Found {len(sorted_ts)} unique frames in the Columbia video.")

    # 2. Load Features
    base_feat_dir = r"D:\FYP\MSSG\data\Columbia\predata"
    
    print("Loading base features (Audio, Teacher, Face Visual)...")
    base_feats = load_columbia_base_features(os.path.join(base_feat_dir, "face_feats_info", "train", "col.csv"))
    
    aux_folders = [
        "face_body_large_feats_info", "face_large_feats_info",
        "face.body_feats_info", "face_small_feats_info", 
        "background_feats_info", "face_down_feats_info"
    ]
    
    aux_feat_dicts = []
    for folder in aux_folders:
        print(f"Loading {folder}...")
        aux_feat_dicts.append(load_columbia_aux_features(os.path.join(base_feat_dir, folder, "train", "col.csv")))

    # 3. Build the 50-Frame Chunks
    output_dir = r"D:\FYP\MSSG\realtime\columbia_cached_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    buffer_size = 50
    stride = 25
    max_speakers = 5
    
    all_v, all_a, all_t, all_s, all_l, all_m, all_names = [], [], [], [], [], [], []
    chunk_idx = 0

    print("Sliding temporal window across the video...")
    for i in tqdm(range(0, len(sorted_ts) - buffer_size + 1, stride)):
        window_ts = sorted_ts[i : i + buffer_size]
        
        unique_people = set()
        for ts in window_ts:
            for person in timestamps[ts]:
                unique_people.add(person['entity_id'])
                
        active_people = list(unique_people)[:max_speakers]
        
        v_tensor = np.zeros((max_speakers, buffer_size, 896), dtype=np.float32)
        a_tensor = np.zeros((max_speakers, buffer_size, 128), dtype=np.float32)
        t_tensor = np.zeros((max_speakers, buffer_size, 128), dtype=np.float32)
        s_tensor = np.zeros((max_speakers, buffer_size, 4), dtype=np.float32)
        l_tensor = np.zeros((max_speakers, buffer_size), dtype=np.int64)
        m_tensor = np.zeros((max_speakers, buffer_size), dtype=np.float32)
        chunk_names = ["Ghost"] * max_speakers
        
        for p_idx, entity_id in enumerate(active_people):
            chunk_names[p_idx] = entity_id.split('_')[0].capitalize()
            
            for t_idx, ts in enumerate(window_ts):
                person_data = next((p for p in timestamps[ts] if p['entity_id'] == entity_id), None)
                
                if person_data:
                    m_tensor[p_idx, t_idx] = 1.0
                    l_tensor[p_idx, t_idx] = person_data['label']
                    s_tensor[p_idx, t_idx] = person_data['box']
                    
                    feat_key = f"{entity_id}:{ts}"
                    
                    if feat_key in base_feats:
                        a_tensor[p_idx, t_idx] = base_feats[feat_key]['a']
                        t_tensor[p_idx, t_idx] = base_feats[feat_key]['t']
                        
                        stitched_v = list(base_feats[feat_key]['v'])
                        for aux_dict in aux_feat_dicts:
                            if feat_key in aux_dict:
                                stitched_v.extend(aux_dict[feat_key])
                            else:
                                stitched_v.extend(np.zeros(128))
                                
                        v_tensor[p_idx, t_idx] = np.array(stitched_v)
        
        all_v.append(torch.tensor(v_tensor))
        all_a.append(torch.tensor(a_tensor))
        all_t.append(torch.tensor(t_tensor))
        all_s.append(torch.tensor(s_tensor))
        all_l.append(torch.tensor(l_tensor))
        all_m.append(torch.tensor(m_tensor))
        all_names.append(chunk_names)
        
        if len(all_v) >= 2000:
            torch.save({
                'v': torch.stack(all_v), 'a': torch.stack(all_a),
                't': torch.stack(all_t), 's': torch.stack(all_s),
                'l': torch.stack(all_l), 'm': torch.stack(all_m),
                'names': all_names 
            }, os.path.join(output_dir, f"col_cache_part_{chunk_idx}.pt"))
            
            chunk_idx += 1
            all_v, all_a, all_t, all_s, all_l, all_m, all_names = [], [], [], [], [], [], []

    if len(all_v) > 0:
        torch.save({
            'v': torch.stack(all_v), 'a': torch.stack(all_a),
            't': torch.stack(all_t), 's': torch.stack(all_s),
            'l': torch.stack(all_l), 'm': torch.stack(all_m),
            'names': all_names
        }, os.path.join(output_dir, f"col_cache_part_{chunk_idx}.pt"))

    print("Columbia Cache Complete!")

if __name__ == '__main__':
    main()