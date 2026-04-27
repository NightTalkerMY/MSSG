import os
import csv
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

def decode_feature(feature_str):
    feature_str = feature_str.strip("[]")
    return np.array([float(x) for x in feature_str.split(',')], dtype=np.float32)

class HoloPiStudentDataset(Dataset):
    def __init__(self, predata_base_dir, label_dir, mode='train', buffer_size=50, stride=25, max_speakers=5):
        self.buffer_size = buffer_size
        self.stride = stride
        self.max_speakers = max_speakers
        self.chunks = []
        
        self.feature_dirs = {
            'face': os.path.join(predata_base_dir, 'face_feats_info', mode),
            'face_body': os.path.join(predata_base_dir, 'face.body_feats_info', mode),
            'face_large': os.path.join(predata_base_dir, 'face_large_feats_info', mode),
            'background': os.path.join(predata_base_dir, 'background_feats_info', mode),
            'face_small': os.path.join(predata_base_dir, 'face_small_feats_info', mode),
            'face_down': os.path.join(predata_base_dir, 'face_down_feats_info', mode),
            'face_body_large': os.path.join(predata_base_dir, 'face_body_large_feats_info', mode)
        }
        
        video_files = glob.glob(os.path.join(self.feature_dirs['face'], "*.csv"))
        print(f"[{mode.upper()}] Found {len(video_files)} video files. Building SCENE-BASED sliding windows...")
        
        for v_file in video_files:
            video_name = os.path.basename(v_file).replace('.csv', '') 
            
            # --- 1. LOAD GROUND TRUTH & SPATIAL CUES ---
            # gt_data[clip_id][timestamp][track_num] = {'label': 0/1, 'spatial': [cx, cy, w, h]}
            gt_data = {}
            gt_file_path = os.path.join(label_dir, f"{video_name}-activespeaker.csv")
            
            if os.path.exists(gt_file_path):
                with open(gt_file_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 8: continue
                        timestamp = round(float(row[1]), 3)
                        x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                        label_str = row[6]
                        
                        full_track_id = row[7] # e.g., "_a9SWtcaNj8_1560_1620:1"
                        parts = full_track_id.split(":")
                        clip_id = parts[0]
                        track_num = parts[1] if len(parts) > 1 else "0"
                        
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w, h = x2 - x1, y2 - y1
                        
                        if clip_id not in gt_data: gt_data[clip_id] = {}
                        if timestamp not in gt_data[clip_id]: gt_data[clip_id][timestamp] = {}
                        
                        gt_data[clip_id][timestamp][track_num] = {
                            'label': 1.0 if label_str == "SPEAKING_AUDIBLE" else 0.0,
                            'spatial': [cx, cy, w, h]
                        }

            # --- 2. THE SCENE FILING CABINET ---
            # scenes[clip_id][timestamp][track_num] = {features...}
            scenes = {}
            
            with open(v_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    title = row[0] 
                    parts = title.split(":")
                    clip_id = parts[0]
                    track_num = parts[1]
                    timestamp = round(float(parts[2]), 3)
                    
                    if clip_id not in scenes: scenes[clip_id] = {}
                    if timestamp not in scenes[clip_id]: scenes[clip_id][timestamp] = {}
                        
                    frame_meta = gt_data.get(clip_id, {}).get(timestamp, {}).get(track_num, {'label': 0.0, 'spatial': [0,0,0,0]})

                    scenes[clip_id][timestamp][track_num] = {
                        'teacher_av': decode_feature(row[1]),
                        'face_v': decode_feature(row[2]),
                        'raw_audio': decode_feature(row[3]),
                        'label': frame_meta['label'],
                        'spatial': frame_meta['spatial']
                    }

            # --- 3. LOAD REMAINING 6 VISUAL CUES ---
            for cue_name, cue_dir in self.feature_dirs.items():
                if cue_name == 'face': continue 
                
                cue_file = os.path.join(cue_dir, f"{video_name}.csv")
                if os.path.exists(cue_file):
                    with open(cue_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            parts = row[0].split(":")
                            clip_id, track_num, timestamp = parts[0], parts[1], round(float(parts[2]), 3)
                            
                            if clip_id in scenes and timestamp in scenes[clip_id] and track_num in scenes[clip_id][timestamp]:
                                # CSV format: [title, feat_av, feat_v, feat_a] — we want feat_v (row[2])
                                scenes[clip_id][timestamp][track_num][f'{cue_name}_v'] = decode_feature(row[2])

            # --- 4. BUILD SCENE-BASED SLIDING WINDOWS ---
            for clip_id, timeline in scenes.items():
                sorted_timestamps = sorted(timeline.keys())
                total_frames = len(sorted_timestamps)
                
                for i in range(0, total_frames - self.buffer_size + 1, self.stride):
                    chunk_timestamps = sorted_timestamps[i : i + self.buffer_size]
                    
                    # Find all unique people who appear anywhere in this 50-frame window
                    unique_tracks = set()
                    for t in chunk_timestamps:
                        unique_tracks.update(timeline[t].keys())
                    
                    # Limit to max_speakers to keep tensors perfectly rectangular
                    active_tracks = list(unique_tracks)[:self.max_speakers]
                    
                    self.chunks.append({
                        'clip_id': clip_id,
                        'timestamps': chunk_timestamps,
                        'tracks': active_tracks,
                        'timeline_ref': timeline # Reference to pull data from
                    })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        timestamps = chunk['timestamps']
        tracks = chunk['tracks']
        timeline = chunk['timeline_ref']
        
        # Initialize empty numpy arrays full of zeros (The Ghost Padding)
        v_out = np.zeros((self.max_speakers, self.buffer_size, 896), dtype=np.float32)
        a_out = np.zeros((self.max_speakers, self.buffer_size, 128), dtype=np.float32)
        t_out = np.zeros((self.max_speakers, self.buffer_size, 128), dtype=np.float32)
        s_out = np.zeros((self.max_speakers, self.buffer_size, 4), dtype=np.float32)
        l_out = np.zeros((self.max_speakers, self.buffer_size), dtype=np.int64)
        valid_mask = np.zeros((self.max_speakers, self.buffer_size), dtype=np.float32)
        
        # Fill the arrays with real data
        for speaker_idx, track_num in enumerate(tracks):
            for frame_idx, t in enumerate(timestamps):
                # Check if this person actually exists in this specific frame
                if track_num in timeline[t]:
                    frame = timeline[t][track_num]
                    
                    combined_visual = np.concatenate((
                        frame['face_v'], 
                        frame.get('face_body_large_v', np.zeros(128, dtype=np.float32)),
                        frame.get('face_body_v', np.zeros(128, dtype=np.float32)),
                        frame.get('face_large_v', np.zeros(128, dtype=np.float32)),
                        frame.get('background_v', np.zeros(128, dtype=np.float32)),
                        frame.get('face_small_v', np.zeros(128, dtype=np.float32)),
                        frame.get('face_down_v', np.zeros(128, dtype=np.float32))
                    ), axis=0)
                    
                    v_out[speaker_idx, frame_idx] = combined_visual
                    a_out[speaker_idx, frame_idx] = frame['raw_audio']
                    t_out[speaker_idx, frame_idx] = frame['teacher_av']
                    s_out[speaker_idx, frame_idx] = frame['spatial']
                    l_out[speaker_idx, frame_idx] = frame['label']
                    valid_mask[speaker_idx, frame_idx] = 1.0 # Mark as a valid node!
                    
        return (
            torch.FloatTensor(v_out),
            torch.FloatTensor(a_out),
            torch.FloatTensor(t_out),
            torch.FloatTensor(s_out),
            torch.LongTensor(l_out),
            torch.FloatTensor(valid_mask) # We need this for the graph builder
        )

# --- DRY RUN TEST ---
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    test_dataset = HoloPiStudentDataset(
        predata_base_dir=r"D:\FYP\MSSG\predata", 
        label_dir=r"D:\FYP\MSSG\data\AVA\ava_activespeaker_train_v1.0", 
        mode='train',
        buffer_size=50, 
        stride=25,
        max_speakers=5 # Set your max concurrent users
    )
    
    # num_workers=0 to prevent the Windows Multiprocessing crash during testing
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    for v_in, a_in, t_out, s_out, l_out, mask in test_loader:
        print("\n--- BATCH SHAPES ---")
        print(f"Visual: {v_in.shape}")       # [16, 5, 50, 896]
        print(f"Audio: {a_in.shape}")        # [16, 5, 50, 128]
        print(f"Teacher Target: {t_out.shape}") # [16, 5, 50, 128]
        print(f"Spatial: {s_out.shape}")     # [16, 5, 50, 4]
        print(f"Labels: {l_out.shape}")      # [16, 5, 50]
        print(f"Valid Mask: {mask.shape}")   # [16, 5, 50]
        break