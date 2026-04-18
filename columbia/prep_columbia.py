import sys, os, glob, subprocess, cv2, csv, argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def get_actual_resolution(video_path):
    V = cv2.VideoCapture(video_path)
    fw = int(V.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(V.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = V.get(cv2.CAP_PROP_FPS) 
    V.release()
    return fw, fh, fps

def main():
    parser = argparse.ArgumentParser(description="Columbia Master Pre-Processor (Track-Based Segments)")
    parser.add_argument('--videoName', type=str, default="col")
    parser.add_argument('--videoPath', type=str, required=True, help="Path to raw mp4")
    parser.add_argument('--labelPath', type=str, required=True, help="Path to col_labels folder (containing tracks_* directories)")
    parser.add_argument('--AVADataPath', type=str, default="./data/Columbia")
    args = parser.parse_args()

    raw_frames_dir = os.path.join(args.AVADataPath, "raw_frames", args.videoName) 
    raw_audio_dir = os.path.join(args.AVADataPath, "raw_audio")
    out_csv_dir = os.path.join(args.AVADataPath, "csv")
    out_video_clips_dir = os.path.join(args.AVADataPath, "clips_videos", "train", args.videoName)
    out_audio_clips_dir = os.path.join(args.AVADataPath, "clips_audios", "train", args.videoName)

    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(raw_audio_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_video_clips_dir, exist_ok=True)
    os.makedirs(out_audio_clips_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Extract Base Media
    # ---------------------------------------------------------
    if not glob.glob(os.path.join(raw_frames_dir, '*.jpg')):
        print("1. Extracting raw frames sequentially...")
        subprocess.call(f"ffmpeg -y -i \"{args.videoPath}\" -qscale:v 2 -threads 10 -f image2 \"{os.path.join(raw_frames_dir, '%06d.jpg')}\" -loglevel panic", shell=True)
    
    raw_audio_file = os.path.join(raw_audio_dir, f"{args.videoName}.wav")
    if not os.path.exists(raw_audio_file):
        print("1. Extracting raw audio...")
        subprocess.call(f"ffmpeg -y -i \"{args.videoPath}\" -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 \"{raw_audio_file}\" -loglevel panic", shell=True)

    # ---------------------------------------------------------
    # STEP 2: Parse Segmented "Tracks" Folders
    # ---------------------------------------------------------
    fw, fh, exact_fps = get_actual_resolution(args.videoPath)
    print(f"2. Detected video: {fw}x{fh} @ {exact_fps:.4f} FPS. Parsing track folders...")
    
    raw_rows = []
    frames_to_crop = defaultdict(list)
    
    # Iterate through every tracks_xxx_xxx folder instead of fusion!
    track_folders = glob.glob(os.path.join(args.labelPath, 'tracks_*'))
    if not track_folders:
        print(f"[ERROR] Could not find 'tracks_*' folders in {args.labelPath}")
        return

    for track_dir in track_folders:
        track_name = os.path.basename(track_dir) # e.g., 'tracks_25000_35000'
        txtFiles = glob.glob(os.path.join(track_dir, '*.txt'))
        
        for file in txtFiles:
            base_identity = os.path.splitext(os.path.basename(file))[0]
            
            # THE MAGIC FIX: Create a unique entity ID for this specific person in this specific scene!
            entity = f"{base_identity}_{track_name}" 
            os.makedirs(os.path.join(out_video_clips_dir, entity), exist_ok=True)
            
            lines = open(file).read().splitlines()
            for line in lines:
                data = line.split('\t')
                if len(data) < 5: continue
                
                frame_idx = int(data[0]) 
                TLx, TLy, width, label = float(data[1]), float(data[2]), float(data[3]), int(data[4])
                
                x1, y1 = max(0.0, TLx / 640.0), max(0.0, TLy / 360.0)
                x2, y2 = min(1.0, (TLx + width) / 640.0), min(1.0, (TLy + width) / 360.0)
                timestamp_str = f"{(frame_idx / exact_fps):.3f}"
                
                raw_rows.append([args.videoName, timestamp_str, x1, y1, x2, y2, 0, entity, label, entity])
                frames_to_crop[frame_idx].append((timestamp_str, entity, x1, y1, x2, y2))

    # Save Segmented CSVs
    header = ["video_id","frame_timestamp","entity_box_x1","entity_box_y1","entity_box_x2","entity_box_y2","label","entity_id","label_id","instance_id"]
    df = pd.DataFrame(raw_rows, columns=header).drop_duplicates(subset=['entity_id', 'frame_timestamp'], keep='first')
    df.to_csv(os.path.join(out_csv_dir, 'train_orig.csv'), index=False)
    
    with open(os.path.join(out_csv_dir, 'train_loader.csv'), 'w') as f:
        for entity_id, group in df.groupby('entity_id'):
            labels_str = str(list(group.sort_values('frame_timestamp')['label_id'].astype(int))).replace(' ', '')
            f.write(f"{entity_id}\t{len(group)}\t{exact_fps:.2f}\t{labels_str}\n")

    # ---------------------------------------------------------
    # STEP 3: Sequential Video Cropping
    # ---------------------------------------------------------
    print("3. Cropping faces (Zero-Drift Sequential Mode)...")
    V = cv2.VideoCapture(args.videoPath)
    total_frames = int(V.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0 
    
    pbar = tqdm(total=total_frames, desc="Decoding Video")
    while True:
        ret, frame = V.read()
        if not ret: break
        
        if current_frame in frames_to_crop:
            for crop_data in frames_to_crop[current_frame]:
                timestamp_str, entity, x1, y1, x2, y2 = crop_data
                
                px1, py1 = max(0, int(x1 * fw)), max(0, int(y1 * fh))
                px2, py2 = min(fw, int(x2 * fw)), min(fh, int(y2 * fh))
                
                face_crop = frame[py1:py2, px1:px2]
                if face_crop.size > 0:
                    cv2.imwrite(os.path.join(out_video_clips_dir, entity, f"{timestamp_str}.jpg"), face_crop)
                    
        current_frame += 1
        pbar.update(1)
        
    V.release()
    pbar.close()

    # ---------------------------------------------------------
    # STEP 4: Segmented Audio Slicing
    # ---------------------------------------------------------
    print("4. Slicing audio tracklets...")
    for entity, insData in df.groupby('entity_id'):
        start_time = float(insData.iloc[0]['frame_timestamp'])
        end_time = float(insData.iloc[-1]['frame_timestamp'])
        duration = (end_time - start_time) + 0.1
        
        out_wav = os.path.join(out_audio_clips_dir, f"{entity}.wav")
        subprocess.call(f"ffmpeg -y -i \"{raw_audio_file}\" -ss {start_time} -t {duration} -acodec pcm_s16le -ar 16000 -ac 1 \"{out_wav}\" -loglevel panic", shell=True)

    print("\n[SUCCESS] Pipeline Complete! Audio and Faces are perfectly segmented like AVA.")

if __name__ == '__main__':
    main()