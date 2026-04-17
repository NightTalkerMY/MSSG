import sys, os, glob, subprocess, cv2, csv, argparse
import pandas as pd

def process_labels_and_csvs(video_name, label_folder, pyframes_path, csv_dir):
    txtPath = os.path.join(label_folder, '*.txt')
    txtFiles = glob.glob(txtPath)
    
    if not txtFiles:
        print(f"[ERROR] Could not find ground truth labels at {txtPath}")
        return

    # 1. Get ACTUAL frame dimensions to normalize bounding boxes [0, 1]
    flist = glob.glob(os.path.join(pyframes_path, '*.jpg'))
    flist.sort()
    if not flist:
        print(f"[ERROR] No frames found in {pyframes_path}")
        return
        
    firstImage = cv2.imread(flist[0])
    fh, fw = firstImage.shape[0], firstImage.shape[1]
    print(f"Detected actual video resolution: {fw}x{fh}")
    
    # Setup CSV paths
    train_orig_path = os.path.join(csv_dir, 'train_orig.csv')
    train_loader_path = os.path.join(csv_dir, 'train_loader.csv')
    
    # 2. Write the raw data to train_orig.csv
    print(f"Writing parsed labels to {train_orig_path}...")
    header = ["video_id","frame_timestamp","entity_box_x1","entity_box_y1","entity_box_x2","entity_box_y2","label","entity_id","label_id","instance_id"]
    
    raw_rows = []
    for file in txtFiles:
        identity = os.path.splitext(os.path.basename(file))[0]
        lines = open(file).read().splitlines()
        
        for line in lines:
            data = line.split('\t')
            if len(data) < 5: continue
            
            # Convert 29.97 FPS label timing to 25 FPS video timing
            frame_idx = int(int(data[0]) / 29.97 * 25)
            
            TLx, TLy = float(data[1]), float(data[2])
            width = float(data[3])
            label = int(data[4])
            
            # Dynamically normalize based on actual frame width/height!
            x1 = max(0.0, TLx / fw)
            y1 = max(0.0, TLy / fh)
            x2 = min(1.0, (TLx + width) / fw)
            y2 = min(1.0, (TLy + width) / fh)
            
            timestamp = f"{frame_idx / 25.0:.3f}"
            
            # Format matches the header
            raw_rows.append([video_name, timestamp, f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}", 0, identity, label, identity])
            
    # Write to Pandas to handle the desync deduplication easily
    df = pd.DataFrame(raw_rows, columns=header)
    
    # 3. Deduplicate (The fix_desync logic)
    print("Deduplicating 29.97fps annotations to match 25fps frames...")
    df_dedup = df.drop_duplicates(subset=['entity_id', 'frame_timestamp'], keep='first')
    df_dedup.to_csv(train_orig_path, index=False)
    print(f" -> Cleaned rows from {len(df)} to {len(df_dedup)}!")
    
    # 4. Generate the DataLoader CSV (The prep_loader_csv logic)
    print(f"Generating {train_loader_path} for PyTorch...")
    with open(train_loader_path, 'w') as f:
        for entity_id, group in df_dedup.groupby('entity_id'):
            group = group.sort_values('frame_timestamp')
            num_frames = len(group)
            
            labels = list(group['label_id'].astype(int))
            labels_str = str(labels).replace(' ', '') # Format: [0,0,1,1,...]
            
            line = f"{entity_id}\t{num_frames}\t25\t{labels_str}\n"
            f.write(line)
            
    print("\n[SUCCESS] All CSVs generated perfectly in AVA structure!")

def main():
    parser = argparse.ArgumentParser(description="Columbia Automated Pre-Processor")
    parser.add_argument('--videoName', type=str, default="col")
    parser.add_argument('--videoPath', type=str, required=True, help="Path to raw mp4")
    parser.add_argument('--labelPath', type=str, required=True, help="Path to fusion txt folder")
    
    # We set default outputs to MATCH the AVA structure directly
    parser.add_argument('--AVADataPath', type=str, default="./data/Columbia")
    args = parser.parse_args()

    # Define exact target directories based on AVA standard
    target_frames_dir = os.path.join(args.AVADataPath, "clips_videos", "train", args.videoName)
    target_audio_dir = os.path.join(args.AVADataPath, "clips_audios", "train")
    target_csv_dir = os.path.join(args.AVADataPath, "csv")
    
    os.makedirs(target_frames_dir, exist_ok=True)
    os.makedirs(target_audio_dir, exist_ok=True)
    os.makedirs(target_csv_dir, exist_ok=True)

    # 1. Extract Video to Target Frame Directory
    if not glob.glob(os.path.join(target_frames_dir, '*.jpg')):
        print(f"Extracting frames directly to {target_frames_dir}...")
        command = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads 10 -r 25 -f image2 {os.path.join(target_frames_dir, '%06d.jpg')} -loglevel panic"
        subprocess.call(command, shell=True)
    else:
        print("Frames already extracted. Skipping...")

    # 2. Extract Audio
    target_audio_file = os.path.join(target_audio_dir, f"{args.videoName}.wav")
    if not os.path.exists(target_audio_file):
        print(f"Extracting audio to {target_audio_file}...")
        command = f"ffmpeg -y -i {args.videoPath} -qscale:a 0 -ac 1 -vn -threads 10 -ar 16000 {target_audio_file} -loglevel panic"
        subprocess.call(command, shell=True)
    else:
        print("Audio already extracted. Skipping...")

    # 3. Generate CSVs directly into the AVA folders
    process_labels_and_csvs(args.videoName, args.labelPath, target_frames_dir, target_csv_dir)

if __name__ == '__main__':
    main()