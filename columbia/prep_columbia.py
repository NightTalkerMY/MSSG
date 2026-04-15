import sys, os, glob, subprocess, cv2, csv, argparse

def export_gt_to_ava_csv(video_name, label_folder, pyframes_path, csv_out_path):
    txtPath = os.path.join(label_folder, '*.txt')
    txtFiles = glob.glob(txtPath)
    
    if not txtFiles:
        print(f"[ERROR] Could not find ground truth labels at {txtPath}")
        return

    # Get frame dimensions to normalize bounding boxes [0, 1]
    flist = glob.glob(os.path.join(pyframes_path, '*.jpg'))
    flist.sort()
    if not flist:
        print(f"[ERROR] No frames found in {pyframes_path}")
        return
        
    firstImage = cv2.imread(flist[0])
    fh, fw = firstImage.shape[0], firstImage.shape[1]
    
    with open(csv_out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        for file in txtFiles:
            identity = os.path.splitext(os.path.basename(file))[0]
            lines = open(file).read().splitlines()
            
            for line in lines:
                data = line.split('\t')
                if len(data) < 5: continue
                
                # Convert 29.97 FPS label timing to 25 FPS video timing
                frame_idx = int(int(data[0]) / 29.97 * 25)
                
                TLx = int(data[1])
                TLy = int(data[2])
                width = int(data[3])
                label = int(data[4])
                
                # Normalize square bounding box
                x1 = max(0.0, TLx / 640.0)
                y1 = max(0.0, TLy / 360.0)
                x2 = min(1.0, (TLx + width) / 640.0)
                y2 = min(1.0, (TLy + width) / 360.0)
                
                timestamp = f"{frame_idx / 25.0:.3f}"
                
                # Format: [video_name, timestamp, x1, y1, x2, y2, dummy, identity, label]
                # row = [video_name, timestamp, f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}", "0", identity, str(label)]
                row = [video_name, timestamp, f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}", "0", identity, str(label), identity]
                writer.writerow(row)
                
    print(f"[SUCCESS] CSV Exported to: {csv_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Columbia Automated Pre-Processor")
    parser.add_argument('--videoName', type=str, default="col")
    parser.add_argument('--videoPath', type=str, required=True, help="Path to raw mp4")
    parser.add_argument('--labelPath', type=str, required=True, help="Path to fusion txt folder")
    # These point to the exact folders the bash script expects
    parser.add_argument('--outFramesDir', type=str, default="./data/Columbia/clips_videos")
    parser.add_argument('--outAudioDir', type=str, default="./data/Columbia/clips_audios")
    parser.add_argument('--outCsvDir', type=str, default="./predata/csv_split/columbia")
    args = parser.parse_args()

    # Target directories
    target_frames_dir = os.path.join(args.outFramesDir, args.videoName)
    os.makedirs(target_frames_dir, exist_ok=True)
    os.makedirs(args.outAudioDir, exist_ok=True)
    os.makedirs(args.outCsvDir, exist_ok=True)

    # 1. Extract Video to Target Frame Directory
    if not glob.glob(os.path.join(target_frames_dir, '*.jpg')):
        print(f"Extracting frames directly to {target_frames_dir}...")
        command = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads 10 -r 25 -f image2 {os.path.join(target_frames_dir, '%06d.jpg')} -loglevel panic"
        subprocess.call(command, shell=True)
    else:
        print("Frames already extracted. Skipping...")

    # 2. Extract Audio (For node initialization later)
    target_audio_file = os.path.join(args.outAudioDir, f"{args.videoName}.wav")
    if not os.path.exists(target_audio_file):
        print("Extracting audio...")
        command = f"ffmpeg -y -i {args.videoPath} -qscale:a 0 -ac 1 -vn -threads 10 -ar 16000 {target_audio_file} -loglevel panic"
        subprocess.call(command, shell=True)

    # 3. Generate CSV
    target_csv_file = os.path.join(args.outCsvDir, f"{args.videoName}_split.csv")
    print("Generating AVA-style CSV...")
    export_gt_to_ava_csv(args.videoName, args.labelPath, target_frames_dir, target_csv_file)

if __name__ == '__main__':
    main()