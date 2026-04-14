import sys, os, glob, subprocess, cv2, csv

def export_gt_to_ava_csv(video_name, video_folder, pyframes_path, pywork_path):
    # The fusion folder contains the final, clean tracks for the 5+ speakers
    txtPath = os.path.join(video_folder, 'col_labels', 'fusion', '*.txt')
    txtFiles = glob.glob(txtPath)
    
    if not txtFiles:
        print(f"[ERROR] Could not find ground truth labels at {txtPath}")
        return

    # Get frame dimensions to normalize bounding boxes [0, 1]
    flist = glob.glob(os.path.join(pyframes_path, '*.jpg'))
    flist.sort()
    firstImage = cv2.imread(flist[0])
    fh, fw = firstImage.shape[0], firstImage.shape[1]
    
    csv_out_path = os.path.join(pywork_path, f"{video_name}_split.csv")
    
    with open(csv_out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        for file in txtFiles:
            identity = os.path.splitext(os.path.basename(file))[0] # e.g., 'bell', 'boll'
            lines = open(file).read().splitlines()
            
            for line in lines:
                data = line.split('\t')
                if len(data) < 5: continue
                
                # The readme states Columbia runs at 29.97 FPS, but ffmpeg extracted at 25 FPS.
                # The original script did this exact conversion to align the frames:
                frame_idx = int(int(data[0]) / 29.97 * 25)
                
                TLx = int(data[1])
                TLy = int(data[2])
                width = int(data[3])
                label = int(data[4])
                
                # Columbia boxes are square (width = height)
                x1 = max(0.0, TLx / fw)
                y1 = max(0.0, TLy / fh)
                x2 = min(1.0, (TLx + width) / fw)
                y2 = min(1.0, (TLy + width) / fh)
                
                # Format timestamp
                timestamp = "{:05d}".format(frame_idx)
                
                # Row mapping to AVA: [video_name, timestamp, x1, y1, x2, y2, dummy, identity, label]
                row = [video_name, timestamp, f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}", "0", identity, str(label)]
                writer.writerow(row)
                
    print(f"\n[SUCCESS] Exported perfect Ground Truth AVA-formatted CSV to: {csv_out_path}")

def main():
    video_name = "col"
    video_folder = "colDataPath" # Update this path to where your data lives
    video_path = os.path.join(video_folder, f"{video_name}.mp4")
    save_path = os.path.join(video_folder, video_name)
    
    pyavi_path = os.path.join(save_path, 'pyavi')
    pyframes_path = os.path.join(save_path, 'pyframes')
    pywork_path = os.path.join(save_path, 'pywork')
    
    os.makedirs(pyavi_path, exist_ok = True) 
    os.makedirs(pyframes_path, exist_ok = True) 
    os.makedirs(pywork_path, exist_ok = True) 

    # 1. Extract Video (Standardized to 25 FPS)
    video_file_path = os.path.join(pyavi_path, 'video.avi')
    if not os.path.exists(video_file_path):
        print("Extracting video to 25 FPS...")
        command = f"ffmpeg -y -i {video_path} -qscale:v 2 -threads 10 -async 1 -r 25 {video_file_path} -loglevel panic"
        subprocess.call(command, shell=True)

    # 2. Extract Audio (16000Hz)
    audio_file_path = os.path.join(pyavi_path, 'audio.wav')
    if not os.path.exists(audio_file_path):
        print("Extracting audio...")
        command = f"ffmpeg -y -i {video_file_path} -qscale:a 0 -ac 1 -vn -threads 10 -ar 16000 {audio_file_path} -loglevel panic"
        subprocess.call(command, shell=True)

    # 3. Extract Frames
    if not glob.glob(os.path.join(pyframes_path, '*.jpg')):
        print("Extracting frames...")
        command = f"ffmpeg -y -i {video_file_path} -qscale:v 2 -threads 10 -f image2 {os.path.join(pyframes_path, '%06d.jpg')} -loglevel panic"
        subprocess.call(command, shell=True)

    # 4. Generate CSV directly from Ground Truth
    export_gt_to_ava_csv(video_name, video_folder, pyframes_path, pywork_path)

if __name__ == '__main__':
    main()