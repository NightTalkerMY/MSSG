import os
import cv2
import csv
from tqdm import tqdm

def crop_baseline_faces():
    csv_path = './data/Columbia/csv/train_orig.csv'
    # FIXED: Added the missing /train/ folder to the path!
    raw_frames_dir = './data/Columbia/clips_videos/train/col'
    out_base_dir = './data/Columbia/clips_videos_faces/train/col'

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skips the header row
        rows = list(reader)

    print(f"Cropping {len(rows)} tight faces for the face model...")

    for row in tqdm(rows):
        if len(row) < 9:
            continue

        video_name = row[0]
        timestamp = float(row[1])
        x1, y1 = float(row[2]), float(row[3])
        x2, y2 = float(row[4]), float(row[5])
        identity = row[7]

        frame_num = int(round(timestamp * 25))
        if frame_num == 0: 
            frame_num = 1
        
        frame_path = f"{raw_frames_dir}/{frame_num:06d}.jpg"
        if not os.path.exists(frame_path):
             frame_path = f"{raw_frames_dir}/{frame_num+1:06d}.jpg"
        
        # NEW: Suppress OpenCV spam by checking if file actually exists FIRST
        if not os.path.exists(frame_path):
            continue
            
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        px1 = max(0, int(x1 * w))
        py1 = max(0, int(y1 * h))
        px2 = min(w, int(x2 * w))
        py2 = min(h, int(y2 * h))
        
        face_crop = img[py1:py2, px1:px2]
        if face_crop.size == 0:
            continue
            
        out_dir = os.path.join(out_base_dir, identity)
        os.makedirs(out_dir, exist_ok=True)
        
        out_filename = f"{timestamp:.3f}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_filename), face_crop)

if __name__ == '__main__':
    crop_baseline_faces()