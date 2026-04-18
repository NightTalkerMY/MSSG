import os
import glob
import cv2
import argparse
from collections import defaultdict
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Columbia Dataset Visual Validator")
    parser.add_argument('--videoPath', type=str, required=True, help="Path to raw mp4")
    parser.add_argument('--labelPath', type=str, required=True, help="Path to col_labels (with tracks_*)")
    parser.add_argument('--outVideo', type=str, default="col_validation.mp4", help="Output video name")
    args = parser.parse_args()

    print("1. Parsing track labels...")
    frames_to_draw = defaultdict(list)
    track_folders = glob.glob(os.path.join(args.labelPath, 'tracks_*'))
    
    if not track_folders:
        print(f"[ERROR] No tracks_* folders found in {args.labelPath}")
        return

    # Parse all text files
    for track_dir in track_folders:
        for txt_file in glob.glob(os.path.join(track_dir, '*.txt')):
            identity = os.path.splitext(os.path.basename(txt_file))[0]
            
            lines = open(txt_file).read().splitlines()
            for line in lines:
                data = line.split('\t')
                if len(data) < 5: continue
                
                frame_idx = int(data[0]) 
                TLx, TLy, width = float(data[1]), float(data[2]), float(data[3])
                is_speaking = int(data[4])
                
                # Normalize to 0-1 based on the 640x360 annotation scale
                x1, y1 = max(0.0, TLx / 640.0), max(0.0, TLy / 360.0)
                x2, y2 = min(1.0, (TLx + width) / 640.0), min(1.0, (TLy + width) / 360.0)
                
                frames_to_draw[frame_idx].append((identity, x1, y1, x2, y2, is_speaking))

    print(f"Loaded annotations for {len(frames_to_draw)} unique frames.")

    print("2. Rendering Validation Video...")
    cap = cv2.VideoCapture(args.videoPath)
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer (mp4v codec is standard for OpenCV mp4 output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.outVideo, fourcc, fps, (fw, fh))

    current_frame = 0
    pbar = tqdm(total=total_frames, desc="Processing Video")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # If we have annotations for this frame, draw them!
        if current_frame in frames_to_draw:
            for box_data in frames_to_draw[current_frame]:
                identity, x1, y1, x2, y2, is_speaking = box_data
                
                # Convert 0-1 back to actual video pixels
                px1, py1 = int(x1 * fw), int(y1 * fh)
                px2, py2 = int(x2 * fw), int(y2 * fh)
                
                # Green for speaking, Red for silent
                color = (0, 255, 0) if is_speaking == 1 else (0, 0, 255)
                status_text = "Speaking" if is_speaking == 1 else "Silent"
                label = f"{identity} ({status_text})"

                # Draw bounding box
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                
                # Draw text background for readability
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (px1, py1 - text_h - 10), (px1 + text_w, py1), color, -1)
                
                # Draw text
                cv2.putText(frame, label, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)
        current_frame += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\n[SUCCESS] Saved validation video to: {args.outVideo}")
    print("Open this file in VLC or Windows Media Player to check the dataset accuracy!")

if __name__ == '__main__':
    main()