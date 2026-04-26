import os
import cv2
import time
import torch
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=UserWarning)

from proposed.data_streamer import RemoteVideoClient, FileVideoClient
from proposed.online_tracker import OnlineTracker
from model.faceDetector.s3fd import S3FD

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "FILE"  
VIDEO_FILE = "columbia/col.mp4"

# --- DYNAMIC FPS MATH ---
cap = cv2.VideoCapture(VIDEO_FILE)
ACTUAL_FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()
print(f"Detected exact video FPS: {ACTUAL_FPS:.2f}")

# TOGGLE THIS TO SEE S3FD IMPACT
# 1 = Run S3FD every frame (Max GPU load)
# 3 = Run S3FD every 3 frames (Lower GPU load)
DETECT_INTERVAL = 1    

# ==========================================
# MAIN PIPELINE (S3FD ISOLATION TEST)
# ==========================================
def main():
    if MODE == "REMOTE":
        video_client = RemoteVideoClient()
    else:
        video_client = FileVideoClient(VIDEO_FILE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading S3FD on {device}...")
    
    # Init ONLY the Face Detector and Tracker
    face_detector = S3FD(device='cuda')
    tracker = OnlineTracker(iou_threshold=0.4, max_lost=2, alpha=0.8)
    
    frame_idx = 0
    print("\nStarting S3FD Isolation Test... Press 'q' to quit.")
    print("OPEN TASK MANAGER -> PERFORMANCE -> GPU. Watch your '3D' utilization.")
    
    video_start_time = None 

    with torch.no_grad():
        while True:
            frame = video_client.get_frame()
            if frame is None: break
            
            display_frame = frame.copy()
            
            # 1. DETECT & TRACK
            t0 = time.time()
            if frame_idx % DETECT_INTERVAL == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # This is the single line we are profiling for GPU usage:
                raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
                last_detections = [det[:-1].tolist() for det in raw_detections]
            else:
                # If we skip a frame, we pass an empty list to the tracker to force it to guess
                last_detections = []
            
            active_tracks = tracker.update(last_detections)
            t1 = time.time()
            
            # Print execution time to console
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}: S3FD + Tracking took {(t1-t0)*1000:.1f} ms")

            # 2. DRAWING
            for track_id, bbox in active_tracks.items():
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw standard blue box for tracking visualization
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
                cv2.putText(display_frame, f"ID:{track_id}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            cv2.imshow("S3FD Isolation Profiler", display_frame)

            if MODE == "FILE":
                if video_start_time is None:
                    video_start_time = time.time() - (frame_idx / ACTUAL_FPS)
                expected_time = video_start_time + (frame_idx / ACTUAL_FPS)
                current_time = time.time()
                if current_time < expected_time:
                    time.sleep(expected_time - current_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    video_client.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()