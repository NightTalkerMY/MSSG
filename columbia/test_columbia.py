import cv2
import subprocess
import os

video_path = "./columbia/colDataPath/col.mp4"

print("--- 1. TESTING AUDIO ---")
# Extract a 5-second clip from minute 14:09 (which is around frame 25472)
# Notice I removed '-loglevel panic' so FFmpeg will print red text if there is an error!
subprocess.call(f"ffmpeg -y -i \"{video_path}\" -ss 00:14:09 -t 5 -acodec pcm_s16le -ar 16000 -ac 1 test_audio.wav", shell=True)
if os.path.exists("test_audio.wav"):
    print("-> Saved test_audio.wav! Please try to play it.")

print("\n--- 2. TESTING FACE CROP ---")
V = cv2.VideoCapture(video_path)
# Jump exactly to frame 25472
V.set(cv2.CAP_PROP_POS_FRAMES, 25472)
ret, frame = V.read()

if ret:
    cv2.imwrite("test_full_frame.jpg", frame)
    print("-> Saved test_full_frame.jpg")
    
    fw, fh = frame.shape[1], frame.shape[0]
    
    # Text file for 25472 says: TLx=0, TLy=48, width=143
    TLx, TLy, width = 0.0, 48.0, 143.0
    
    x1, y1 = max(0.0, TLx / 640.0), max(0.0, TLy / 360.0)
    x2, y2 = min(1.0, (TLx + width) / 640.0), min(1.0, (TLy + width) / 360.0)
    
    px1, py1 = max(0, int(x1 * fw)), max(0, int(y1 * fh))
    px2, py2 = min(fw, int(x2 * fw)), min(fh, int(y2 * fh))
    
    crop = frame[py1:py2, px1:px2]
    cv2.imwrite("test_crop.jpg", crop)
    print("-> Saved test_crop.jpg. Does it show a full face or a forehead?")
    
V.release()