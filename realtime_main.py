# import cv2
# import time
# import torch
# import numpy as np
# import python_speech_features
# from collections import deque

# # Import your custom modules
# from data_streamer import RemoteVideoClient, RemoteAudioClient, FileVideoClient, FileAudioClient
# from online_tracker import OnlineTracker

# # Import original paper's modules (do not modify these!)
# from model.faceDetector.s3fd import S3FD
# from ASD import ASD

# # ==========================================
# # CONFIGURATION
# # ==========================================
# MODE = "FILE"  # Change to "REMOTE" when using your Tailscale Pi/Laptop setup
# VIDEO_FILE = "colDataPath/col.mp4"
# AUDIO_FILE = "colDataPath/col.wav"

# SENSITIVITY = 0.0    # Lower to -1.0 or -2.0 if it struggles to hear you
# DETECT_INTERVAL = 3  # Only run face detector every 3rd frame
# LOOKAHEAD = 15
# HISTORY = 15
# WINDOW_SIZE = HISTORY + 1 + LOOKAHEAD  # 31 frames total
# FPS = 25
# AUDIO_SAMPLE_RATE = 16000
# SAMPLES_PER_WINDOW = int((WINDOW_SIZE / FPS) * AUDIO_SAMPLE_RATE) # 19840 samples

# def crop_and_preprocess(frame, bbox, crop_scale=0.40):
#     """Crops the face using the exact 0.40 scaling from the original paper."""
#     x1, y1, x2, y2 = bbox
    
#     # Calculate center and size
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2
#     size = max(x2 - x1, y2 - y1) / 2  
    
#     # Calculate padding amount 
#     pad_amount = int(size * (1 + 2 * crop_scale))
    
#     # Pad the frame with 110 (as done in the original offline script)
#     frame_padded = np.pad(frame, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 
#                           'constant', constant_values=(110, 110))
                          
#     # Recalculate center coordinates with padding offset
#     my = center_y + pad_amount
#     mx = center_x + pad_amount
    
#     # Execute the exact crop bounds from the original script
#     y_start = int(my - size)
#     y_end = int(my + size * (1 + 2 * crop_scale))
#     x_start = int(mx - size * (1 + crop_scale))
#     x_end = int(mx + size * (1 + crop_scale))
    
#     face = frame_padded[y_start:y_end, x_start:x_end]
    
#     if face.size == 0:
#         return np.zeros((112, 112), dtype=np.uint8)
        
#     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     face_resized = cv2.resize(face_gray, (224, 224))
    
#     # Extreme center crop to 112x112
#     center_crop = face_resized[56:168, 56:168]
    
#     return center_crop

# def main():
#     # 1. Initialize Data Sources
#     if MODE == "REMOTE":
#         video_client = RemoteVideoClient()
#         audio_client = RemoteAudioClient()
#     else:
#         video_client = FileVideoClient(VIDEO_FILE)
#         audio_client = FileAudioClient(AUDIO_FILE)

#     # 2. Initialize Models & Tracker
#     print("Loading AI Models onto GPU...")
#     face_detector = S3FD(device='cuda')
#     asd_model = ASD()
#     asd_model.loadParameters("weight/pretrain_AVA.model")
#     asd_model.eval()
    
#     tracker = OnlineTracker(iou_threshold=0.4, max_lost=5)

#     # 3. Initialize Buffers
#     track_history = {}      # {track_id: deque(maxlen=31) of face crops}
#     bbox_history = {}       # {track_id: deque(maxlen=31) of raw bounding boxes}
#     display_buffer = deque(maxlen=LOOKAHEAD + 1) # Holds full frames to delay video for sync
    
#     frame_idx = 0
#     last_detections = [] 
#     print("\nStarting Real-Time Pipeline... Press 'q' to quit.")
    
#     video_start_time = None # Absolute clock reference for FILE mode pacer

#     with torch.no_grad():
#         while True:
#             # --- A. Fetch Data ---
#             frame = video_client.get_frame()
#             if frame is None:
#                 print("End of stream.")
#                 break
                
#             display_buffer.append(frame.copy())
            
#             # --- B. Face Detection & Tracking ---
#             # Only run the heavy S3FD model every N frames
#             if frame_idx % DETECT_INTERVAL == 0:
#                 img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
#                 last_detections = [det[:-1].tolist() for det in raw_detections]
            
#             # The tracker uses the fresh detection, OR the last known detection on skipped frames
#             active_tracks = tracker.update(last_detections)
            
#             # --- C. Update Sliding Windows ---
#             # Remove stale tracks
#             stale_ids = [tid for tid in track_history.keys() if tid not in active_tracks]
#             for tid in stale_ids:
#                 del track_history[tid]
#                 del bbox_history[tid]

#             # Append current faces
#             for track_id, bbox in active_tracks.items():
#                 face_crop = crop_and_preprocess(frame, bbox)
                
#                 if track_id not in track_history:
#                     track_history[track_id] = deque(maxlen=WINDOW_SIZE)
#                     bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)
                    
#                 track_history[track_id].append(face_crop)
#                 bbox_history[track_id].append(bbox)

#             # --- D. Run Inference (If we have a full window) ---
#             # We delay the display by LOOKAHEAD frames so the visual matches the AI prediction
#             if len(display_buffer) == LOOKAHEAD + 1:
#                 display_frame = display_buffer[0] # This is the "Center Frame" (Frame X - 15)
                
#                 # Fetch Audio
#                 if MODE == "REMOTE":
#                     raw_audio = audio_client.get_latest_audio(SAMPLES_PER_WINDOW)
#                 else:
#                     raw_audio = audio_client.get_latest_audio(SAMPLES_PER_WINDOW, frame_idx)

#                 # Process each person
#                 for track_id, face_deque in track_history.items():
#                     if len(face_deque) == WINDOW_SIZE and raw_audio is not None:
                        
#                         # --- 1. SMART AUDIO FORMATTING ---
#                         # Check if Tailscale audio arrived as float32 [-1.0, 1.0] and scale to PCM [-32768, 32767]
#                         if np.max(np.abs(raw_audio)) <= 2.0:
#                             scaled_audio = raw_audio * 32767.0
#                         else:
#                             scaled_audio = raw_audio

#                         # --- 2. NOISE GATE & GENTLE GAIN ---
#                         max_amp = np.max(np.abs(scaled_audio))
#                         NOISE_FLOOR = 1000.0  # Adjust this: Increase if static triggers it, decrease if it ignores your voice.
                        
#                         if max_amp > NOISE_FLOOR:
#                             # You are speaking. Apply a safe 2x volume boost (capped to prevent distortion).
#                             gain = min(32767.0 / max_amp, 2.0) 
#                             normalized_audio = scaled_audio * gain
#                         else:
#                             # Room is quiet. Mute the static completely to feed the AI perfect silence.
#                             normalized_audio = scaled_audio * 0.0

#                         # 3. Extract MFCC 
#                         audio_mfcc = python_speech_features.mfcc(
#                             normalized_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010
#                         )
                        
#                         # 4. STRICT 4:1 ALIGNMENT
#                         expected_audio_frames = WINDOW_SIZE * 4  # 31 * 4 = 124
                        
#                         if audio_mfcc.shape[0] < expected_audio_frames:
#                             pad_length = expected_audio_frames - audio_mfcc.shape[0]
#                             audio_mfcc = np.pad(audio_mfcc, ((0, pad_length), (0, 0)), mode='edge')
#                         elif audio_mfcc.shape[0] > expected_audio_frames:
#                             audio_mfcc = audio_mfcc[:expected_audio_frames, :]
                        
#                         # 5. Prepare Tensors
#                         video_tensor = np.array(face_deque) 
#                         input_v = torch.FloatTensor(video_tensor).unsqueeze(0).cuda()
#                         input_a = torch.FloatTensor(audio_mfcc).unsqueeze(0).cuda()
                        
#                         # 6. Forward Pass
#                         embedA = asd_model.model.forward_audio_frontend(input_a)
#                         embedV = asd_model.model.forward_visual_frontend(input_v)
#                         out = asd_model.model.forward_audio_visual_backend(embedA, embedV)

#                         # 7. Extract Score & Apply Output Smoothing
#                         scores = asd_model.lossAV.forward(out, labels=None)
                        
#                         # Median smoothing prevents micro-pauses from breaking detection
#                         smoothed_score = np.median(scores[HISTORY - 2 : HISTORY + 3])
                        
#                         target_score = float(smoothed_score)
                        
#                         # Apply sensitivity threshold
#                         is_speaking = target_score > SENSITIVITY
                        
#                         # --- E. Draw Results on the Display Frame ---
#                         target_bbox = bbox_history[track_id][HISTORY] 
#                         x1, y1, x2, y2 = map(int, target_bbox)
                        
#                         color = (0, 255, 0) if is_speaking else (0, 0, 255)
#                         label = f"ID:{track_id} Spk:{target_score:.1f}" if is_speaking else f"ID:{track_id} Sil"
                        
#                         cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
#                         cv2.putText(display_frame, label, (x1, max(30, y1 - 10)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#                 cv2.imshow("LR-ASD Real-Time Prototype", display_frame)

#                 # --- BULLETPROOF ABSOLUTE PACER ---
#                 if MODE == "FILE":
#                     # Anchor the clock ONLY when the first frame hits the screen
#                     if video_start_time is None:
#                         video_start_time = time.time() - (frame_idx / FPS)

#                     # Calculate the exact real-world time this frame *should* appear
#                     expected_time = video_start_time + (frame_idx / FPS)
#                     current_time = time.time()
                    
#                     if current_time < expected_time:
#                         # Sleep for the exact remainder to strictly lock the framerate
#                         time.sleep(expected_time - current_time)
                        
#                     key = cv2.waitKey(1) & 0xFF
#                 else:
#                     key = cv2.waitKey(1) & 0xFF
                    
#                 if key == ord('q'):
#                     break

#             frame_idx += 1

#     # Clean up
#     video_client.close()
#     audio_client.close()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import os
import cv2
import glob
import time
import torch
import numpy as np
import python_speech_features
from collections import deque
from sklearn.metrics import f1_score, accuracy_score

# Import your custom modules
from data_streamer import RemoteVideoClient, RemoteAudioClient, FileVideoClient, FileAudioClient
from online_tracker import OnlineTracker, calculate_iou

# Import original paper's modules
from model.faceDetector.s3fd import S3FD
from ASD import ASD

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "FILE"  
VIDEO_FILE = "colDataPath/col.mp4"
AUDIO_FILE = "colDataPath/col.wav"
LABELS_FOLDER = "colDataPath/col_labels/fusion" # Path to GT labels

SENSITIVITY = 0.0    
DETECT_INTERVAL = 3  
LOOKAHEAD = 15
HISTORY = 15
WINDOW_SIZE = HISTORY + 1 + LOOKAHEAD  
FPS = 25
AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_WINDOW = int((WINDOW_SIZE / FPS) * AUDIO_SAMPLE_RATE) 

def crop_and_preprocess(frame, bbox, crop_scale=0.40):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) / 2  
    pad_amount = int(size * (1 + 2 * crop_scale))
    
    frame_padded = np.pad(frame, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 
                          'constant', constant_values=(110, 110))
                          
    my = center_y + pad_amount
    mx = center_x + pad_amount
    
    y_start = int(my - size)
    y_end = int(my + size * (1 + 2 * crop_scale))
    x_start = int(mx - size * (1 + crop_scale))
    x_end = int(mx + size * (1 + crop_scale))
    
    face = frame_padded[y_start:y_end, x_start:x_end]
    if face.size == 0:
        return np.zeros((112, 112), dtype=np.uint8)
        
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (224, 224))
    center_crop = face_resized[56:168, 56:168]
    return center_crop

def eval_iou(boxA, boxB):
    """Calculates the specific overlapping metric used for the Columbia dataset."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    
    if boxAArea == 0:
        return 0.0
    # Original paper calculates intersection over Prediction Area for Columbia
    return interArea / float(boxAArea)

def evaluate_live_pipeline(live_predictions, labels_folder):
    """Cross-references our live pipeline's predictions against the Columbia GT."""
    print("\n--- RUNNING COLUMBIA EVALUATION ---")
    txtPath = os.path.join(labels_folder, '*.txt')
    predictionSet = {name: [[],[]] for name in ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas']}
    dictGT = {}
    
    # 1. Load Ground Truth
    txtFiles = glob.glob(txtPath)
    if not txtFiles:
        print(f"Warning: No ground truth files found in {labels_folder}. Skipping evaluation.")
        return

    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = os.path.splitext(os.path.basename(file))[0]
        for line in lines:
            data = line.split('\t')
            frame = int(int(data[0]) / 29.97 * 25) # Convert 29.97fps GT to 25fps
            x1, y1 = int(data[1]), int(data[2])
            x2, y2 = x1 + int(data[3]), y1 + int(data[3])
            gt = int(data[4])
            
            if frame not in dictGT:
                dictGT[frame] = []
            dictGT[frame].append([x1, y1, x2, y2, gt, idName])

    # 2. Match Live Predictions to Ground Truth
    for fidx in dictGT:
        for gtThisFrame in dictGT[fidx]:
            faceGT = gtThisFrame[0:4]
            labelGT = gtThisFrame[4]
            idGT = gtThisFrame[5]
            
            ious = []
            if fidx in live_predictions:
                for pred in live_predictions[fidx]:
                    # --- THE FIX: SCALE AND CUSTOM IOU ---
                    # Columbia GT is in half-resolution, so we // 2 our coordinates
                    pred_bbox_scaled = [v // 2 for v in pred['bbox']]
                    iou = eval_iou(pred_bbox_scaled, faceGT)
                    
                    if iou > 0.5:
                        ious.append([iou, pred['score']])
                        
            if len(ious) > 0:
                ious.sort()
                labelPredict = ious[-1][1] # Best matching bounding box
            else:                   
                labelPredict = -1.0 # Failed to detect face
                
            predictionSet[idGT][0].append(labelPredict)
            predictionSet[idGT][1].append(labelGT)

    # 3. Calculate Final F1 Score
    names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas']
    names.sort()
    F1s = 0
    for i in names:
        scores = np.array(predictionSet[i][0])
        labels = np.array(predictionSet[i][1])
        scores = np.int64(scores > SENSITIVITY) # Threshold the predictions
        
        if len(labels) > 0:
            F1 = f1_score(labels, scores)
            ACC = accuracy_score(labels, scores)
            if i != 'abbas':
                F1s += F1
                print(f"{i.capitalize()} -> ACC: {100 * ACC:.2f}%, F1: {100 * F1:.2f}%")
                
    print(f"\n>> LIVE PIPELINE AVERAGE F1: {100 * (F1s / 5):.2f}% <<")


def main():
    if MODE == "REMOTE":
        video_client = RemoteVideoClient()
        audio_client = RemoteAudioClient()
    else:
        video_client = FileVideoClient(VIDEO_FILE)
        audio_client = FileAudioClient(AUDIO_FILE)

    print("Loading AI Models onto GPU...")
    face_detector = S3FD(device='cuda')
    asd_model = ASD()
    asd_model.loadParameters("weight/pretrain_AVA.model")
    asd_model.eval()
    tracker = OnlineTracker(iou_threshold=0.4, max_lost=5)

    track_history = {}      
    bbox_history = {}       
    display_buffer = deque(maxlen=LOOKAHEAD + 1) 
    
    frame_idx = 0
    last_detections = [] 
    live_predictions = {} # <--- EVALUATION LOGGER
    
    print("\nStarting Real-Time Pipeline... Press 'q' to quit.")
    video_start_time = None 

    with torch.no_grad():
        while True:
            loop_start_time = time.time()
            frame = video_client.get_frame()
            if frame is None:
                break
                
            display_buffer.append(frame.copy())
            
            if frame_idx % DETECT_INTERVAL == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
                last_detections = [det[:-1].tolist() for det in raw_detections]
            
            active_tracks = tracker.update(last_detections)
            
            stale_ids = [tid for tid in track_history.keys() if tid not in active_tracks]
            for tid in stale_ids:
                del track_history[tid]
                del bbox_history[tid]

            for track_id, bbox in active_tracks.items():
                face_crop = crop_and_preprocess(frame, bbox)
                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=WINDOW_SIZE)
                    bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)
                track_history[track_id].append(face_crop)
                bbox_history[track_id].append(bbox)

            if len(display_buffer) == LOOKAHEAD + 1:
                display_frame = display_buffer[0] 
                target_frame_idx = frame_idx - LOOKAHEAD # The actual frame we are evaluating
                
                if MODE == "REMOTE":
                    raw_audio = audio_client.get_latest_audio(SAMPLES_PER_WINDOW)
                else:
                    raw_audio = audio_client.get_latest_audio(SAMPLES_PER_WINDOW, frame_idx)

                for track_id, face_deque in track_history.items():
                    if len(face_deque) == WINDOW_SIZE and raw_audio is not None:
                        
                        if np.max(np.abs(raw_audio)) <= 2.0:
                            scaled_audio = raw_audio * 32767.0
                        else:
                            scaled_audio = raw_audio

                        max_amp = np.max(np.abs(scaled_audio))
                        NOISE_FLOOR = 1000.0  
                        if max_amp > NOISE_FLOOR:
                            gain = min(32767.0 / max_amp, 2.0) 
                            normalized_audio = scaled_audio * gain
                        else:
                            normalized_audio = scaled_audio * 0.0

                        audio_mfcc = python_speech_features.mfcc(
                            normalized_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010
                        )
                        
                        expected_audio_frames = WINDOW_SIZE * 4  
                        if audio_mfcc.shape[0] < expected_audio_frames:
                            pad_length = expected_audio_frames - audio_mfcc.shape[0]
                            audio_mfcc = np.pad(audio_mfcc, ((0, pad_length), (0, 0)), mode='edge')
                        elif audio_mfcc.shape[0] > expected_audio_frames:
                            audio_mfcc = audio_mfcc[:expected_audio_frames, :]
                        
                        video_tensor = np.array(face_deque) 
                        input_v = torch.FloatTensor(video_tensor).unsqueeze(0).cuda()
                        input_a = torch.FloatTensor(audio_mfcc).unsqueeze(0).cuda()
                        
                        embedA = asd_model.model.forward_audio_frontend(input_a)
                        embedV = asd_model.model.forward_visual_frontend(input_v)
                        out = asd_model.model.forward_audio_visual_backend(embedA, embedV)

                        scores = asd_model.lossAV.forward(out, labels=None)
                        smoothed_score = np.median(scores[HISTORY - 2 : HISTORY + 3])
                        target_score = float(smoothed_score)
                        
                        # --- LOG PREDICTION FOR EVALUATION ---
                        if target_frame_idx not in live_predictions:
                            live_predictions[target_frame_idx] = []
                            
                        target_bbox = bbox_history[track_id][HISTORY] 
                        x1, y1, x2, y2 = map(int, target_bbox)
                        
                        live_predictions[target_frame_idx].append({
                            'bbox': [x1, y1, x2, y2],
                            'score': target_score
                        })
                        
                        # --- DRAW ON SCREEN ---
                        is_speaking = target_score > SENSITIVITY
                        color = (0, 255, 0) if is_speaking else (0, 0, 255)
                        label = f"ID:{track_id} Spk:{target_score:.1f}" if is_speaking else f"ID:{track_id} Sil"
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(display_frame, label, (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("LR-ASD Real-Time Prototype", display_frame)

                if MODE == "FILE":
                    if video_start_time is None:
                        video_start_time = time.time() - (frame_idx / FPS)
                    expected_time = video_start_time + (frame_idx / FPS)
                    current_time = time.time()
                    if current_time < expected_time:
                        time.sleep(expected_time - current_time)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF
                    
                if key == ord('q'):
                    break

            frame_idx += 1

    # Clean up and Evaluate
    video_client.close()
    audio_client.close()
    cv2.destroyAllWindows()
    
    # Run the evaluation if we are in FILE mode and labels exist
    if MODE == "FILE" and os.path.exists(LABELS_FOLDER):
        evaluate_live_pipeline(live_predictions, LABELS_FOLDER)

if __name__ == "__main__":
    main()