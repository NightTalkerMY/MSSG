import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import python_speech_features
from collections import deque
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=UserWarning)

# --- Your Custom Modules ---
from proposed.data_streamer import RemoteVideoClient, RemoteAudioClient, FileVideoClient, FileAudioClient
from proposed.online_tracker import OnlineTracker
from model.faceDetector.s3fd import S3FD
from proposed.student_model import CausalStudentASD
from ASD import ASD

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "FILE"  
VIDEO_FILE = "columbia/col.mp4"
AUDIO_FILE = "columbia/col.wav"
PRETRAIN_DIR = r"D:\FYP\MSSG\predata\pretrain_model" 
STUDENT_WEIGHTS = r"D:\FYP\MSSG\proposed\realtime_epoch_mse_ce\holopi_student_epoch_30.pt"

# --- FIX 1: DYNAMIC FPS MATH ---
cap = cv2.VideoCapture(VIDEO_FILE)
ACTUAL_FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()
print(f"Detected exact video FPS: {ACTUAL_FPS:.2f}")

SENSITIVITY = 0.5    
DETECT_INTERVAL = 1    # FIX 2: Detect EVERY frame to prevent ID swapping/hallucinations
INFERENCE_INTERVAL = 4 
WINDOW_SIZE = 50       
MAX_SPEAKERS = 5       
AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE / ACTUAL_FPS)

# ==========================================
# THE MASTER TRANSLATOR (Sequence-Based)
# ==========================================
class MasterTranslator(nn.Module):
    def __init__(self, pretrain_dir):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # self.cue_configs = [
        #     {'name': 'face',            'path': 'face_model',             'size': 112},
        #     {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
        #     {'name': 'face_large',      'path': 'face_large_model',       'size': 112}, 
        #     {'name': 'face_body',       'path': 'body_model',             'size': 224}, 
        #     {'name': 'face_small',      'path': 'face_small_model',       'size': 112},
        #     {'name': 'background',      'path': 'background_model.model', 'size': 224}, 
        #     {'name': 'face_down',       'path': 'face_down_model',        'size': 112}
        # ]
        self.cue_configs = [
            {'name': 'face',            'path': 'face_model',             'size': 112},
            {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
            {'name': 'face_body',       'path': 'body_model',             'size': 224}, # Moved up
            {'name': 'face_large',      'path': 'face_large_model',       'size': 112}, # Moved down
            {'name': 'background',      'path': 'background_model.model', 'size': 224}, # Moved up
            {'name': 'face_small',      'path': 'face_small_model',       'size': 112}, # Moved down
            {'name': 'face_down',       'path': 'face_down_model',        'size': 112}
        ]
            
        self.visual_encoders = nn.ModuleDict()
        self.audio_encoder = None
        
        print("--- Booting Master Translator (Loading 7 CNNs to VRAM) ---")
        for config in self.cue_configs:
            name = config['name']
            full_path = os.path.join(pretrain_dir, config['path'])
            s = ASD(encoder_struct=[32,64,128])
            if os.path.isdir(full_path):
                s.loadParameters_multi(full_path)
            else:
                s.loadParameters(full_path)
                
            s.eval()
            self.visual_encoders[name] = s.model.visualEncoder.to(self.device)
            if name == 'face':
                self.audio_encoder = s.model.audioEncoder.to(self.device)

    def forward_visual(self, crops_dict):
        visual_vectors = []
        with torch.no_grad():
            for config in self.cue_configs:
                name = config['name']
                crop_seq = crops_dict[name] # [50, size, size]
                
                crop_tensor = torch.FloatTensor(crop_seq).unsqueeze(0).unsqueeze(0).to(self.device)
                crop_tensor = (crop_tensor / 255.0 - 0.4161) / 0.1688
                
                v_feat = self.visual_encoders[name](crop_tensor) # [1, 50, 128]
                visual_vectors.append(v_feat.squeeze(0).cpu().numpy()) 
                
        return np.concatenate(visual_vectors, axis=-1) # [50, 896]

    def forward_audio(self, continuous_audio):
        with torch.no_grad():
            a_tensor = torch.FloatTensor(continuous_audio).unsqueeze(0).to(self.device)
            a_tensor = a_tensor.unsqueeze(1).transpose(2, 3) 
            a_feat = self.audio_encoder(a_tensor) 
        return a_feat.squeeze(0).cpu().numpy() # [50, 128]

def get_crop(img_gray, x1, y1, x2, y2, size):
    h, w = img_gray.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = img_gray[y1:y2, x1:x2]
    if crop.size == 0: return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(crop, (size, size))

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    if MODE == "REMOTE":
        video_client = RemoteVideoClient()
        audio_client = RemoteAudioClient()
    else:
        video_client = FileVideoClient(VIDEO_FILE)
        audio_client = FileAudioClient(AUDIO_FILE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    face_detector = S3FD(device='cuda')
    tracker = OnlineTracker(iou_threshold=0.4, max_lost=2, alpha=0.8)
    master_translator = MasterTranslator(PRETRAIN_DIR).to(device).eval()
    
    asd_model = CausalStudentASD().to(device)
    asd_model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device))
    asd_model.eval()

    track_history = {}      
    bbox_history = {}       
    display_buffer = deque(maxlen=WINDOW_SIZE) 
    
    # NEW: Global RAW audio buffer to prevent the stuttering bug
    global_audio_buffer = deque(maxlen=WINDOW_SIZE)
    
    # State cache to hold predictions between PyTorch firings
    ai_state_cache = {} 
    
    frame_idx = 0
    print("\nStarting Real-Time Pipeline... Press 'q' to quit.")
    video_start_time = None 

    with torch.no_grad():
        while True:
            frame = video_client.get_frame()
            if frame is None: break
            display_buffer.append(frame.copy())
            
            # 1. DETECT & TRACK
            if frame_idx % DETECT_INTERVAL == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
                last_detections = [det[:-1].tolist() for det in raw_detections]
            
            active_tracks = tracker.update(last_detections)
            stale_ids = [tid for tid in track_history.keys() if tid not in active_tracks]
            for tid in stale_ids:
                del track_history[tid]
                del bbox_history[tid]
                if tid in ai_state_cache: del ai_state_cache[tid]

            # 2. RAW AUDIO PREP
            if MODE == "REMOTE":
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME)
            else:
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME, frame_idx, fps=ACTUAL_FPS)
                
            if raw_audio is None or len(raw_audio) == 0:
                raw_audio = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            
            # --- STORE RAW WAVE, NOT MFCC ---
            # We must buffer the continuous wave to prevent audio stuttering
            global_audio_buffer.append(raw_audio)

            # 3. VISUAL CROP BUFFERING
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            for track_id, bbox in active_tracks.items():
                x1, y1, x2, y2 = bbox
                
                # --- S3FD to AVA Box Expansion ---
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                size = max(x2 - x1, y2 - y1) * 2.2 / 2.0  
                
                col_x1, col_y1 = cx - size, cy - size
                col_x2, col_y2 = cx + size, cy + size
                bw, bh = col_x2 - col_x1, col_y2 - col_y1
                
                crops = {
                    'face': get_crop(frame_gray, col_x1, col_y1, col_x2, col_y2, 112),
                    'face_body_large': get_crop(frame_gray, col_x1 - 0.5*bw, col_y1 - 0.1*bh, col_x2 + 0.5*bw, col_y2 + 1.0*bh, 224),
                    'face_large': get_crop(frame_gray, col_x1 - 0.05*bw, col_y1 - 0.05*bh, col_x2 + 0.05*bw, col_y1 + 0.75*bh, 112),
                    'face_body': get_crop(frame_gray, col_x1 - 0.75*bw, col_y1, col_x2 + 0.75*bw, col_y2 + 3.0*bh, 224),
                    'face_small': get_crop(frame_gray, col_x1 + 0.25*bw, col_y1 + 0.15*bh, col_x2 - 0.25*bw, col_y1 + 0.55*bh, 112),
                    'face_down': get_crop(frame_gray, col_x1, col_y1 + 0.35*bh, col_x2, col_y1 + 0.60*bh, 112)
                }
                
                bg_frame = frame_gray.copy()
                cv2.rectangle(bg_frame, (int(col_x1), int(col_y1)), (int(col_x2), int(col_y2)), (0, 0, 0), thickness=cv2.FILLED)
                crops['background'] = get_crop(bg_frame, 0, 0, w, h, 224)

                # --- THE SPATIAL FIX: Use Center X and Center Y ---
                norm_cx = (col_x1 + col_x2) / 2.0 / w
                norm_cy = (col_y1 + col_y2) / 2.0 / h
                norm_bw, norm_bh = bw / w, bh / h
                spatial_4d = np.array([norm_cx, norm_cy, norm_bw, norm_bh], dtype=np.float32)

                if track_id not in track_history:
                    track_history[track_id] = {'raw_v': {k: deque(maxlen=WINDOW_SIZE) for k in crops.keys()}, 's': deque(maxlen=WINDOW_SIZE)}
                    bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)
                
                for k in crops.keys():
                    track_history[track_id]['raw_v'][k].append(crops[k])
                    
                track_history[track_id]['s'].append(spatial_4d)
                bbox_history[track_id].append([x1, y1, x2, y2])

            # 4. HEAVY INFERENCE (ONLY EVERY 4 FRAMES)
            if len(display_buffer) == WINDOW_SIZE and frame_idx % INFERENCE_INTERVAL == 0:
                
                # --- THE TRUE AUDIO FIX ---
                # Combine the raw audio into a perfect 2-second continuous wave first
                continuous_raw_audio = np.concatenate(list(global_audio_buffer), axis=0)
                scaled_audio = continuous_raw_audio * 32767.0 if np.max(np.abs(continuous_raw_audio)) <= 2.0 else continuous_raw_audio
                
                # Run MFCC on the 2-second wave. This creates ~198 perfectly smooth, stepping windows!
                audio_mfcc = python_speech_features.mfcc(scaled_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010)
                
                if audio_mfcc.shape[0] > 200: audio_mfcc = audio_mfcc[:200, :]
                elif audio_mfcc.shape[0] < 200: audio_mfcc = np.pad(audio_mfcc, ((0, 200 - audio_mfcc.shape[0]), (0, 0)), mode='edge')
                
                seq_a_128 = master_translator.forward_audio(audio_mfcc)
                
                v_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 896), dtype=np.float32)
                a_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 128), dtype=np.float32)
                s_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 4), dtype=np.float32)
                m_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE), dtype=np.float32)
                
                speaker_mapping = {} 
                slot_idx = 0
                
                for track_id, history in track_history.items():
                    if len(history['s']) == WINDOW_SIZE and slot_idx < MAX_SPEAKERS:
                        
                        stacked_crops = {k: np.stack(history['raw_v'][k]) for k in history['raw_v'].keys()}
                        seq_v_896 = master_translator.forward_visual(stacked_crops)
                        
                        v_tensor[0, slot_idx] = seq_v_896 
                        a_tensor[0, slot_idx] = seq_a_128 
                        s_tensor[0, slot_idx] = np.array(history['s'])
                        m_tensor[0, slot_idx] = 1.0 
                        
                        speaker_mapping[slot_idx] = track_id
                        slot_idx += 1
                        
                if slot_idx > 0: 
                    v_t = torch.FloatTensor(v_tensor).to(device)
                    a_t = torch.FloatTensor(a_tensor).to(device)
                    s_t = torch.FloatTensor(s_tensor).to(device)
                    m_t = torch.FloatTensor(m_tensor).to(device)
                    
                    logits, _ = asd_model(v_t, a_t, s_t, m_t)
                    frame_probs = torch.nn.functional.softmax(logits[0, :, -1, :], dim=-1) 
                    
                    # Update State Cache
                    for slot, track_id in speaker_mapping.items():
                        prob_speaking = frame_probs[slot, 1].item()
                        ai_state_cache[track_id] = prob_speaking
                        # Debug Print - Watch this spike!
                        # print(f"Frame {frame_idx} | Slot {slot} (ID {track_id}): {prob_speaking:.4f}")

            # 5. SMOOTH ASYNCHRONOUS DRAWING (EVERY FRAME)
            if len(display_buffer) == WINDOW_SIZE:
                display_frame = display_buffer[-1].copy()
                
                for track_id, bbox_list in bbox_history.items():
                    if len(bbox_list) > 0:
                        prob_speaking = ai_state_cache.get(track_id, 0.0)
                        is_speaking = prob_speaking > SENSITIVITY
                        
                        tx1, ty1, tx2, ty2 = map(int, bbox_list[-1])
                        
                        color = (0, 255, 0) if is_speaking else (0, 0, 255)
                        label = f"ID:{track_id} Spk:{prob_speaking:.2f}" if is_speaking else f"ID:{track_id} Sil"
                        
                        cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), color, 3)
                        cv2.putText(display_frame, label, (tx1, max(30, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("HoloPi Real-Time Pipeline", display_frame)

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
    audio_client.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()