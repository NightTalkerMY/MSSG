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

# Hide legacy PyTorch warnings from S3FD
warnings.filterwarnings("ignore", category=UserWarning)

# --- Your Custom Modules ---
from proposed.data_streamer import RemoteVideoClient, RemoteAudioClient, FileVideoClient, FileAudioClient
from proposed.online_tracker import OnlineTracker
from model.faceDetector.s3fd import S3FD

# --- The New Phase 2 Student Model ---
from proposed.student_model import CausalStudentASD

# --- The Original MSSG Backend ---
from ASD import ASD

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "FILE"  
VIDEO_FILE = "gay/supergay.mp4"
AUDIO_FILE = "gay/supergay.wav"
PRETRAIN_DIR = r"D:\FYP\MSSG\predata\pretrain_model" 
STUDENT_WEIGHTS = r"D:\FYP\MSSG\proposed\realtime_epoch_mse_ce\holopi_student_epoch_30.pt"

SENSITIVITY = 0.5    
DETECT_INTERVAL = 3  
WINDOW_SIZE = 50       
MAX_SPEAKERS = 5       
FPS = 25
AUDIO_SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE / FPS)

# ==========================================
# THE MASTER TRANSLATOR (Hybrid Optimized)
# ==========================================
class MasterTranslator(nn.Module):
    def __init__(self, pretrain_dir):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cue_configs = [
            {'name': 'face',            'path': 'face_model',             'size': 112},
            {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
            {'name': 'face_large',      'path': 'face_large_model',       'size': 112}, 
            {'name': 'face_body',       'path': 'body_model',             'size': 224}, 
            {'name': 'face_small',      'path': 'face_small_model',       'size': 112},
            {'name': 'background',      'path': 'background_model.model', 'size': 224}, 
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
        """Passes ONE frame through the 7 Visual CNNs."""
        visual_vectors = []
        with torch.no_grad():
            for config in self.cue_configs:
                name = config['name']
                crop = crops_dict[name] 
                
                crop_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
                crop_tensor = (crop_tensor / 255.0 - 0.4161) / 0.1688
                
                v_feat = self.visual_encoders[name](crop_tensor) # [1, 1, 128]
                visual_vectors.append(v_feat.view(128).cpu().numpy()) 
                
        return np.concatenate(visual_vectors, axis=0) # [896]

    def forward_audio(self, audio_sequence):
        """Passes 50 FRAMES of audio through the Audio CNN sequence."""
        with torch.no_grad():
            # audio_sequence is [200, 13]
            a_tensor = torch.FloatTensor(audio_sequence).unsqueeze(0).to(self.device)
            a_tensor = a_tensor.unsqueeze(1).transpose(2, 3) # [1, 1, 13, 200]
            a_feat = self.audio_encoder(a_tensor) # [1, 50, 128]
            
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
    
    # NEW: Global audio buffer! Audio is the same for everyone in the room.
    global_audio_buffer = deque(maxlen=WINDOW_SIZE)
    
    frame_idx = 0
    print("\nStarting Real-Time Pipeline... Press 'q' to quit.")
    video_start_time = None 

    with torch.no_grad():
        while True:
            frame = video_client.get_frame()
            if frame is None: break
            display_buffer.append(frame.copy())
            
            # 1. DETECT
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_detections = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
            last_detections = [det[:-1].tolist() for det in raw_detections]
            
            active_tracks = tracker.update(last_detections)
            stale_ids = [tid for tid in track_history.keys() if tid not in active_tracks]
            for tid in stale_ids:
                del track_history[tid]
                del bbox_history[tid]

            # 2. AUDIO PREP (GLOBAL)
            if MODE == "REMOTE":
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME)
            else:
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME, frame_idx)
                
            if raw_audio is not None and len(raw_audio) > 0:
                scaled_audio = raw_audio * 32767.0 if np.max(np.abs(raw_audio)) <= 2.0 else raw_audio
                audio_mfcc = python_speech_features.mfcc(scaled_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010)
                if audio_mfcc.shape[0] > 4: audio_mfcc = audio_mfcc[:4, :]
                elif audio_mfcc.shape[0] < 4: audio_mfcc = np.pad(audio_mfcc, ((0, 4 - audio_mfcc.shape[0]), (0, 0)), mode='edge')
            else:
                audio_mfcc = np.zeros((4, 13), dtype=np.float32)
                
            global_audio_buffer.append(audio_mfcc)

            # 3. INCREMENTAL VISUAL TRANSLATION (Fast)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            for track_id, bbox in active_tracks.items():
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                size = max(x2 - x1, y2 - y1) * 1.3 / 2.0  
                
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

                # Fetch ONLY the visual vector incrementally
                v_896 = master_translator.forward_visual(crops)

                norm_x1 = max(0, col_x1) / w
                norm_y1 = max(0, col_y1) / h
                norm_bw, norm_bh = bw / w, bh / h
                spatial_4d = np.array([norm_x1, norm_y1, norm_bw, norm_bh], dtype=np.float32)

                if track_id not in track_history:
                    track_history[track_id] = {'v': deque(maxlen=WINDOW_SIZE), 's': deque(maxlen=WINDOW_SIZE)}
                    bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)
                
                track_history[track_id]['v'].append(v_896)
                track_history[track_id]['s'].append(spatial_4d)
                bbox_history[track_id].append([x1, y1, x2, y2])

            # 4. LIGHTWEIGHT CAUSAL INFERENCE
            if len(display_buffer) == WINDOW_SIZE:
                display_frame = display_buffer[-1] 
                
                # Extract Audio Sequence ONCE globally!
                stacked_audio = np.concatenate(list(global_audio_buffer), axis=0) # [200, 13]
                seq_a_128 = master_translator.forward_audio(stacked_audio) # [50, 128]
                
                v_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 896), dtype=np.float32)
                a_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 128), dtype=np.float32)
                s_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 4), dtype=np.float32)
                m_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE), dtype=np.float32)
                
                speaker_mapping = {} 
                slot_idx = 0
                
                for track_id, history in track_history.items():
                    if len(history['v']) == WINDOW_SIZE and slot_idx < MAX_SPEAKERS:
                        
                        v_tensor[0, slot_idx] = np.array(history['v'])
                        a_tensor[0, slot_idx] = seq_a_128 # The exact same audio for all speakers
                        s_tensor[0, slot_idx] = np.array(history['s'])
                        m_tensor[0, slot_idx] = 1.0 
                        
                        speaker_mapping[slot_idx] = track_id
                        slot_idx += 1
                        
                v_t = torch.FloatTensor(v_tensor).to(device)
                a_t = torch.FloatTensor(a_tensor).to(device)
                s_t = torch.FloatTensor(s_tensor).to(device)
                m_t = torch.FloatTensor(m_tensor).to(device)
                
                if slot_idx > 0: 
                    logits, _ = asd_model(v_t, a_t, s_t, m_t)
                    frame_probs = torch.nn.functional.softmax(logits[0, :, -1, :], dim=-1) 
                    
                    for slot, track_id in speaker_mapping.items():
                        prob_speaking = frame_probs[slot, 1].item()
                        is_speaking = prob_speaking > SENSITIVITY
                        
                        target_bbox = bbox_history[track_id][-1] 
                        tx1, ty1, tx2, ty2 = map(int, target_bbox)
                        
                        color = (0, 255, 0) if is_speaking else (0, 0, 255)
                        label = f"ID:{track_id} Spk:{prob_speaking:.2f}" if is_speaking else f"ID:{track_id} Sil"
                        
                        cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), color, 3)
                        cv2.putText(display_frame, label, (tx1, max(30, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("HoloPi Real-Time Pipeline", display_frame)

                if MODE == "FILE":
                    if video_start_time is None:
                        video_start_time = time.time() - (frame_idx / FPS)
                    expected_time = video_start_time + (frame_idx / FPS)
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