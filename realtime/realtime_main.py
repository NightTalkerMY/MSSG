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
VIDEO_FILE = "columbia/col.mp4"
AUDIO_FILE = "columbia/col.wav"
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
# THE MASTER TRANSLATOR (The Real 896-D Bridge)
# ==========================================
class MasterTranslator(nn.Module):
    """Replicates the entire offline 7-Crop PyTorch extraction pipeline in real-time."""
    def __init__(self, pretrain_dir):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Order is critical to match the cache builder concatenation!
        # self.cue_configs = [
        #     {'name': 'face',            'path': 'face_model',             'size': 112},
        #     {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
        #     {'name': 'face_body',       'path': 'body_model',             'size': 224},
        #     {'name': 'face_large',      'path': 'face_large_model',       'size': 112},
        #     {'name': 'background',      'path': 'background_model.model', 'size': 224}, # <--- FIXED
        #     # {'name': 'background',      'path': 'background224_model_0027_79.81.model', 'size': 224}, # Assuming single file
        #     {'name': 'face_small',      'path': 'face_small_model',       'size': 112},
        #     {'name': 'face_down',       'path': 'face_down_model',        'size': 112}
        # ]
        # Order is critically matched to the offline Columbia cache!
        self.cue_configs = [
            {'name': 'face',            'path': 'face_model',             'size': 112},
            {'name': 'face_body_large', 'path': 'body_large_model',       'size': 224},
            {'name': 'face_large',      'path': 'face_large_model',       'size': 112}, # FIXED ORDER
            {'name': 'face_body',       'path': 'body_model',             'size': 224}, # FIXED ORDER
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
            print(f"Loading {name} weights...")
            
            s = ASD(encoder_struct=[32,64,128])
            if os.path.isdir(full_path):
                s.loadParameters_multi(full_path)
            else:
                # Fallback for single .model files like background
                s.loadParameters(full_path)
                
            s.eval()
            # We ONLY keep the visual frontend to save VRAM
            self.visual_encoders[name] = s.model.visualEncoder.to(self.device)
            
            # Steal the audio encoder from the main face model
            if name == 'face':
                self.audio_encoder = s.model.audioEncoder.to(self.device)

    def _safe_crop(self, img_gray, x1, y1, x2, y2, size):
        """Extracts crop, resizes, and shapes it for the MSSG Frontend."""
        h, w = img_gray.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        crop = img_gray[y1:y2, x1:x2]
        
        if crop.size == 0:
            # Fallback tensor if person walks off screen [B=1, T=1, W=size, H=size]
            return torch.zeros((1, 1, size, size)).to(self.device)
            
        crop = cv2.resize(crop, (size, size))
        
        # The visualEncoder expects [B, T, W, H]
        crop_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0).to(self.device)
        return crop_tensor

    def forward(self, frame_bgr, bbox, audio_mfcc):
        """Passes 1 frame and 1 audio chunk through the 7 CNNs."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        h, w = frame_bgr.shape[:2]
        
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. GENERATE THE 7 CROPS MATHEMATICALLY
        crops = {}
        
        # Face
        crops['face'] = self._safe_crop(frame_gray, x1, y1, x2, y2, 112)
        
        # Face Body Large
        crops['face_body_large'] = self._safe_crop(frame_gray, x1 - 0.5*bw, y1 - 0.1*bh, x2 + 0.5*bw, y2 + 1.0*bh, 224)
        
        # Face Body
        crops['face_body'] = self._safe_crop(frame_gray, x1 - 0.75*bw, y1, x2 + 0.75*bw, y2 + 3.0*bh, 224)
        
        # Face Large
        crops['face_large'] = self._safe_crop(frame_gray, x1 - 0.05*bw, y1 - 0.05*bh, x2 + 0.05*bw, y1 + 0.75*bh, 112)
        
        # Background (Full frame, mask face)
        bg_frame = frame_gray.copy()
        cv2.rectangle(bg_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=cv2.FILLED)
        crops['background'] = self._safe_crop(bg_frame, 0, 0, w, h, 224)
        
        # Face Small
        crops['face_small'] = self._safe_crop(frame_gray, x1 + 0.25*bw, y1 + 0.15*bh, x2 - 0.25*bw, y1 + 0.55*bh, 112)
        
        # Face Down
        crops['face_down'] = self._safe_crop(frame_gray, x1, y1 + 0.35*bh, x2, y1 + 0.60*bh, 112)

        # 2. RUN CNN EXTRACTION
        visual_vectors = []
        with torch.no_grad():
            for config in self.cue_configs:
                name = config['name']
                crop_tensor = crops[name]
                
                # Apply normalization math from ASD_Model
                crop_tensor = crop_tensor.view(1, 1, 1, config['size'], config['size'])
                crop_tensor = (crop_tensor / 255.0 - 0.4161) / 0.1688
                
                v_feat = self.visual_encoders[name](crop_tensor) # outputs [1, 1, 128]
                visual_vectors.append(v_feat.view(-1).cpu().numpy())
                
            # 3. EXTRACT AUDIO
            # Audio is [1, 4, 13] for one frame's worth of MFCCs
            a_tensor = torch.FloatTensor(audio_mfcc).unsqueeze(0).to(self.device)
            # Apply unsqueeze/transpose logic from ASD_Model
            a_tensor = a_tensor.unsqueeze(1).transpose(2, 3) 
            a_feat = self.audio_encoder(a_tensor) # outputs [1, 1, 128]
            
        final_v_896 = np.concatenate(visual_vectors, axis=0)
        final_a_128 = a_feat.view(-1).cpu().numpy()
        
        return final_v_896, final_a_128

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
    
    # Init Backend
    face_detector = S3FD(device='cuda')
    tracker = OnlineTracker(iou_threshold=0.4, max_lost=5, alpha=0.8)
    master_translator = MasterTranslator(PRETRAIN_DIR).to(device).eval()
    
    # Init Student
    asd_model = CausalStudentASD().to(device)
    asd_model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device))
    asd_model.eval()

    track_history = {}      
    bbox_history = {}       
    display_buffer = deque(maxlen=WINDOW_SIZE) 
    
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

            # 2. AUDIO PREP
            if MODE == "REMOTE":
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME)
            else:
                raw_audio = audio_client.get_latest_audio(SAMPLES_PER_FRAME, frame_idx)
                
            if raw_audio is not None and len(raw_audio) > 0:
                scaled_audio = raw_audio * 32767.0 if np.max(np.abs(raw_audio)) <= 2.0 else raw_audio
                # We expect exactly 4 windows for a 25fps stream
                audio_mfcc = python_speech_features.mfcc(scaled_audio, AUDIO_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010)
                if audio_mfcc.shape[0] > 4: audio_mfcc = audio_mfcc[:4, :]
                elif audio_mfcc.shape[0] < 4: audio_mfcc = np.pad(audio_mfcc, ((0, 4 - audio_mfcc.shape[0]), (0, 0)), mode='edge')
            else:
                audio_mfcc = np.zeros((4, 13), dtype=np.float32)

            # 3. MASTER TRANSLATOR (Raw -> 896-D / 128-D)
            for track_id, bbox in active_tracks.items():
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                
                # Get the authentic features
                visual_896d, audio_128d = master_translator(frame, bbox, audio_mfcc)
                
                # Normalize spatial cue
                cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                spatial_4d = np.array([cx, cy, bw, bh], dtype=np.float32)
                
                if track_id not in track_history:
                    track_history[track_id] = {'v': deque(maxlen=WINDOW_SIZE), 'a': deque(maxlen=WINDOW_SIZE), 's': deque(maxlen=WINDOW_SIZE)}
                    bbox_history[track_id] = deque(maxlen=WINDOW_SIZE)
                
                track_history[track_id]['v'].append(visual_896d)
                track_history[track_id]['a'].append(audio_128d)
                track_history[track_id]['s'].append(spatial_4d)
                bbox_history[track_id].append([x1, y1, x2, y2])

            # 4. CAUSAL STUDENT INFERENCE
            if len(display_buffer) == WINDOW_SIZE:
                # Get the NEWEST frame (Index -1) instead of the oldest!
                display_frame = display_buffer[-1] 
                
                v_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 896), dtype=np.float32)
                a_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 128), dtype=np.float32)
                s_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE, 4), dtype=np.float32)
                m_tensor = np.zeros((1, MAX_SPEAKERS, WINDOW_SIZE), dtype=np.float32)
                
                speaker_mapping = {} 
                slot_idx = 0
                
                for track_id, history in track_history.items():
                    if len(history['v']) == WINDOW_SIZE and slot_idx < MAX_SPEAKERS:
                        v_tensor[0, slot_idx] = np.array(history['v'])
                        a_tensor[0, slot_idx] = np.array(history['a'])
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
                    
                    # Read the prediction at the CURRENT frame (Index -1), not frame 0!
                    frame_probs = torch.nn.functional.softmax(logits[0, :, -1, :], dim=-1) 
                    
                    # 5. DRAW BBOX ON SCREEN
                    for slot, track_id in speaker_mapping.items():
                        prob_speaking = frame_probs[slot, 1].item()
                        is_speaking = prob_speaking > SENSITIVITY
                        
                        # Draw the bounding box for the NEWEST frame
                        target_bbox = bbox_history[track_id][-1] 
                        x1, y1, x2, y2 = map(int, target_bbox)
                        
                        color = (0, 255, 0) if is_speaking else (0, 0, 255)
                        label = f"ID:{track_id} Spk:{prob_speaking:.2f}" if is_speaking else f"ID:{track_id} Sil"
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(display_frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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