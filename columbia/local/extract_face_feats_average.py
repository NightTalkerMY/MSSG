import time, os, sys, torch, argparse, warnings, glob
# --- FORCE GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -----------------
custom_lib_path = os.path.abspath('./')
sys.path.append(custom_lib_path)
custom_lib_path = os.path.abspath('../')
sys.path.append(custom_lib_path)
from dataLoader import feat_extract_loader
import tqdm
# from utils.tools import *
from ASD import ASD
import random
import numpy as np
import csv
import torch 
# Add this right here! It forces Windows to use your 4080 Super.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import average_precision_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次卷积运算的确定性
    torch.backends.cudnn.benchmark = False     # 禁用cuDNN的自动优化
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Model Training")
    parser.add_argument('--background_H',    type=int,   default='112',  help='background_H')
    parser.add_argument('--dataPathAVA',  type=str, default="/share/home/liguanjun/data/AVA", help='Save path of AVA dataset')
    parser.add_argument('--out_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/feats_info")
    parser.add_argument('--face_model_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/pretrain_model/face_test")
    parser.add_argument('--clip_info_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/clip_info")
    args = parser.parse_args()
    
    # set_seed(111)
    print(f'Using {torch.cuda.device_count()} GPUs')

    train_loader = feat_extract_loader(trialFileName = os.path.join(args.dataPathAVA, 'csv', 'train_loader.csv'), \
			                  audioPath     = os.path.join(args.dataPathAVA, 'clips_audios', 'train'), \
			                  visualPath    = os.path.join(args.dataPathAVA, 'clips_videos_faces', 'train'), \
			                  H = 112,
                              type='train',
			                  )
    val_loader = feat_extract_loader(trialFileName = os.path.join(args.dataPathAVA, 'csv', 'val_loader.csv'), \
		                  	audioPath     = os.path.join(args.dataPathAVA, 'clips_audios', 'val'), \
			                visualPath    = os.path.join(args.dataPathAVA, 'clips_videos', 'val'), \
		                    H = 112,
                            type='val',
		                    )
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = False)
    valLoader = torch.utils.data.DataLoader(val_loader, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = False)
    
    s = ASD(encoder_struct=[32,64,128])
    s.loadParameters_multi(args.face_model_path)
    s.eval()
    
    for name in ['train']:
    # for name in ['val']:
    # for name in ['train']:
        if name == 'train':
            dataloader = trainLoader
        else:
            dataloader = valLoader
        clip_info = torch.load(os.path.join(args.clip_info_path, name+'_clip_info.pt'))
        feat_save_path = os.path.join(args.out_path, name)
        os.makedirs(feat_save_path, exist_ok=True)
        csv_fid = None  # Safely initialize this first!
        pre_video_name = 'uuu'
        predScores = []
        target_total = []
        soft_total = []
        for item in tqdm.tqdm(dataloader):
            clip_name = item[0][0]
            if clip_name in ['abbas', 'bell', 'boll', 'lieb', 'long', 'sick']:
                video_name = 'col'
            else:
                video_name = clip_name[:11]
                
            clip_timestamps = clip_info[clip_name]
            audio_feats = item[1][0]
            visual_feats = item[2][0]
            labels = item[3][0]

            # --- SAFETY BYPASS ---
            if len(visual_feats.shape) < 3 or visual_feats.shape[0] == 0:
                print(f"\n[WARNING] 0 images found for {clip_name}! PyTorch is skipping it.")
                continue
            # ---------------------

            # with torch.no_grad(): 
            #     outsAV, outsV, outsA = s.model(audio_feats.cuda(), visual_feats.cuda())
            #     labels = labels.reshape((-1)).cuda() 
            #     _, predScore, _, _ = s.lossAV(outsAV, labels)  
            
            # scores = predScore[:, 1].tolist()
            # soft_total.extend(scores)
            # target_total.extend(labels[:].tolist())

            # --- VRAM OOM PREVENTION (CHUNKING) ---
            chunk_size = 1500  # Safe size for standard GPUs
            total_frames = visual_feats.shape[1]  # <--- FIX: Dimension 1 is Time!
            
            outsAV_list, outsV_list, outsA_list, predScore_list = [], [], [], []
            
            # Reshape labels to a flat 1D list [T] first to make chunking easy
            labels = labels.reshape((-1)).cuda() 
            
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                
                # Slice Dimension 1 (Time) while preserving Dimension 0 (Batch)
                a_chunk = audio_feats[:, start*4 : end*4, :].cuda().contiguous()
                v_chunk = visual_feats[:, start : end, :, :].cuda().contiguous()
                l_chunk = labels[start : end].contiguous()
                
                with torch.no_grad():
                    o_AV, o_V, o_A = s.model(a_chunk, v_chunk)
                    _, p_Score, _, _ = s.lossAV(o_AV, l_chunk)
                    
                    outsAV_list.append(o_AV.cpu())
                    outsV_list.append(o_V.cpu())
                    outsA_list.append(o_A.cpu())
                    predScore_list.append(p_Score.cpu())
            
            # Recombine the chunks back into full track tensors
            outsAV = torch.cat(outsAV_list, dim=0)
            outsV = torch.cat(outsV_list, dim=0)
            outsA = torch.cat(outsA_list, dim=0)
            predScore = torch.cat(predScore_list, dim=0)
            # --------------------------------------
            
            scores = predScore[:, 1].tolist()
            soft_total.extend(scores)
            target_total.extend(labels[:].tolist())
            
            if video_name != pre_video_name:
                if csv_fid is not None:
                    csv_fid.close()
                print(f"\nProcessing video: {video_name}")
                csv_fid = open(os.path.join(args.out_path, name, video_name+'.csv'), mode='w', newline='')
                writer = csv.writer(csv_fid, delimiter=',')
                
            pre_video_name = video_name
            ii = -1
            feat_av = outsAV.cpu().numpy()
            feat_v = outsV.cpu().numpy()
            feat_a = outsA.cpu().numpy()
            
            assert feat_av.shape[0] == len(clip_timestamps)
            for timestamp in clip_timestamps:
                ii += 1
                title = clip_name + ":" + timestamp
                cur_feat_av = list(feat_av[ii,:])
                cur_feat_v = list(feat_v[ii,:])
                cur_feat_a = list(feat_a[ii,:])
                writer.writerow([title,cur_feat_av,cur_feat_v,cur_feat_a])
            
        if csv_fid is not None:
            csv_fid.close()    
            
        mAP = average_precision_score(target_total, soft_total) * 100
        print(f"mAP: {mAP}")