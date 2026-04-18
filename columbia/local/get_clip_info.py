import time, os, sys, torch, argparse, warnings, glob
custom_lib_path = os.path.abspath('./')
sys.path.append(custom_lib_path)
custom_lib_path = os.path.abspath('../')
sys.path.append(custom_lib_path)
import tqdm
import random
import numpy as np
import csv
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Extract Clip Info")
    parser.add_argument('--dataPathAVA',  type=str, default="./data/Columbia", help='Save path of AVA dataset')
    parser.add_argument('--out_path',     type=str, default="./predata/clip_info")
    args = parser.parse_args()

    for name in ['train', 'val']:
        cur_csvfile = os.path.join(args.dataPathAVA, 'csv', name+'_orig.csv')
        
        # --- MISSING VAL SET BYPASS ---
        if not os.path.exists(cur_csvfile):
            print(f"Skipping {name}... file not found: {cur_csvfile}")
            continue
        # ------------------------------
        
        with open(cur_csvfile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            clip_info = {}
            pre_clip_name = 'xxx'
            pre_video_name = 'uuu'
            for row in reader:
                i += 1
                if i == 0: continue
                
                video_name = row[0] # MAGIC FIX: Just read the actual video ID from the CSV!
                clip_name = row[7]  # The track-level entity ID
                
                if video_name != pre_video_name:
                    print(video_name)
                    
                if clip_name != pre_clip_name:
                    clip_info[clip_name] = [row[1]]
                else:
                    clip_info[clip_name].append(row[1])
                    
                pre_clip_name = clip_name
                pre_video_name = video_name
                
        os.makedirs(args.out_path, exist_ok=True)
        torch.save(clip_info, os.path.join(args.out_path, name+'_clip_info.pt'))