import os, sys
# --- ADD THIS TO PREVENT IMPORT ERRORS ---
custom_lib_path = os.path.abspath('./')
sys.path.append(custom_lib_path)
# -----------------------------------------

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import torch
import numpy as np
import random
import argparse
import csv
from model.models_gnn import *
from dataLoader import GraphDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Evaluate Columbia')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run the eval')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for eval')
parser.add_argument('--time_edge', type=int, default=0, help='time_edge')
parser.add_argument('--graph_path', type=str, default='./data/Columbia/graphs', help='graph_path')
parser.add_argument('--checkpoint', type=str, required=True, help='path to pretrained weights')
# Added the CSV path to map predictions back to speakers
parser.add_argument('--csv_path', type=str, default='./data/Columbia/predata/csv_split/train/col.csv', help='Path to sorted CSV')

def main():
    print("--- Booting Columbia Evaluation ---")
    args = parser.parse_args()
    
    # Lock seeds for reproducibility
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 1. Load the Columbia Graph Data
    print("Loading Graph Data...")
    Fdataset_val = GraphDataset(args, mode='train')
    val_loader = DataLoader(Fdataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # 2. Initialize the Model
    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')    
    model = MSSG()
    
    # 3. Load the Pre-trained Weights
    print(f"Loading Pre-trained Weights from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval() 

    # 4. Evaluation Loop
    target_total = []
    soft_total = []
    
    print("Evaluating Graph Nodes...")
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data.to(device)
            output = model(data)
            
            scores = F.softmax(output, dim=-1)[:,1].tolist()
            soft_total.extend(scores)
            target_total.extend(data.y[:, 0].tolist())

    # 5. Map Predictions to Speakers via CSV
    print("Mapping predictions to individual speakers...")
    with open(args.csv_path, 'r') as f:
        reader = csv.reader(f)
        data_f = list(reader)
        
    # Sort exactly like generate_graph.py did!
    data_f.sort(key=lambda x: float(x[1]))
    
    assert len(data_f) == len(soft_total), f"Mismatch! CSV has {len(data_f)} nodes, Model predicted {len(soft_total)}."

    speaker_preds = {}
    speaker_targets = {}
    
    for i, row in enumerate(data_f):
        speaker_id = row[7].split('_')[0].capitalize() # 'abbas', 'bell', etc.
        
        if speaker_id not in speaker_preds:
            speaker_preds[speaker_id] = []
            speaker_targets[speaker_id] = []
            
        speaker_preds[speaker_id].append(soft_total[i])
        speaker_targets[speaker_id].append(target_total[i])

    # 6. Calculate Metrics per Speaker
    print("\n" + "="*50)
    print(f"{'Speaker':<15} | {'mAP (%)':<15} | {'Best F1 (%)':<15}")
    print("-" * 50)
    
    f1_list = []
    map_list = []
    
    for speaker in sorted(speaker_preds.keys()):
        sp_soft = np.array(speaker_preds[speaker])
        sp_target = np.array(speaker_targets[speaker])
        
        # Calculate mAP
        try:
            sp_map = average_precision_score(sp_target, sp_soft) * 100
        except:
            sp_map = 0.0 # Safety fallback if a speaker has only 1 class
            
        map_list.append(sp_map)
        
        # Calculate Dynamic F1
        precision, recall, thresholds = precision_recall_curve(sp_target, sp_soft)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores) * 100
        f1_list.append(best_f1)
        
        print(f"{speaker:<15} | {sp_map:<15.2f} | {best_f1:<15.2f}")
        
    print("-" * 50)
    print(f"{'AVERAGE':<15} | {np.mean(map_list):<15.2f} | {np.mean(f1_list):<15.2f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()