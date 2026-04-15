import os
import torch
import numpy as np
import argparse
from model.models_gnn import *
from dataLoader import GraphDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Columbia Graph Evaluator')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run the eval')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--graph_path', type=str, default='./data/Columbia/graphs', help='Path to your generated Columbia graphs')
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the fully trained MSSG checkpoint (.pt file)')

def main():
    args = parser.parse_args()
    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')    
    
    print(f"Loading Columbia Graph from {args.graph_path}...")
    # We load it as 'train' mode because that's the folder we forced it into during graph generation
    dataset = GraphDataset(args, mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Loading Pre-Trained Model from {args.model_checkpoint}...")
    model = MSSG()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Dictionaries to track predictions per speaker
    speaker_predictions = {}
    speaker_ground_truths = {}

    print("Running Inference...")
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            output = model(data)
            
            # Get binary predictions (1 if score > 0.5, else 0)
            scores = F.softmax(output, dim=-1)[:, 1]
            predictions = (scores > 0.5).int().cpu().numpy()
            ground_truths = data.y[:, 0].int().cpu().numpy()
            
            # The node identities (speaker names) are sadly lost inside the PyG Data object during generic loading.
            # To strictly calculate F1 per speaker, we need to adapt this section based on how identity is tracked in GraphDataset.
            # For a quick global baseline of the graph's performance on Columbia:
            
            if 'global' not in speaker_predictions:
                speaker_predictions['global'] = []
                speaker_ground_truths['global'] = []
            
            speaker_predictions['global'].extend(predictions)
            speaker_ground_truths['global'].extend(ground_truths)

    print("\n--- Columbia F1 Results ---")
    y_pred = speaker_predictions['global']
    y_true = speaker_ground_truths['global']
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Global F1-Score: {f1:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")

if __name__ == '__main__':
    main()