import pandas as pd
import os
import shutil

def generate_loader_csv():
    csv_path = './data/Columbia/csv/train_orig.csv'
    out_path = './data/Columbia/csv/train_loader.csv'
    val_path = './data/Columbia/csv/val_loader.csv'
    
    print("Reading frame-level CSV...")
    df = pd.read_csv(csv_path)
    
    print("Generating track-level summary for dataLoader...")
    with open(out_path, 'w') as f:
        # Group all frames by speaker
        for entity_id, group in df.groupby('entity_id'):
            group = group.sort_values('frame_timestamp')
            num_frames = len(group)
            fps = 25  # Columbia video is 25 fps
            
            # Extract their labels (0 or 1) into a single flat list string
            labels = list(group['label_id'].astype(int))
            labels_str = str(labels).replace(' ', '') # Format: [0,0,1,1,...]
            
            # Write the exact tab-separated format the dataLoader demands
            line = f"{entity_id}\t{num_frames}\t{fps}\t{labels_str}\n"
            f.write(line)
            print(f" -> Processed {entity_id} ({num_frames} frames)")
            
    # Duplicate for the val loader
    shutil.copy(out_path, val_path)
    print(f"\nSuccessfully generated {out_path} and {val_path}!")

if __name__ == '__main__':
    generate_loader_csv()