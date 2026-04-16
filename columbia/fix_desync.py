import pandas as pd
import shutil

def fix_csv_sync():
    csv_path = './data/Columbia/csv/train_orig.csv'

    print("1. Deduplicating train_orig.csv (Converting 29.97fps annotations to 25fps)...")
    df = pd.read_csv(csv_path)
    
    # Drop duplicate timestamps for the same speaker
    df_dedup = df.drop_duplicates(subset=['entity_id', 'frame_timestamp'], keep='first')
    
    # Save it back
    df_dedup.to_csv(csv_path, index=False)
    
    # Mirror to val_orig.csv just in case
    val_path = './data/Columbia/csv/val_orig.csv'
    df_dedup.to_csv(val_path, index=False)
    print(f" -> Reduced rows from {len(df)} to {len(df_dedup)}!")

    print("2. Regenerating train_loader.csv...")
    out_path = './data/Columbia/csv/train_loader.csv'
    val_loader_path = './data/Columbia/csv/val_loader.csv'
    
    with open(out_path, 'w') as f:
        for entity_id, group in df_dedup.groupby('entity_id'):
            group = group.sort_values('frame_timestamp')
            num_frames = len(group)
            labels = list(group['label_id'].astype(int))
            labels_str = str(labels).replace(' ', '') 
            
            line = f"{entity_id}\t{num_frames}\t25\t{labels_str}\n"
            f.write(line)
            
    shutil.copy(out_path, val_loader_path)
    print("SUCCESS: CSVs are now perfectly synced with your image folders!")

if __name__ == '__main__':
    fix_csv_sync()