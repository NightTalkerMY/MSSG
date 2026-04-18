import os, subprocess, glob, pandas, tqdm, cv2, numpy
import argparse

def safe_name(name):
    return str(name).replace(":", "_")

def run(args):
    for dataType in ['train', 'val']:
        csv_path = os.path.join(args.AVADataPath, 'csv', f'{dataType}_orig.csv')
        
        # --- PREVENT CRASH IF VAL DOESN'T EXIST ---
        if not os.path.exists(csv_path):
            print(f"Skipping {dataType}... File not found: {csv_path}")
            continue

        print(f'Deal with {dataType}...')
        outDir = os.path.join(args.down_out_path, dataType)
        
        # Read the main CSV directly
        df = pandas.read_csv(csv_path)
        
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, safe_name(l[0]))
            os.makedirs(d, exist_ok=True)
            
        for entity in tqdm.tqdm(entityList, total=len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            entityID = insData.iloc[0]['entity_id']
            
            # --- COLUMBIA DIRECT VIDEO PATH FIX ---
            videoFile = f"./columbia/colDataPath/{videoKey}.mp4"
            if not os.path.exists(videoFile):
                print(f"Warning: Could not find video at {videoFile}")
                continue 
            # --------------------------------------

            V = cv2.VideoCapture(videoFile)
            fps = V.get(cv2.CAP_PROP_FPS) 
            
            insDir = os.path.join(outDir, safe_name(videoKey), safe_name(entityID))
            os.makedirs(insDir, exist_ok=True)
            
            for _, row in insData.iterrows():
                imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                
                # --- ZERO-DRIFT FRAME SEEKING ---
                frame_num = int(round(row['frame_timestamp'] * fps))
                V.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # --------------------------------
                
                ret, frame = V.read()
                if not ret or frame is None:
                    continue
                    
                h = numpy.size(frame, 0)
                w = numpy.size(frame, 1)
                
                x1 = row['entity_box_x1']
                y1 = row['entity_box_y1']
                x2 = row['entity_box_x2']
                y2 = row['entity_box_y2']
                
                # --- COLUMBIA SPECIFIC LOWER-HALF FACE MATH (SHIFTED DOWN) ---
                # We keep the full width (x1 to x2).
                # We drop the starting line to 35% (under the nose)
                # and extend the bottom to 60% (just under the chin).
                new_x1 = x1
                new_x2 = x2
                
                new_y1 = y1 + 0.35*(y2-y1) 
                new_y2 = y1 + 0.60*(y2-y1)
                
                if new_x1 < 0: new_x1 = 0
                if new_x2 > 1: new_x2 = 1
                if new_y1 < 0: new_y1 = 0
                if new_y2 > 1: new_y2 = 1
                
                halfface = frame[int(new_y1*h):int(new_y2*h), int(new_x1*w):int(new_x2*w), :]
                
                if halfface.size > 0:
                    cv2.imwrite(imageFilename, halfface)
                
            V.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Lower Half Face images")
    parser.add_argument('--AVADataPath', type=str, default='./data/Columbia', help="AVADataPath")
    parser.add_argument('--down_out_path', type=str, default='./data/Columbia/clips_videos_down_face', help="out_path")
    # Ignored arguments kept for bash script compatibility
    parser.add_argument('--input_file_list', type=str, default='')
    parser.add_argument('--size', type=int, default=112)
    parser.add_argument('--csv_split_root', type=str, default='')
    args = parser.parse_args()
    run(args)