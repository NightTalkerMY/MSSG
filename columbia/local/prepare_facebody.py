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
        outDir = os.path.join(args.out_path, dataType)
        
        # We read the main CSV directly instead of dealing with the messy file lists
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
                
                # Expand box for Face+Body
                new_x1 = x1 - 0.75*(x2-x1)
                new_x2 = x2 + 0.75*(x2-x1)
                if new_x1 < 0: new_x1 = 0
                if new_x2 > 1: new_x2 = 1
                
                new_y1 = y1
                new_y2 = y2 + 3*(y2-y1)
                if new_y2 > 1: new_y2 = 1
                
                facebody = frame[int(new_y1*h):int(new_y2*h), int(new_x1*w):int(new_x2*w), :]
                
                if facebody.size > 0:
                    cv2.imwrite(imageFilename, facebody)
                
            V.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract FaceBody images")
    parser.add_argument('--AVADataPath', type=str, default='./data/Columbia', help="AVADataPath")
    parser.add_argument('--out_path', type=str, default='./data/Columbia/clips_videos_facebody', help="out_path")
    # Kept to prevent argparse from crashing if the bash script passes it
    parser.add_argument('--input_file_list', type=str, default='', help="Ignored. Reads directly from AVADataPath.")
    args = parser.parse_args()
    run(args)