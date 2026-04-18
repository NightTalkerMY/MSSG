import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse

def safe_name(name):
    return str(name).replace(":", "_")

def run(args):
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val']:
        csv_path = os.path.join(args.AVADataPath, 'csv', '%s_orig.csv'%(dataType))
        
        if not os.path.exists(csv_path):
            print(f"Skipping {dataType}... File not found: {csv_path}")
            continue

        print(f'Deal with {dataType}...')
        outDir = os.path.join(args.out_path, dataType)
        df = pandas.read_csv(csv_path)
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, safe_name(l[0]))
            os.makedirs(d, exist_ok=True)
            
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            entityID = insData.iloc[0]['entity_id']
            
            # --- COLUMBIA DIRECT VIDEO PATH FIX ---
            # Point directly to your actual video location!
            videoFile = f"./columbia/colDataPath/{videoKey}.mp4"
            
            if not os.path.exists(videoFile):
                print(f"Warning: Could not find video at {videoFile}")
                continue 
            # --------------------------------------

            V = cv2.VideoCapture(videoFile)
            fps = V.get(cv2.CAP_PROP_FPS) # Get exact FPS for math
            
            insDir = os.path.join(outDir, safe_name(videoKey), safe_name(entityID))
            os.makedirs(insDir, exist_ok=True)
            
            j = 0
            for _, row in insData.iterrows():
                imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                
                # --- ZERO-DRIFT FRAME SEEKING ---
                # Convert timestamp back to exact frame number to avoid Time Slips!
                frame_num = int(round(row['frame_timestamp'] * fps))
                V.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # --------------------------------
                
                ret, frame = V.read()
                if not ret or frame is None:
                    continue

                h = numpy.size(frame, 0)
                w = numpy.size(frame, 1)
                x1 = int(row['entity_box_x1'] * w)
                y1 = int(row['entity_box_y1'] * h)
                x2 = int(row['entity_box_x2'] * w)
                y2 = int(row['entity_box_y2'] * h)
                
                # Black out the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=cv2.FILLED)
                j = j+1
                cv2.imwrite(imageFilename, frame)
                
            V.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='./data/Columbia', help="AVADataPath")
    parser.add_argument('--out_path', type=str, dest='out_path', default='./data/Columbia/clips_videos_background', help="out_path")
    args = parser.parse_args()
    run(args)