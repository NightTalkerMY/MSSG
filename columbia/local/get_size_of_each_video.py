import os, glob, cv2, numpy, argparse, pickle

def run(args):
    # --- COLUMBIA FIX: Point directly to our raw video folder ---
    videoDir = "./columbia/colDataPath"
    videoFile = glob.glob(os.path.join(videoDir, '*.mp4'))
    
    size_dic = {}
    for cur_video in videoFile:
        video_name = os.path.splitext(os.path.basename(cur_video))[0]
        print(f"Extracting dimensions for: {video_name}")
        
        V = cv2.VideoCapture(cur_video)
        ret, frame = V.read()
        if ret:
            h = numpy.size(frame, 0)
            w = numpy.size(frame, 1)
            size_dic[video_name] = [w, h]
            print(f"Found {w}x{h}")
            
        V.release()
        
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, 'size_dic.pkl'), 'wb') as file:
        pickle.dump(size_dic, file)
        print("Successfully saved size_dic.pkl!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get Video Sizes")
    parser.add_argument('--AVADataPath', type=str, default='./data/Columbia')
    parser.add_argument('--out_path', type=str, default='./data/Columbia/predata/video_size_info')
    args = parser.parse_args()
    run(args)