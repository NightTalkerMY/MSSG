# 1. Manually create the split file (Copy train_orig.csv but skip the 1st header line)
mkdir -p ./data/Columbia/predata/csv_split/train
tail -n +2 ./data/Columbia/csv/train_orig.csv > ./data/Columbia/predata/csv_split/train/col.csv

# 2. Point our list to the newly generated split
mkdir -p ./log
echo "./data/Columbia/predata/csv_split/train/col.csv" > ./log/col_graph_list.txt

# 3. Extract the video size info
python ./local/get_size_of_each_video.py \
  --AVADataPath ./data/Columbia \
  --out_path ./data/Columbia/predata/video_size_info

# 4. Construct the final graph!
python ./local/generate_graph.py \
  --input_file_list=./log/col_graph_list.txt \
  --face_feats_info_path=./data/Columbia/predata/face_feats_info/train \
  --background_feats_info_path=./data/Columbia/predata/background_feats_info/train \
  --face_body_feats_info_path=./data/Columbia/predata/face.body_feats_info/train \
  --face_large_feats_info_path=./data/Columbia/predata/face_large_feats_info/train \
  --face_small_feats_info_path=./data/Columbia/predata/face_small_feats_info/train \
  --face_down_feats_info_path=./data/Columbia/predata/face_down_feats_info/train \
  --face_body_large_feats_info_path=./data/Columbia/predata/face_body_large_feats_info/train \
  --out_path=./data/Columbia/graphs/graph_2000_0/train/processed \
  --numv=2000 \
  --time_edge=0 \
  --video_size_info=./data/Columbia/predata/video_size_info/size_dic.pkl