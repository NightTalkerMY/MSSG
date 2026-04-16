#!/usr/bin/env bash
# Modification for Columbia Dataset Feature Extraction

stage=$1

AVADataPath="./data/Columbia"
ColPreData="./data/Columbia/predata"
GPU=0

# generate baseline face crops (equivalent to tools.py extract_video_clips)
if [ $stage -eq 1 ]; then
    echo "Generating Baseline Face Crops for Columbia..."
    python ./columbia/prep_baseline_faces.py
fi

# initialize node e_face
if [ $stage -eq 2 ]; then
    echo "Extracting Face Features..."
    
    mkdir -p ./data/Columbia/predata/clip_info
        
    python ./columbia/local/get_clip_info.py --dataPathAVA $AVADataPath --out_path $ColPreData/clip_info
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_feats_average.py --dataPathAVA $AVADataPath \
                                   --clip_info_path $ColPreData/clip_info \
                                   --face_model_path ./predata/pretrain_model/face_model \
                                   --out_path $ColPreData/face_feats_info
fi

# initialize node e_bg
if [ $stage -eq 3 ]; then
    echo "Extracting Background Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_background_feats.py --dataPathAVA $AVADataPath \
                                   --clip_info_path $ColPreData/clip_info \
                                   --background_model ./predata/pretrain_model/background_model.model \
                                   --out_path $ColPreData/background_feats_info \
                                   --background_H 224
fi

# initialize node e_smallface
if [ $stage -eq 4 ]; then
    echo "Extracting Small Face Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_small_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path $ColPreData/clip_info \
                            --face_model_path ./predata/pretrain_model/face_small_model \
                            --out_path $ColPreData/face_small_feats_info
fi

# initialize node e_body
if [ $stage -eq 5 ]; then 
    echo "Extracting Body Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_body_feats_average.py --dataPathAVA $AVADataPath \
                        --clip_info_path $ColPreData/clip_info \
                        --face_model_path ./predata/pretrain_model/body_model \
                        --out_path $ColPreData/face.body_feats_info
fi

# initialize node e_head
if [ $stage -eq 6 ]; then
    echo "Extracting Large Face (Head) Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_large_feats_average.py --dataPathAVA $AVADataPath \
                                --clip_info_path $ColPreData/clip_info \
                                --face_model_path ./predata/pretrain_model/face_large_model \
                                --out_path $ColPreData/face_large_feats_info
fi

# initialize node e_mouse (lower face)
if [ $stage -eq 7 ]; then
    echo "Extracting Lower Face Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_down_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path $ColPreData/clip_info \
                            --face_model_path ./predata/pretrain_model/face_down_model \
                            --out_path $ColPreData/face_down_feats_info
fi

# initialize node e_largebody
if [ $stage -eq 8 ]; then
    echo "Extracting Large Body Features..."
    CUDA_VISIBLE_DEVICES=${GPU} python ./columbia/local/extract_face_body_large_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path $ColPreData/clip_info \
                            --face_model_path ./predata/pretrain_model/body_large_model \
                            --out_path $ColPreData/face_body_large_feats_info
fi

# generate speaker graph G
if [ $stage -eq 9 ]; then
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
fi