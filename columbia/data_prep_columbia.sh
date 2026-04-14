#!/usr/bin/env bash
# Modification for Columbia Dataset
# Automatically move up one folder to the root of the project
# so that ./local/ and ./predata/ can be found correctly!
cd ..

stage=$1

# Configure the path to your new Columbia dataset directory
ColDataPath="./data/Columbia"
CSV_DIR="./predata/csv_split/columbia"

# We use nj=1 because we only have a handful of CSV files (bell, boll, etc.) 
# compared to the thousands in AVA. Splitting into 64 jobs will crash the script.
nj=1 

mkdir -p ./predata
mkdir -p $CSV_DIR

# -------------------------------------------------------------------------
# STAGE 0: Directory Preparation & Symlinking (CRITICAL)
# The original Python scripts look for frames inside $ColDataPath/clips_videos/
# We need to make sure the frames extracted by prep_columbia.py are here.
# -------------------------------------------------------------------------
if [ $stage -eq 0 ]; then
    echo "Please ensure your col_split.csv is placed inside $CSV_DIR"
    echo "Please ensure your raw frames are in $ColDataPath/clips_videos/"
    # Example: If your video is "col", frames should be in $ColDataPath/clips_videos/col/
fi

# -------------------------------------------------------------------------
# STAGE 1 & 2: Extract and Resize F_bg (Eq 8)
# -------------------------------------------------------------------------
if [ $stage -eq 2 ]; then 
    echo "Extracting Background..."
    python ./local/prepare_background.py --AVADataPath=$ColDataPath --out_path=$ColDataPath/clips_videos_background
    
    mkdir -p ./predata/background_process/columbia_list
    python ./local/get_leaf_dir_list.py --AVADataPath=$ColDataPath/clips_videos --out_file=./predata/background_process/columbia_list

    background_H=224
    mkdir -p ./predata/background_process/split
    
    file_split_scp="./predata/background_process/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/background_process/columbia_list $file_split_scp
    
    mkdir -p ./log/resize_background
    mkdir -p $ColDataPath/clips_videos_background_resize_${background_H}
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/resize_background/resize_background.JOB.log \
        python ./local/resize_background.py --input_file_list=./predata/background_process/split/columbia_file_list.JOB \
                                            --AVADataPath $ColDataPath \
                                            --out_path=$ColDataPath/clips_videos_background_resize_${background_H} \
                                            --size=$background_H
fi

# -------------------------------------------------------------------------
# STAGE 3: Extract F_body (Eq 6)
# -------------------------------------------------------------------------
if [ $stage -eq 4 ]; then
    echo "Extracting Body..."
    mkdir -p ./predata/facebody_process/split
    
    find $CSV_DIR -type f > ./predata/facebody_process/columbia_list
    file_split_scp="./predata/facebody_process/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/facebody_process/columbia_list $file_split_scp
    
    mkdir -p ./log/facebody_process
    mkdir -p $ColDataPath/clips_videos_face_body
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/facebody_process/facebody_process.JOB.log \
        python ./local/prepare_facebody.py --input_file_list=./predata/facebody_process/split/columbia_file_list.JOB \
                                           --AVADataPath $ColDataPath \
                                           --out_path=$ColDataPath/clips_videos_face_body
fi

# -------------------------------------------------------------------------
# STAGE 4: Extract F_head (Eq 3)
# -------------------------------------------------------------------------
if [ $stage -eq 5 ]; then
    echo "Extracting Head..."
    mkdir -p ./predata/face_process/split
    
    find $CSV_DIR -type f > ./predata/face_process/columbia_list
    file_split_scp="./predata/face_process/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/face_process/columbia_list $file_split_scp
    
    mkdir -p ./log/face_process
    mkdir -p $ColDataPath/clips_videos_face_large_region
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/face_process/face_process.JOB.log \
        python ./local/prepare_largeface.py --input_file_list=./predata/face_process/split/columbia_file_list.JOB \
                                            --AVADataPath $ColDataPath \
                                            --out_path=$ColDataPath/clips_videos_face_large_region
fi

# -------------------------------------------------------------------------
# STAGE 5: Extract F_smallface (Eq 4)
# -------------------------------------------------------------------------
if [ $stage -eq 6 ]; then
    echo "Extracting Small Face..."
    mkdir -p ./predata/face_process_small/split
    
    find $CSV_DIR -type f > ./predata/face_process_small/columbia_list
    file_split_scp="./predata/face_process_small/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/face_process_small/columbia_list $file_split_scp
    
    mkdir -p ./log/face_process_small
    mkdir -p $ColDataPath/clips_videos_face_small_region
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/face_process_small/face_process_small.JOB.log \
        python ./local/prepare_smallface.py --input_file_list=./predata/face_process_small/split/columbia_file_list.JOB \
                                            --AVADataPath $ColDataPath \
                                            --out_path=$ColDataPath/clips_videos_face_small_region
fi

# -------------------------------------------------------------------------
# STAGE 6: Extract F_mouse (Eq 5)
# -------------------------------------------------------------------------
if [ $stage -eq 7 ]; then
    echo "Extracting Mouse (Lower Face)..."
    python ./local/get_leaf_dir_list.py --AVADataPath=$ColDataPath/clips_videos --out_file=./predata/face_process/columbia_list_dirs
    
    mkdir -p ./predata/face_process/split
    file_split_scp="./predata/face_process/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/face_process/columbia_list_dirs $file_split_scp
    
    mkdir -p ./log/extract_mouse
    mkdir -p $ColDataPath/clips_videos_down_face
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/extract_mouse/extract_mouse.JOB.log \
        python ./local/prepare_halfface.py --input_file_list=./predata/face_process/split/columbia_file_list.JOB \
                                           --AVADataPath $ColDataPath \
                                           --down_out_path=$ColDataPath/clips_videos_down_face \
                                           --csv_split_root=$CSV_DIR
fi

# -------------------------------------------------------------------------
# STAGE 7: Extract F_largebody (Eq 7)
# -------------------------------------------------------------------------
if [ $stage -eq 8 ]; then
    echo "Extracting Large Body..."
    mkdir -p ./predata/facebody_large_process/split
    
    find $CSV_DIR -type f > ./predata/facebody_large_process/columbia_list
    file_split_scp="./predata/facebody_large_process/split/columbia_file_list.1"
    ./local/utils/split_scp.pl ./predata/facebody_large_process/columbia_list $file_split_scp
    
    mkdir -p ./log/facebody_large_process
    mkdir -p $ColDataPath/clips_videos_face_body_large
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/facebody_large_process/facebody_large_process.JOB.log \
        python ./local/prepare_facebody_large.py --input_file_list=./predata/facebody_large_process/split/columbia_file_list.JOB \
                                                 --AVADataPath $ColDataPath \
                                                 --out_path=$ColDataPath/clips_videos_face_body_large
fi