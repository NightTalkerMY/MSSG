#!/usr/bin/env bash
# Modification for Columbia Dataset

stage=$1
AVADataPath="./data/Columbia"
ColPreData="./data/Columbia/predata"  
nj=1 

# -------------------------------------------------------------------------
# STAGE 0: The AVA Illusion Setup
# -------------------------------------------------------------------------
if [ $stage -eq 0 ]; then
    echo "Setting up Columbia Preprocessing..."

    python columbia/prep_columbia.py --videoPath ./columbia/colDataPath/col.mp4 --labelPath ./columbia/colDataPath/real_labels

    echo "Columbia Preprocessing Complete!"
fi

# -------------------------------------------------------------------------
# STAGE 2: Extract F_bg
# -------------------------------------------------------------------------
if [ $stage -eq 2 ]; then 
    echo "Extracting Background..."
    python ./columbia/local/prepare_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background
    
    mkdir -p $ColPreData/background_process
    name="train"
    
    # FIXED: Point to clips_videos_background instead of clips_videos!
    python ./columbia/local/get_leaf_dir_list.py --AVADataPath=$AVADataPath/clips_videos_background/$name --out_file=$ColPreData/background_process/${name}_list
    
    background_H=224
    mkdir -p $ColPreData/background_process/split
    file_split_scp="$ColPreData/background_process/split/${name}_file_list.1"
    ./columbia/local/utils/split_scp.pl $ColPreData/background_process/${name}_list $file_split_scp
    
    mkdir -p ./log/resize_background
    mkdir -p $AVADataPath/clips_videos_background_resize_${background_H}
    
    ./columbia/local/utils/run.pl JOB=1:$nj \
    ./log/resize_background/resize_background.JOB.log \
        python ./columbia/local/resize_background.py --input_file_list=$ColPreData/background_process/split/${name}_file_list.JOB \
                                            --AVADataPath $AVADataPath \
                                            --out_path=$AVADataPath/clips_videos_background_resize_${background_H} \
                                            --size=$background_H
fi

# -------------------------------------------------------------------------
# STAGE 4: Extract F_body
# -------------------------------------------------------------------------
if [ $stage -eq 4 ]; then
    echo "Extracting Body..."
    
    # Create the output directory
    mkdir -p $AVADataPath/clips_videos_face_body
    
    # Run our smart Python script directly!
    python ./columbia/local/prepare_facebody.py \
        --AVADataPath $AVADataPath \
        --out_path $AVADataPath/clips_videos_face_body
        
    echo "Face-Body extraction complete!"
fi

# -------------------------------------------------------------------------
# STAGE 5: Extract F_head
# -------------------------------------------------------------------------
if [ $stage -eq 5 ]; then
    echo "Extracting Head..."
    
    # Create the output directory
    mkdir -p $AVADataPath/clips_videos_face_large_region
    
    # Run our smart Python script directly!
    python ./columbia/local/prepare_largeface.py \
        --AVADataPath $AVADataPath \
        --out_path $AVADataPath/clips_videos_face_large_region
        
    echo "F_head extraction complete!"
fi

# -------------------------------------------------------------------------
# STAGE 6: Extract F_smallface
# -------------------------------------------------------------------------
if [ $stage -eq 6 ]; then
    echo "Extracting Small Face..."
    
    # Create the output directory
    mkdir -p $AVADataPath/clips_videos_face_small_region
    
    # Run our smart Python script directly!
    python ./columbia/local/prepare_smallface.py \
        --AVADataPath $AVADataPath \
        --out_path $AVADataPath/clips_videos_face_small_region
        
    echo "F_smallface extraction complete!"
fi

# -------------------------------------------------------------------------
# STAGE 7: Extract F_mouse (Lower Face)
# -------------------------------------------------------------------------
if [ $stage -eq 7 ]; then
    echo "Extracting Lower Face (Mouth)..."
    
    # Create the output directory
    mkdir -p $AVADataPath/clips_videos_down_face
    
    # Run our smart Python script directly!
    python ./columbia/local/prepare_halfface.py \
        --AVADataPath $AVADataPath \
        --down_out_path $AVADataPath/clips_videos_down_face
        
    echo "Lower Face extraction complete!"
fi

# -------------------------------------------------------------------------
# STAGE 8: Extract F_largebody
# -------------------------------------------------------------------------
if [ $stage -eq 8 ]; then
    echo "Extracting Large Body..."
    name="train"
    mkdir -p $ColPreData/facebody_large_process/split
    
    find $ColPreData/csv_split/$name -type f > $ColPreData/facebody_large_process/${name}_list
    file_split_scp="$ColPreData/facebody_large_process/split/${name}_file_list.1"
    ./columbia/local/utils/split_scp.pl $ColPreData/facebody_large_process/${name}_list $file_split_scp
    
    mkdir -p ./log/facebody_large_process
    mkdir -p $AVADataPath/clips_videos_face_body_large
    
    ./columbia/local/utils/run.pl JOB=1:$nj \
    ./log/facebody_large_process/facebody_large_process.JOB.log \
        python ./columbia/local/prepare_facebody_large.py --input_file_list=$ColPreData/facebody_large_process/split/${name}_file_list.JOB \
                                                 --AVADataPath $AVADataPath \
                                                 --out_path=$AVADataPath/clips_videos_face_body_large
fi