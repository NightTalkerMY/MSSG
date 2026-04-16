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
    echo "Setting up the Fake AVA structure with Headers..."
    
    mkdir -p $AVADataPath/csv
    # HEADER="video_id,frame_timestamp,entity_box_x1,entity_box_y1,entity_box_x2,entity_box_y2,instance_id,entity_id,label_id"
    HEADER="video_id,frame_timestamp,entity_box_x1,entity_box_y1,entity_box_x2,entity_box_y2,label,entity_id,label_id,instance_id"
    
    echo $HEADER > $AVADataPath/csv/train_orig.csv
    cat $ColPreData/csv_split/columbia/col_split.csv >> $AVADataPath/csv/train_orig.csv
    
    echo $HEADER > $AVADataPath/csv/val_orig.csv
    cat $ColPreData/csv_split/columbia/col_split.csv >> $AVADataPath/csv/val_orig.csv
    
    mkdir -p $ColPreData/csv_split/train
    cp $ColPreData/csv_split/columbia/col_split.csv $ColPreData/csv_split/train/col_split.csv
    
    if [ -d "$AVADataPath/clips_videos/col" ]; then
        mkdir -p $AVADataPath/clips_videos/train
        mv $AVADataPath/clips_videos/col $AVADataPath/clips_videos/train/col
    fi

    mkdir -p $AVADataPath/orig_videos/trainval
    cp columbia/colDataPath/col.mp4 $AVADataPath/orig_videos/trainval/col.mp4

    echo "AVA Illusion Complete!"
fi

# -------------------------------------------------------------------------
# STAGE 2: Extract F_bg
# -------------------------------------------------------------------------
if [ $stage -eq 2 ]; then 
    echo "Extracting Background..."
    python ./local/prepare_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background
    
    mkdir -p $ColPreData/background_process
    name="train"
    
    # FIXED: Point to clips_videos_background instead of clips_videos!
    python ./local/get_leaf_dir_list.py --AVADataPath=$AVADataPath/clips_videos_background/$name --out_file=$ColPreData/background_process/${name}_list
    
    background_H=224
    mkdir -p $ColPreData/background_process/split
    file_split_scp="$ColPreData/background_process/split/${name}_file_list.1"
    ./local/utils/split_scp.pl $ColPreData/background_process/${name}_list $file_split_scp
    
    mkdir -p ./log/resize_background
    mkdir -p $AVADataPath/clips_videos_background_resize_${background_H}
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/resize_background/resize_background.JOB.log \
        python ./local/resize_background.py --input_file_list=$ColPreData/background_process/split/${name}_file_list.JOB \
                                            --AVADataPath $AVADataPath \
                                            --out_path=$AVADataPath/clips_videos_background_resize_${background_H} \
                                            --size=$background_H
fi

# -------------------------------------------------------------------------
# STAGE 4: Extract F_body
# -------------------------------------------------------------------------
if [ $stage -eq 4 ]; then
    echo "Extracting Body..."
    name="train"
    mkdir -p $ColPreData/facebody_process/split
    
    find $ColPreData/csv_split/$name -type f > $ColPreData/facebody_process/${name}_list
    file_split_scp="$ColPreData/facebody_process/split/${name}_file_list.1"
    ./local/utils/split_scp.pl $ColPreData/facebody_process/${name}_list $file_split_scp
    
    mkdir -p ./log/facebody_process
    mkdir -p $AVADataPath/clips_videos_face_body
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/facebody_process/facebody_process.JOB.log \
        python ./local/prepare_facebody.py --input_file_list=$ColPreData/facebody_process/split/${name}_file_list.JOB \
                                           --AVADataPath $AVADataPath \
                                           --out_path=$AVADataPath/clips_videos_face_body
fi

# -------------------------------------------------------------------------
# STAGE 5: Extract F_head
# -------------------------------------------------------------------------
if [ $stage -eq 5 ]; then
    echo "Extracting Head..."
    name="train"
    mkdir -p $ColPreData/face_process/split
    
    find $ColPreData/csv_split/$name -type f > $ColPreData/face_process/${name}_list
    file_split_scp="$ColPreData/face_process/split/${name}_file_list.1"
    ./local/utils/split_scp.pl $ColPreData/face_process/${name}_list $file_split_scp
    
    mkdir -p ./log/face_process
    mkdir -p $AVADataPath/clips_videos_face_large_region
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/face_process/face_process.JOB.log \
        python ./local/prepare_largeface.py --input_file_list=$ColPreData/face_process/split/${name}_file_list.JOB \
                                            --AVADataPath $AVADataPath \
                                            --out_path=$AVADataPath/clips_videos_face_large_region
fi

# -------------------------------------------------------------------------
# STAGE 6: Extract F_smallface
# -------------------------------------------------------------------------
if [ $stage -eq 6 ]; then
    echo "Extracting Small Face..."
    name="train"
    mkdir -p $ColPreData/face_process_small/split
    
    find $ColPreData/csv_split/$name -type f > $ColPreData/face_process_small/${name}_list
    file_split_scp="$ColPreData/face_process_small/split/${name}_file_list.1"
    ./local/utils/split_scp.pl $ColPreData/face_process_small/${name}_list $file_split_scp
    
    mkdir -p ./log/face_process_small
    mkdir -p $AVADataPath/clips_videos_face_small_region
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/face_process_small/face_process_small.JOB.log \
        python ./local/prepare_smallface.py --input_file_list=$ColPreData/face_process_small/split/${name}_file_list.JOB \
                                            --AVADataPath $AVADataPath \
                                            --out_path=$AVADataPath/clips_videos_face_small_region
fi

# -------------------------------------------------------------------------
# STAGE 7: Extract F_mouse (Lower Face)
# -------------------------------------------------------------------------
if [ $stage -eq 7 ]; then
    echo "Extracting Lower Face..."
    name="train"
    mkdir -p $ColPreData/face_process
    
    # FIX: Point to clips_videos_faces!
    python ./local/get_leaf_dir_list.py --AVADataPath=$AVADataPath/clips_videos_faces/$name --out_file=$ColPreData/face_process/${name}_list
    
    mkdir -p $ColPreData/face_process/split
    file_split_scp="$ColPreData/face_process/split/${name}_file_list.1"
    ./local/utils/split_scp.pl $ColPreData/face_process/${name}_list $file_split_scp
    
    mkdir -p ./log/extract_mouse
    mkdir -p $AVADataPath/clips_videos_down_face
    
    nj=1
    ./local/utils/run.pl JOB=1:$nj \
    ./log/extract_mouse/extract_mouse.JOB.log \
        python ./local/prepare_halfface.py --input_file_list=$ColPreData/face_process/split/${name}_file_list.JOB \
                                           --AVADataPath $AVADataPath \
                                           --down_out_path=$AVADataPath/clips_videos_down_face \
                                           --csv_split_root=$ColPreData/csv_split/$name
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
    ./local/utils/split_scp.pl $ColPreData/facebody_large_process/${name}_list $file_split_scp
    
    mkdir -p ./log/facebody_large_process
    mkdir -p $AVADataPath/clips_videos_face_body_large
    
    ./local/utils/run.pl JOB=1:$nj \
    ./log/facebody_large_process/facebody_large_process.JOB.log \
        python ./local/prepare_facebody_large.py --input_file_list=$ColPreData/facebody_large_process/split/${name}_file_list.JOB \
                                                 --AVADataPath $AVADataPath \
                                                 --out_path=$AVADataPath/clips_videos_face_body_large
fi