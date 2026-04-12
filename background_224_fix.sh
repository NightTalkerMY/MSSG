#!/usr/bin/env bash
set -e

# ===== CONFIG =====
AVADataPath="./data/AVA"

# ===== RUN =====
mkdir -p ./predata/background_process

for name in train val; do
    echo "Generating list for $name ..."
    python ./local/get_leaf_dir_list.py \
        --AVADataPath="$AVADataPath/clips_videos_background/$name" \
        --out_file="./predata/background_process/${name}_list"
done

echo "Done generating lists."

