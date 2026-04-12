#!/usr/bin/env bash

set -u

# ====== CONFIG ======
AVADataPath="D:/FYP/MSSG/data/AVA"
PROJECT_ROOT="D:/FYP/MSSG"
PYTHON_EXE="python"

DATASET_TYPE="train"
# OUT_PATH="$AVADataPath/clips_videos_face_body"
OUT_PATH="$AVADataPath/clips_videos_face_large_region"


# Folder to store temporary csv list files
# TMP_DIR="./predata/facebody_process/debug_missing"
# LOG_DIR="./log/facebody_process/debug_missing"
TMP_DIR="./predata/face_process/debug_missing"
LOG_DIR="./log/face_process/debug_missing"

mkdir -p "$TMP_DIR"
mkdir -p "$LOG_DIR"

# Missing video IDs to debug
VIDEO_IDS=(
  "OGNnUvJq9RI"
  "O_NYCUhZ9zw"
  "OfMdakd4bHI"
  "Ov0za6Xb1LM"
  "P90hF2S1JzA"
  "PNZQ2UJfyQE"
  "QCLQYnt3aMo"
  "QMwT7DFA5O4"
  "oD_wxyTHJ2I"
  "oq_bufAhyl8"
  "phVLLTMzmKk"
  "phrYEKv0rmw"
  "plkJ45_-pMk"
)

echo "Debugging missing facebody videos..."
echo "Dataset type: $DATASET_TYPE"
echo

for vid in "${VIDEO_IDS[@]}"; do
    csv_path="./predata/csv_split/${DATASET_TYPE}/${vid}.csv"
    list_file="$TMP_DIR/${DATASET_TYPE}_${vid}.list"
    log_file="$LOG_DIR/${DATASET_TYPE}_${vid}.log"

    echo "=========================================="
    echo "Video ID: $vid"
    echo "CSV path : $csv_path"
    echo "Log file : $log_file"

    # Check whether csv exists first
    if [ ! -f "$csv_path" ]; then
        echo "ERROR: CSV does not exist: $csv_path" | tee "$log_file"
        continue
    fi

    # Write a one-line list file because prepare_facebody.py expects --input_file_list
    printf "%s\n" "$csv_path" > "$list_file"

    # Run and capture all stdout/stderr
    $PYTHON_EXE ./local/prepare_facebody.py \
        --input_file_list="$list_file" \
        --AVADataPath="$AVADataPath" \
        --out_path="$OUT_PATH" \
        > "$log_file" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $vid finished successfully"
    else
        echo "FAILED : $vid exited with code $exit_code"
        echo "Check log: $log_file"
    fi
done

echo
echo "Done. Logs are in: $LOG_DIR"