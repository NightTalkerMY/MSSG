#!/usr/bin/env bash

set -Eeuo pipefail

# Git Bash path corresponding to C:\Users\wwon0076\Desktop.
# Override it by passing a different archive root as the first argument.
archive_root="${1:-/c/Users/wwon0076/Desktop}"
state_dir="./predata/remaining_pipeline_state"
log_dir="./log/remaining_pipeline"

mkdir -p "$archive_root" "$state_dir" "$log_dir"

current_step="startup"
archive_pid=""
archive_job_name=""

on_error() {
    exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED during: ${current_step}" >&2
    echo "Exit code: ${exit_code}" >&2
    echo "Free space on the MSSG and archive disks:" >&2
    df -h . "$archive_root" >&2 || true
    echo "After resolving the problem, rerun this same command. Completed steps will be skipped." >&2
    exit "$exit_code"
}
trap on_error ERR

run_logged() {
    step_name=$1
    shift
    log_file="$log_dir/${step_name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entering ${step_name}..."
    "$@" 2>&1 | tee -a "$log_file"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${step_name}."
}

archive_crops() {
    source_dir=$1
    archive_name=$2
    archive_marker=$3
    destination_dir="$archive_root/$archive_name"

    if [[ -f "$archive_marker" ]]; then
        echo "Archive already verified for $archive_name; skipping."
        return
    fi

    if [[ ! -d "$source_dir" ]]; then
        echo "Expected crop directory does not exist: $source_dir" >&2
        return 1
    fi

    source_real=$(realpath "$source_dir")
    repo_real=$(realpath .)
    case "$source_real" in
        "$repo_real/data/AVA/"*) ;;
        *)
            echo "Refusing to archive an unexpected source path: $source_real" >&2
            return 1
            ;;
    esac

    mkdir -p "$destination_dir"

    echo "Copying $source_dir to $destination_dir..."
    # Copy first so a full archive disk never destroys the source data.
    cp -a "$source_dir/." "$destination_dir/"

    source_count=$(find "$source_dir" -type f | wc -l)
    destination_count=$(find "$destination_dir" -type f | wc -l)

    echo "Source files: $source_count; archived files: $destination_count"
    if [[ "$source_count" -eq 0 || "$source_count" -ne "$destination_count" ]]; then
        echo "Archive verification failed; source data has been preserved." >&2
        return 1
    fi

    echo "Archive verified. Removing the original crop directory to recover space..."
    rm -rf --one-file-system -- "$source_real"
    touch "$archive_marker"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Archived and removed $source_dir."
}

wait_for_archive() {
    if [[ -z "$archive_pid" ]]; then
        return
    fi

    echo "Waiting for background archive job: $archive_job_name"
    if wait "$archive_pid"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Background archive completed: $archive_job_name"
    else
        archive_exit=$?
        echo "Background archive failed: $archive_job_name" >&2
        echo "The original crop directory was preserved. Check $log_dir/archive_${archive_job_name}.log." >&2
        return "$archive_exit"
    fi

    archive_pid=""
    archive_job_name=""
}

start_archive() {
    source_dir=$1
    archive_name=$2
    archive_marker=$3

    # Permit one archive transfer at a time. The previous transfer overlaps
    # with the preparation and feature extraction that just completed.
    wait_for_archive

    archive_job_name="$archive_name"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting background archive: $archive_name"
    (
        set -o pipefail
        archive_crops "$source_dir" "$archive_name" "$archive_marker" 2>&1 \
            | tee -a "$log_dir/archive_${archive_name}.log"
    ) &
    archive_pid=$!
}

run_stage() {
    prep_stage=$1
    graph_stage=$2
    crop_dir=$3
    feature_dir=$4
    archive_name=$5

    prep_marker="$state_dir/data_prep_${prep_stage}.done"
    feature_marker="$state_dir/graph_init_${graph_stage}.done"
    archive_marker="$state_dir/archive_${archive_name}.done"

    if [[ ! -f "$prep_marker" ]]; then
        current_step="data_prep stage $prep_stage"
        run_logged "data_prep_${prep_stage}" bash ./data_prep.sh "$prep_stage"
        touch "$prep_marker"
    else
        echo "data_prep stage $prep_stage already completed; skipping."
    fi

    if [[ ! -f "$feature_marker" ]]; then
        current_step="graph_init stage $graph_stage"
        run_logged "graph_init_${graph_stage}" bash ./graph_init.sh "$graph_stage"

        feature_count=$(find "$feature_dir" -type f -name '*.csv' -size +0c | wc -l)
        if [[ "$feature_count" -eq 0 ]]; then
            echo "No non-empty feature CSVs found under $feature_dir." >&2
            return 1
        fi
        echo "Verified $feature_count non-empty feature CSV files under $feature_dir."
        touch "$feature_marker"
    else
        echo "graph_init stage $graph_stage already completed; skipping."
    fi

    current_step="starting archive for $archive_name"
    start_archive "$crop_dir" "$archive_name" "$archive_marker"
}

echo "MSSG disk and archive disk free space before starting:"
df -h . "$archive_root"

# data_prep stage -> graph_init feature-extraction stage
run_stage 6 4 \
    "./data/AVA/clips_videos_face_small_region" \
    "./predata/face_small_feats_info" \
    "clips_videos_face_small_region"

run_stage 7 7 \
    "./data/AVA/clips_videos_down_face" \
    "./predata/face_down_feats_info" \
    "clips_videos_down_face"

run_stage 8 8 \
    "./data/AVA/clips_videos_face_body_large" \
    "./predata/face_body_large_feats_info" \
    "clips_videos_face_body_large"

current_step="waiting for final background archive"
wait_for_archive

current_step="complete"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All remaining preparation, feature extraction, and archiving completed successfully."
