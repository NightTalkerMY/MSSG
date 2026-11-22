#!/usr/bin/env bash

set -euo pipefail

stages=(6 7 8)

for stage in "${stages[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entering data_prep stage ${stage}..."

    bash ./data_prep.sh "${stage}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed data_prep stage ${stage}."
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All remaining data_prep stages completed successfully."
