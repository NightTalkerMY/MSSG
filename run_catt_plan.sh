#!/usr/bin/env bash
set -Eeuo pipefail
python_cmd="${PYTHON:-python}"
command_name="${1:-help}"

run_training() {
    PYTHONHASHSEED="$2" "$python_cmd" proposed/train_student_seeded.py \
        --variant "$1" --seed "$2" --epochs "$3" --run-name "$4"
}

case "$command_name" in
    audit-inputs) "$python_cmd" proposed/audit_student_inputs.py ;;
    build-train-cache) "$python_cmd" proposed/build_ava_cache.py --split train ;;
    build-val-cache) "$python_cmd" proposed/build_ava_cache.py --split val ;;
    audit-caches) "$python_cmd" proposed/audit_student_caches.py ;;
    audit-columbia-inputs) "$python_cmd" proposed/audit_columbia_inputs.py ;;
    build-columbia-cache) "$python_cmd" proposed/build_columbia_cache.py ;;
    audit-columbia-cache) "$python_cmd" proposed/audit_columbia_cache.py ;;
    smoke) run_training ce_mse 42 1 smoke_ce_mse_seed_42 ;;
    single) run_training ce_mse 42 50 ce_mse_seed_42 ;;
    single-ablation) run_training ce_only 42 50 ce_only_seed_42 ;;
    multi)
        for seed in 42 123 2026; do
            for variant in ce_mse ce_only; do
                run_name="${variant}_seed_${seed}"
                output="realtime/experiments/$run_name"
                if [[ -f "$output/run_manifest.json" ]]; then
                    echo "Completed run exists; skipping $run_name"
                else
                    run_training "$variant" "$seed" 50 "$run_name"
                fi
            done
        done
        ;;
    help|*)
        cat <<'EOF'
Usage: bash run_catt_plan.sh COMMAND
  audit-inputs       Verify all seven AVA CSV families.
  build-train-cache  Build reusable AVA training tensors.
  build-val-cache    Build reusable AVA validation tensors.
  audit-caches       Validate both tensor caches.
  audit-columbia-inputs  Verify all seven held-out Columbia feature files.
  build-columbia-cache   Build reusable held-out Columbia tensors.
  audit-columbia-cache   Validate the held-out Columbia cache.
  smoke              One epoch, seed 42, CE+MSE; pipeline check only.
  single             Full 50-epoch proposed run, seed 42.
  single-ablation    Full 50-epoch CE-only run, seed 42.
  multi              Three paired seeds; run only after reviewing single-seed results.
EOF
        ;;
esac
