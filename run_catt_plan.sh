#!/usr/bin/env bash
set -Eeuo pipefail
python_cmd="${PYTHON:-python}"
command_name="${1:-help}"
notify_enabled=false

if [[ $# -gt 0 ]]; then
    shift
fi
for option in "$@"; do
    case "$option" in
        --notify) notify_enabled=true ;;
        *)
            echo "Unknown option: $option"
            echo "Supported optional flag: --notify"
            exit 2
            ;;
    esac
done

notify_event() {
    local status="$1"
    local elapsed="${2:-0}"
    local exit_code="${3:-}"
    if [[ "$notify_enabled" != true ]]; then
        return 0
    fi
    local arguments=(
        proposed/telegram_notify.py
        --status "$status"
        --command "$command_name"
        --elapsed-seconds "$elapsed"
    )
    if [[ -n "$exit_code" ]]; then
        arguments+=(--exit-code "$exit_code")
    fi
    "$python_cmd" "${arguments[@]}" || true
}

started_epoch=$(date +%s)
finish_notification() {
    local status=$?
    trap - EXIT
    local elapsed=$(( $(date +%s) - started_epoch ))
    if [[ $status -eq 0 ]]; then
        notify_event succeeded "$elapsed" "$status"
    else
        notify_event failed "$elapsed" "$status"
    fi
    exit "$status"
}

if [[ "$notify_enabled" == true ]]; then
    notify_event started 0
    trap finish_notification EXIT
fi

run_training() {
    PYTHONHASHSEED="$2" "$python_cmd" proposed/train_student_seeded.py \
        --variant "$1" --seed "$2" --epochs "$3" --run-name "$4"
}

case "$command_name" in
    download-columbia) "$python_cmd" columbia/download_columbia.py ;;
    audit-inputs) "$python_cmd" proposed/audit_student_inputs.py ;;
    audit-inputs-deep) "$python_cmd" proposed/audit_student_features_deep.py ;;
    prepare-ava-labels) "$python_cmd" proposed/prepare_ava_cache_labels.py ;;
    build-train-cache) "$python_cmd" proposed/build_ava_cache.py --split train ;;
    build-val-cache) "$python_cmd" proposed/build_ava_cache.py --split val ;;
    audit-caches) "$python_cmd" proposed/audit_student_caches.py ;;
    audit-columbia-inputs) "$python_cmd" proposed/audit_columbia_inputs.py ;;
    build-columbia-cache) "$python_cmd" proposed/build_columbia_cache.py ;;
    audit-columbia-cache) "$python_cmd" proposed/audit_columbia_cache.py ;;
    build-mssg-ava-val-graphs) MSSG_GRAPH_SPLITS=val bash graph_init.sh 9 ;;
    build-mssg-ava-graphs) bash graph_init.sh 9 ;;
    evaluate-mssg-ava) "$python_cmd" proposed/evaluate_mssg_streaming.py --dataset ava ;;
    evaluate-mssg-columbia) "$python_cmd" proposed/evaluate_mssg_streaming.py --dataset columbia ;;
    evaluate-mssg-native-ava) "$python_cmd" proposed/evaluate_mssg_native.py --dataset ava ;;
    evaluate-mssg-native-columbia) "$python_cmd" proposed/evaluate_mssg_native.py --dataset columbia ;;
    evaluate-lrasd-ava) "$python_cmd" proposed/evaluate_lrasd_native.py --dataset ava ;;
    evaluate-lrasd-columbia) "$python_cmd" proposed/evaluate_lrasd_native.py --dataset columbia ;;
    notify-test) echo "Telegram notification test completed." ;;
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
Usage: bash run_catt_plan.sh COMMAND [--notify]
  download-columbia  Download and inventory Columbia video and LR-ASD labels.
  audit-inputs       Verify all seven AVA CSV families.
  audit-inputs-deep  Read and validate every feature CSV row and value.
  prepare-ava-labels Create/audit per-video AVA labels needed by cache building.
  build-train-cache  Build reusable AVA training tensors.
  build-val-cache    Build reusable AVA validation tensors.
  audit-caches       Validate both tensor caches.
  audit-columbia-inputs  Verify all seven held-out Columbia feature files.
  build-columbia-cache   Build reusable held-out Columbia tensors.
  audit-columbia-cache   Validate the held-out Columbia cache.
  build-mssg-ava-val-graphs Build only the native AVA validation graphs needed for evaluation.
  build-mssg-ava-graphs Build both native MSSG AVA training and validation graphs.
  evaluate-mssg-ava      Evaluate retained MSSG on AVA under the matched 50-frame protocol.
  evaluate-mssg-columbia Evaluate retained MSSG on Columbia under the matched 50-frame protocol.
  evaluate-mssg-native-ava      Optional native MSSG graph-node evaluation on AVA.
  evaluate-mssg-native-columbia Optional native MSSG graph-node evaluation on Columbia.
  evaluate-lrasd-ava     Evaluate official LR-ASD weights on AVA tracks.
  evaluate-lrasd-columbia Evaluate official LR-ASD weights on corrected Columbia tracks.
  notify-test         Test optional Telegram start/success notifications.
  smoke              One epoch, seed 42, CE+MSE; pipeline check only.
  single             Full 50-epoch proposed run, seed 42.
  single-ablation    Full 50-epoch CE-only run, seed 42.
  multi              Three paired seeds; run only after reviewing single-seed results.

Optional:
  --notify           Read .env and send Telegram start/success/failure messages.

Environment:
  MSSG_GRAPH_NJ      Stage-9 graph workers (default: 6).
  MSSG_GRAPH_SPLITS  Stage-9 splits to build (default: "train val").
EOF
        ;;
esac
