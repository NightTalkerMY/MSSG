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
    local arguments=(
        proposed/train_student_seeded.py
        --variant "$1" --seed "$2" --epochs "$3" --run-name "$4"
    )
    if [[ -n "${5:-}" ]]; then
        arguments+=(--alpha "$5")
    fi
    arguments+=(--architecture "${6:-full}")
    PYTHONHASHSEED="$2" "$python_cmd" "${arguments[@]}"
}

evaluate_multiseed_outputs() {
    for seed in 42 123 2026; do
        for run_name in "ce_mse_alpha_08_seed_${seed}" "ce_only_seed_${seed}"; do
            for dataset in ava columbia; do
                result="realtime/catt_results/$run_name/$dataset/result.json"
                if [[ -f "$result" ]]; then
                    echo "Completed result exists; skipping $run_name $dataset"
                else
                    "$python_cmd" proposed/evaluate_student_checkpoint.py \
                        --run-name "$run_name" --dataset "$dataset"
                fi
            done
        done
    done
}

train_remaining_ablations() {
    for architecture in no_buffer no_cross_speaker noncausal; do
        run_name="ce_mse_alpha_08_${architecture}_seed_42"
        output="realtime/experiments/$run_name"
        if [[ -f "$output/run_manifest.json" && -f "$output/best_ava.pt" ]]; then
            echo "Completed run exists; skipping $run_name"
        else
            run_training ce_mse 42 50 "$run_name" 0.8 "$architecture"
        fi
    done
}

evaluate_remaining_ablations() {
    for architecture in no_buffer no_cross_speaker noncausal; do
        run_name="ce_mse_alpha_08_${architecture}_seed_42"
        for dataset in ava columbia; do
            result="realtime/catt_results/$run_name/$dataset/result.json"
            if [[ -f "$result" ]]; then
                echo "Completed result exists; skipping $run_name $dataset"
            else
                "$python_cmd" proposed/evaluate_student_checkpoint.py \
                    --run-name "$run_name" --dataset "$dataset"
            fi
        done
    done
}

run_remaining_ablations() {
    for architecture in no_buffer no_cross_speaker noncausal; do
        run_name="ce_mse_alpha_08_${architecture}_seed_42"
        output="realtime/experiments/$run_name"
        if [[ -f "$output/run_manifest.json" && -f "$output/best_ava.pt" ]]; then
            echo "Completed run exists; skipping $run_name"
        else
            run_training ce_mse 42 50 "$run_name" 0.8 "$architecture"
        fi
        for dataset in ava columbia; do
            result="realtime/catt_results/$run_name/$dataset/result.json"
            if [[ -f "$result" ]]; then
                echo "Completed result exists; skipping $run_name $dataset"
            else
                "$python_cmd" proposed/evaluate_student_checkpoint.py \
                    --run-name "$run_name" --dataset "$dataset"
            fi
        done
    done
}

run_required_ablations() {
    for architecture in no_buffer no_cross_speaker; do
        run_name="ce_mse_alpha_08_${architecture}_seed_42"
        output="realtime/experiments/$run_name"
        if [[ -f "$output/run_manifest.json" && -f "$output/best_ava.pt" ]]; then
            echo "Completed run exists; skipping $run_name"
        else
            run_training ce_mse 42 50 "$run_name" 0.8 "$architecture"
        fi
        for dataset in ava columbia; do
            result="realtime/catt_results/$run_name/$dataset/result.json"
            if [[ -f "$result" ]]; then
                echo "Completed result exists; skipping $run_name $dataset"
            else
                "$python_cmd" proposed/evaluate_student_checkpoint.py \
                    --run-name "$run_name" --dataset "$dataset"
            fi
        done
    done
}

verify_catt_causality() {
    for seed in 42 123 2026; do
        run_name="ce_mse_alpha_08_seed_${seed}"
        output="realtime/analysis/causality/${run_name}.json"
        if [[ -f "$output" ]]; then
            echo "Completed causality result exists; skipping $run_name"
        else
            "$python_cmd" proposed/verify_student_causality.py \
                --run-name "$run_name" --expect causal
        fi
    done
    run_name="ce_mse_alpha_08_noncausal_seed_42"
    output="realtime/analysis/causality/${run_name}.json"
    if [[ -f "$output" ]]; then
        echo "Completed causality result exists; skipping $run_name"
    else
        "$python_cmd" proposed/verify_student_causality.py \
            --run-name "$run_name" --expect noncausal
    fi
}

verify_selected_catt_causality() {
    for seed in 42 123 2026; do
        run_name="ce_mse_alpha_08_seed_${seed}"
        output="realtime/analysis/causality/${run_name}.json"
        if [[ -f "$output" ]]; then
            echo "Completed causality result exists; skipping $run_name"
        else
            "$python_cmd" proposed/verify_student_causality.py \
                --run-name "$run_name" --expect causal
        fi
    done
}

evaluate_catt_av_sync() {
    for seed in 42 123 2026; do
        run_name="ce_mse_alpha_08_seed_${seed}"
        output="realtime/analysis/av_sync/${run_name}/result.json"
        if [[ -f "$output" ]]; then
            echo "Completed AV-sync result exists; skipping $run_name"
        else
            "$python_cmd" proposed/evaluate_av_sync.py --run-name "$run_name"
        fi
    done
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
    evaluate-lrasd-ava) "$python_cmd" proposed/evaluate_lrasd_streaming.py --dataset ava ;;
    evaluate-lrasd-columbia) "$python_cmd" proposed/evaluate_lrasd_streaming.py --dataset columbia ;;
    evaluate-lrasd-native-ava) "$python_cmd" proposed/evaluate_lrasd_native.py --dataset ava ;;
    evaluate-lrasd-native-columbia) "$python_cmd" proposed/evaluate_lrasd_native.py --dataset columbia ;;
    evaluate-catt-seed42)
        for run_name in ce_mse_seed_42 ce_only_seed_42; do
            for dataset in ava columbia; do
                result="realtime/catt_results/$run_name/$dataset/result.json"
                if [[ -f "$result" ]]; then
                    echo "Completed result exists; skipping $run_name $dataset"
                else
                    "$python_cmd" proposed/evaluate_student_checkpoint.py \
                        --run-name "$run_name" --dataset "$dataset"
                fi
            done
        done
        ;;
    tune-catt-alpha)
        for alpha_spec in "0.6:ce_mse_alpha_06_seed_42" "0.8:ce_mse_alpha_08_seed_42"; do
            alpha="${alpha_spec%%:*}"
            run_name="${alpha_spec#*:}"
            output="realtime/experiments/$run_name"
            if [[ -f "$output/run_manifest.json" ]]; then
                echo "Completed run exists; skipping $run_name"
            else
                run_training ce_mse 42 50 "$run_name" "$alpha"
            fi
        done
        ;;
    train-catt-projected-cosine)
        run_name="ce_projected_cosine_beta_01_seed_42"
        output="realtime/experiments/$run_name"
        if [[ -f "$output/run_manifest.json" ]]; then
            echo "Completed run exists; skipping $run_name"
        else
            PYTHONHASHSEED=42 "$python_cmd" \
                proposed/train_student_projected_cosine.py \
                --seed 42 --epochs 50 --distill-weight 0.1 \
                --run-name "$run_name"
        fi
        ;;
    evaluate-catt-projected-cosine-ava)
        "$python_cmd" proposed/evaluate_student_checkpoint.py \
            --run-name ce_projected_cosine_beta_01_seed_42 --dataset ava
        ;;
    notify-test) echo "Telegram notification test completed." ;;
    smoke-catt-architectures) "$python_cmd" proposed/smoke_student_architectures.py ;;
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
    multi-selected)
        for seed in 42 123 2026; do
            selected_run="ce_mse_alpha_08_seed_${seed}"
            selected_output="realtime/experiments/$selected_run"
            if [[ -f "$selected_output/run_manifest.json" ]]; then
                echo "Completed run exists; skipping $selected_run"
            else
                run_training ce_mse "$seed" 50 "$selected_run" 0.8
            fi

            ablation_run="ce_only_seed_${seed}"
            ablation_output="realtime/experiments/$ablation_run"
            if [[ -f "$ablation_output/run_manifest.json" ]]; then
                echo "Completed run exists; skipping $ablation_run"
            else
                run_training ce_only "$seed" 50 "$ablation_run"
            fi
        done
        ;;
    evaluate-catt-multiseed)
        evaluate_multiseed_outputs
        ;;
    summarize-catt-multiseed)
        "$python_cmd" proposed/analyze_multiseed_results.py
        ;;
    post-multiseed)
        evaluate_multiseed_outputs
        "$python_cmd" proposed/analyze_multiseed_results.py
        ;;
    train-catt-remaining-ablations)
        train_remaining_ablations
        ;;
    evaluate-catt-remaining-ablations)
        evaluate_remaining_ablations
        ;;
    run-catt-remaining-ablations)
        run_remaining_ablations
        ;;
    run-catt-required-ablations)
        run_required_ablations
        ;;
    verify-catt-causality)
        verify_catt_causality
        ;;
    evaluate-catt-av-sync)
        evaluate_catt_av_sync
        ;;
    required-experiments)
        evaluate_multiseed_outputs
        "$python_cmd" proposed/analyze_multiseed_results.py
        "$python_cmd" proposed/smoke_student_architectures.py
        run_required_ablations
        verify_selected_catt_causality
        evaluate_catt_av_sync
        "$python_cmd" proposed/summarize_remaining_experiments.py \
            --allow-missing-noncausal
        ;;
    remaining-experiments)
        evaluate_multiseed_outputs
        "$python_cmd" proposed/analyze_multiseed_results.py
        "$python_cmd" proposed/smoke_student_architectures.py
        run_remaining_ablations
        verify_catt_causality
        evaluate_catt_av_sync
        "$python_cmd" proposed/summarize_remaining_experiments.py
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
  evaluate-lrasd-ava     Evaluate official LR-ASD weights on matched AVA 50-frame windows.
  evaluate-lrasd-columbia Evaluate official LR-ASD weights on matched Columbia 50-frame windows.
  evaluate-lrasd-native-ava      Optional native LR-ASD full-track evaluation on AVA.
  evaluate-lrasd-native-columbia Optional native LR-ASD full-track evaluation on Columbia.
  evaluate-catt-seed42 Evaluate retained CE+MSE and CE-only seed-42 checkpoints on both datasets.
  tune-catt-alpha     Train alpha=0.6 and 0.8 seed-42 candidates; select by AVA only.
  train-catt-projected-cosine Train the projected-cosine candidate on AVA only.
  evaluate-catt-projected-cosine-ava Evaluate its selected checkpoint on AVA only.
  notify-test         Test optional Telegram start/success notifications.
  smoke-catt-architectures Fast forward/backward test for all remaining model variants.
  smoke              One epoch, seed 42, CE+MSE; pipeline check only.
  single             Full 50-epoch proposed run, seed 42.
  single-ablation    Full 50-epoch CE-only run, seed 42.
  multi              Legacy alpha=0.4 paired-seed command; retained only for provenance.
  multi-selected     Train selected alpha=0.8 and CE-only for paired seeds 42, 123, 2026.
  evaluate-catt-multiseed Evaluate all selected and CE-only seed checkpoints on both datasets.
  summarize-catt-multiseed Aggregate completed runs and compute AVA-frozen Columbia F1.
  post-multiseed     Evaluate missing seed outputs, aggregate mean/std, and compute frozen-threshold F1.
  train-catt-remaining-ablations Train no-buffer, no-cross-speaker, and non-causal seed-42 variants.
  evaluate-catt-remaining-ablations Evaluate those three retrained variants on AVA and Columbia.
  run-catt-remaining-ablations Train and immediately evaluate each remaining variant in priority order.
  run-catt-required-ablations Train and evaluate only no-buffer and no-cross-speaker variants.
  verify-catt-causality Run future-input perturbation tests on causal and non-causal heads.
  evaluate-catt-av-sync Evaluate selected seeds under causal audio/video lag perturbations.
  required-experiments Complete the time-critical reviewer requirements and skip non-causal training.
  remaining-experiments Resume and complete all post-multi-seed CATT-ASD experiments above.

Optional:
  --notify           Read .env and send Telegram start/success/failure messages.

Environment:
  MSSG_GRAPH_NJ      Stage-9 graph workers (default: 6).
  MSSG_GRAPH_SPLITS  Stage-9 splits to build (default: "train val").
EOF
        ;;
esac
