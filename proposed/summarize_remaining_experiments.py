"""Create one manifest for all post-multi-seed CATT-ASD experiments."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
ARCHITECTURES = ("no_buffer", "no_cross_speaker", "noncausal")


def load(path):
    if not path.is_file():
        raise SystemExit(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else None,
    }


def main():
    analysis_root = ROOT / "realtime" / "analysis"
    multiseed = load(analysis_root / "catt_multiseed" / "summary.json")

    ablations = {}
    for architecture in ARCHITECTURES:
        run_name = f"ce_mse_alpha_08_{architecture}_seed_42"
        run_root = ROOT / "realtime" / "catt_results" / run_name
        ava = load(run_root / "ava" / "result.json")
        columbia = load(run_root / "columbia" / "result.json")
        ablations[architecture] = {
            "run_name": run_name,
            "ava_average_precision_percent": ava["ava_average_precision_percent"],
            "columbia_macro_best_f1_percent": columbia[
                "columbia_six_speaker_macro_best_f1_percent"
            ],
            "parameters": ava["parameters"],
            "latency_mean_ms_per_window_ava": ava["latency_mean_ms_per_window"],
            "latency_mean_ms_per_window_columbia": columbia[
                "latency_mean_ms_per_window"
            ],
            "columbia_per_speaker": columbia["columbia_per_speaker"],
        }

    causality = {}
    for seed in SEEDS:
        run_name = f"ce_mse_alpha_08_seed_{seed}"
        causality[run_name] = load(
            analysis_root / "causality" / f"{run_name}.json"
        )
    noncausal_name = "ce_mse_alpha_08_noncausal_seed_42"
    causality[noncausal_name] = load(
        analysis_root / "causality" / f"{noncausal_name}.json"
    )

    sync_runs = [
        load(
            analysis_root / "av_sync" / f"ce_mse_alpha_08_seed_{seed}" /
            "result.json"
        )
        for seed in SEEDS
    ]
    sync_by_offset = {}
    for index, offset in enumerate(sync_runs[0]["offsets_frames"]):
        rows = [run["results"][index] for run in sync_runs]
        if any(row["offset_frames"] != offset for row in rows):
            raise RuntimeError("AV-sync result offsets are inconsistent across seeds")
        sync_by_offset[str(offset)] = {
            "perturbation": rows[0]["perturbation"],
            "ava_average_precision_percent": stats(
                [row["ava_average_precision_percent"] for row in rows]
            ),
            "columbia_macro_best_f1_percent": stats(
                [row["columbia_macro_best_f1_percent"] for row in rows]
            ),
            "columbia_macro_ava_frozen_threshold_f1_percent": stats(
                [
                    row["columbia_macro_ava_frozen_threshold_f1_percent"]
                    for row in rows
                ]
            ),
        }

    output = {
        "schema_version": 1,
        "multiseed": multiseed,
        "retrained_seed_42_architecture_comparisons": ablations,
        "future_input_perturbation": causality,
        "audio_video_sync_three_seed_aggregate": sync_by_offset,
        "scope": {
            "training_and_checkpoint_selection": "AVA only",
            "columbia_role": "held-out cross-dataset evaluation",
            "causality_test": "prediction head with respect to supplied features",
            "audio_video_offset_unit": "frames at 25 fps",
        },
    }
    output_path = analysis_root / "catt_remaining_experiments_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved complete summary to {output_path}")


if __name__ == "__main__":
    main()
