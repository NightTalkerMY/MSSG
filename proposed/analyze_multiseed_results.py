"""Aggregate paired CATT-ASD runs and evaluate frozen Columbia thresholds.

For every method/run, the operating threshold is selected from its retained
AVA predictions and then applied unchanged to every Columbia speaker.  This is
reported alongside the existing per-speaker best-F1 metric; it does not replace
or rewrite any retained evaluation result.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "realtime" / "analysis" / "catt_multiseed"
SEEDS = (42, 123, 2026)
SPEAKERS = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")


def read_json(path):
    if not path.is_file():
        raise SystemExit(f"Missing evaluation result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path):
    if not path.is_file():
        raise SystemExit(f"Missing prediction file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"label", "score"}.issubset(rows[0]):
        raise RuntimeError(f"Prediction schema is invalid: {path}")
    return rows


def best_threshold(rows):
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2.0 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(values))
    return float(thresholds[index]) if index < len(thresholds) else 1.0


def fixed_columbia_f1(rows, threshold):
    per_speaker = {}
    for speaker in SPEAKERS:
        selected = [row for row in rows if row.get("speaker") == speaker]
        if not selected:
            raise RuntimeError(f"No Columbia predictions for {speaker}")
        labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
        predicted = np.asarray(
            [float(row["score"]) >= threshold for row in selected], dtype=np.int64
        )
        per_speaker[speaker] = float(
            f1_score(labels, predicted, zero_division=0) * 100.0
        )
    return per_speaker, float(np.mean(list(per_speaker.values())))


def evaluate_pair(name, ava_dir, columbia_dir):
    ava_result = read_json(ava_dir / "result.json")
    columbia_result = read_json(columbia_dir / "result.json")
    ava_rows = read_predictions(ava_dir / "predictions.csv")
    columbia_rows = read_predictions(columbia_dir / "predictions.csv")
    threshold = best_threshold(ava_rows)
    fixed_per_speaker, fixed_macro = fixed_columbia_f1(columbia_rows, threshold)
    return {
        "name": name,
        "ava_average_precision_percent": ava_result["ava_average_precision_percent"],
        "ava_calibrated_threshold": threshold,
        "columbia_macro_best_f1_percent": columbia_result[
            "columbia_six_speaker_macro_best_f1_percent"
        ],
        "columbia_per_speaker_best_f1_percent": {
            speaker: columbia_result["columbia_per_speaker"][speaker][
                "best_f1_percent"
            ]
            for speaker in SPEAKERS
        },
        "columbia_fixed_threshold_per_speaker_f1_percent": fixed_per_speaker,
        "columbia_fixed_threshold_macro_f1_percent": fixed_macro,
        "latency_mean_ms_per_window_ava": ava_result.get(
            "latency_mean_ms_per_window"
        ),
        "latency_mean_ms_per_window_columbia": columbia_result.get(
            "latency_mean_ms_per_window"
        ),
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)),
    }


def summarize_group(records):
    summary = {
        "ava_average_precision_percent": summarize(
            [record["ava_average_precision_percent"] for record in records]
        ),
        "columbia_macro_best_f1_percent": summarize(
            [record["columbia_macro_best_f1_percent"] for record in records]
        ),
        "columbia_fixed_threshold_macro_f1_percent": summarize(
            [record["columbia_fixed_threshold_macro_f1_percent"] for record in records]
        ),
        "columbia_fixed_threshold_per_speaker_f1_percent": {},
        "columbia_per_speaker_best_f1_percent": {},
    }
    for speaker in SPEAKERS:
        summary["columbia_per_speaker_best_f1_percent"][speaker] = summarize(
            [
                record["columbia_per_speaker_best_f1_percent"][speaker]
                for record in records
            ]
        )
        summary["columbia_fixed_threshold_per_speaker_f1_percent"][speaker] = summarize(
            [
                record["columbia_fixed_threshold_per_speaker_f1_percent"][speaker]
                for record in records
            ]
        )
    return summary


def main():
    records = {"selected_alpha_08": [], "ce_only": []}
    for seed in SEEDS:
        for group, run_name in (
            ("selected_alpha_08", f"ce_mse_alpha_08_seed_{seed}"),
            ("ce_only", f"ce_only_seed_{seed}"),
        ):
            base = ROOT / "realtime" / "catt_results" / run_name
            record = evaluate_pair(run_name, base / "ava", base / "columbia")
            record["seed"] = seed
            records[group].append(record)

    baselines = {}
    for method in ("mssg", "lrasd"):
        base = ROOT / "realtime" / "baseline_runs"
        baselines[method] = evaluate_pair(
            method.upper() if method == "mssg" else "LR-ASD",
            base / f"{method}_ava_streaming",
            base / f"{method}_columbia_streaming",
        )

    output = {
        "schema_version": 1,
        "threshold_protocol": (
            "Each run/method selects one threshold from its AVA predictions; "
            "that threshold is frozen and applied unchanged to all six Columbia speakers."
        ),
        "seeds": list(SEEDS),
        "runs": records,
        "aggregates": {
            group: summarize_group(group_records)
            for group, group_records in records.items()
        },
        "baselines": baselines,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
