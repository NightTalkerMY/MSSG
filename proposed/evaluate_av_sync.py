"""Evaluate CATT-ASD sensitivity to causal audio/video lag perturbations."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proposed"))

from student_model import CausalStudentASD  # noqa: E402


OFFSETS = (-10, -5, -2, -1, 0, 1, 2, 5, 10)
SPEAKERS = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")
LAST_FRAME = 49


def cache_files(cache_dir, dataset):
    prefix = "val" if dataset == "ava" else "col"
    files = sorted(cache_dir.glob(f"{prefix}_cache_part_*.pt"))
    if not files:
        raise SystemExit(f"No {prefix} cache files found in {cache_dir}")
    return files


def lag(tensor, frames):
    if frames <= 0:
        return tensor
    shifted = torch.zeros_like(tensor)
    shifted[:, frames:] = tensor[:, :-frames]
    return shifted


def perturb(visual, audio, spatial, offset):
    if offset > 0:
        return visual, lag(audio, offset), spatial
    if offset < 0:
        amount = -offset
        return lag(visual, amount), audio, lag(spatial, amount)
    return visual, audio, spatial


def best_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2.0 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(values))
    return float(thresholds[index]) if index < len(thresholds) else 1.0


def best_f1(labels, scores):
    threshold = best_threshold(labels, scores)
    predicted = np.asarray(scores) >= threshold
    return float(f1_score(labels, predicted, zero_division=0) * 100.0)


def evaluate(model, files, dataset, device):
    rows = {offset: [] for offset in OFFSETS}
    window_index = 0
    with torch.no_grad():
        for path in files:
            data = torch.load(path, map_location="cpu")
            count = data["l"].shape[0]
            for index in tqdm(range(count), desc=f"AV sync {dataset} {path.name}"):
                visual, audio = data["v"][index], data["a"][index]
                spatial, labels, mask = (
                    data["s"][index], data["l"][index], data["m"][index]
                )
                names = data["names"][index] if dataset == "columbia" else None
                active = [
                    slot for slot in range(mask.shape[0])
                    if mask[slot, LAST_FRAME] > 0.5
                    and (names is None or names[slot] != "Ghost")
                ]
                if not active:
                    window_index += 1
                    continue
                for offset in OFFSETS:
                    shifted_v, shifted_a, shifted_s = perturb(
                        visual, audio, spatial, offset
                    )
                    logits, _ = model(
                        shifted_v.unsqueeze(0).to(device),
                        shifted_a.unsqueeze(0).to(device),
                        shifted_s.unsqueeze(0).to(device),
                        mask.unsqueeze(0).to(device),
                    )
                    scores = F.softmax(
                        logits[0, :, LAST_FRAME], dim=-1
                    )[:, 1].cpu()
                    for slot in active:
                        label = int(labels[slot, LAST_FRAME].item())
                        rows[offset].append({
                            "dataset": dataset,
                            "offset_frames": offset,
                            "window_index": window_index,
                            "slot": slot,
                            "speaker": "" if names is None else names[slot],
                            "label": 0 if label == 2 else label,
                            "score": float(scores[slot].item()),
                        })
                window_index += 1
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = ROOT / "realtime" / "experiments" / args.run_name
    manifest_path, checkpoint = run_dir / "run_manifest.json", run_dir / "best_ava.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise SystemExit(f"Missing retained run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    architecture = manifest.get("architecture", "full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalStudentASD(architecture=architecture).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ava_rows = evaluate(
        model,
        cache_files(ROOT / "realtime" / "val_cached_chunks", "ava"),
        "ava", device,
    )
    columbia_rows = evaluate(
        model,
        cache_files(ROOT / "realtime" / "columbia_cached_chunks", "columbia"),
        "columbia", device,
    )
    base_labels = [row["label"] for row in ava_rows[0]]
    base_scores = [row["score"] for row in ava_rows[0]]
    frozen_threshold = best_threshold(base_labels, base_scores)

    results = []
    for offset in OFFSETS:
        ava_labels = [row["label"] for row in ava_rows[offset]]
        ava_scores = [row["score"] for row in ava_rows[offset]]
        per_speaker_best, per_speaker_fixed = {}, {}
        for speaker in SPEAKERS:
            selected = [
                row for row in columbia_rows[offset] if row["speaker"] == speaker
            ]
            labels = [row["label"] for row in selected]
            scores = [row["score"] for row in selected]
            per_speaker_best[speaker] = best_f1(labels, scores)
            predicted = np.asarray(scores) >= frozen_threshold
            per_speaker_fixed[speaker] = float(
                f1_score(labels, predicted, zero_division=0) * 100.0
            )
        results.append({
            "offset_frames": offset,
            "offset_milliseconds_at_25_fps": offset * 40,
            "perturbation": (
                "video_lag" if offset < 0 else
                "audio_lag" if offset > 0 else "aligned"
            ),
            "ava_average_precision_percent": float(
                average_precision_score(ava_labels, ava_scores) * 100.0
            ),
            "columbia_macro_best_f1_percent": float(
                np.mean(list(per_speaker_best.values()))
            ),
            "columbia_macro_ava_frozen_threshold_f1_percent": float(
                np.mean(list(per_speaker_fixed.values()))
            ),
            "columbia_per_speaker_best_f1_percent": per_speaker_best,
            "columbia_per_speaker_ava_frozen_threshold_f1_percent": per_speaker_fixed,
        })

    output_dir = args.output_dir or (
        ROOT / "realtime" / "analysis" / "av_sync" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        row for dataset_rows in (ava_rows, columbia_rows)
        for offset in OFFSETS for row in dataset_rows[offset]
    ]
    with gzip.open(output_dir / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "seed": manifest["seed"],
        "architecture": architecture,
        "offset_definition": (
            "negative values delay video/spatial features; positive values delay audio; "
            "only past features are used and vacated positions are zero-filled"
        ),
        "offsets_frames": list(OFFSETS),
        "ava_calibrated_threshold_from_aligned_predictions": frozen_threshold,
        "results": results,
        "prediction_file": str((output_dir / "predictions.csv.gz").resolve()),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
