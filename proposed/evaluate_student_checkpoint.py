"""Evaluate one retained CATT-ASD checkpoint on AVA or Columbia."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proposed"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from student_model import CausalStudentASD


LAST_FRAME = 49
SPEAKERS = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(values))
    threshold = float(thresholds[index]) if index < len(thresholds) else 1.0
    return float(values[index] * 100.0), threshold


def cache_files(cache_dir, dataset):
    prefix = "val" if dataset == "ava" else "col"
    files = sorted(cache_dir.glob(f"{prefix}_cache_part_*.pt"))
    if not files:
        raise SystemExit(f"No {prefix} cache files found in {cache_dir}")
    return files


def iter_cache(files, dataset):
    for path in files:
        data = torch.load(path, map_location="cpu")
        required = {"v", "a", "s", "l", "m"}
        if dataset == "columbia":
            required.add("names")
        missing = required.difference(data)
        if missing:
            raise RuntimeError(f"{path} is missing keys: {sorted(missing)}")
        for index in range(data["l"].shape[0]):
            names = data["names"][index] if dataset == "columbia" else None
            yield (
                data["v"][index], data["a"][index], data["s"][index],
                data["l"][index], data["m"][index], names,
            )


def warmup(model, device):
    visual = torch.zeros(1, 5, 50, 896, device=device)
    audio = torch.zeros(1, 5, 50, 128, device=device)
    spatial = torch.zeros(1, 5, 50, 4, device=device)
    mask = torch.ones(1, 5, 50, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(visual, audio, spatial, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=("ava", "columbia"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = ROOT / "realtime" / "experiments" / args.run_name
    checkpoint = args.checkpoint or run_dir / "best_ava.pt"
    manifest_path = run_dir / "run_manifest.json"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing selected CATT-ASD checkpoint: {checkpoint}")
    if not manifest_path.is_file():
        raise SystemExit(f"Missing training manifest: {manifest_path}")
    training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("run_name") != args.run_name:
        raise RuntimeError("Training manifest run name does not match --run-name")

    if args.cache_dir is None:
        cache_name = "val_cached_chunks" if args.dataset == "ava" else "columbia_cached_chunks"
        args.cache_dir = ROOT / "realtime" / cache_name
    if args.output_dir is None:
        args.output_dir = ROOT / "realtime" / "catt_results" / args.run_name / args.dataset
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    files = cache_files(args.cache_dir, args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = training_manifest.get("architecture", "full")
    model = CausalStudentASD(architecture=architecture).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    warmup(model, device)

    predictions = []
    elapsed = []
    windows = 0
    with torch.no_grad():
        iterator = iter_cache(files, args.dataset)
        for visual, audio, spatial, labels, mask, names in tqdm(
            iterator, desc=f"{args.run_name} {args.dataset}"
        ):
            active = [
                index for index in range(mask.shape[0])
                if mask[index, LAST_FRAME] > 0.5
            ]
            if names is not None:
                active = [index for index in active if names[index] != "Ghost"]
            windows += 1
            if not active:
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits, _features = model(
                visual.unsqueeze(0).to(device),
                audio.unsqueeze(0).to(device),
                spatial.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )
            probabilities = F.softmax(
                logits[0, :, LAST_FRAME, :], dim=-1
            )[:, 1].detach().cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)

            for slot in active:
                label = int(labels[slot, LAST_FRAME].item())
                predictions.append({
                    "window_index": windows - 1,
                    "slot": slot,
                    "speaker": names[slot] if names is not None else "",
                    "label": 0 if label == 2 else label,
                    "score": float(probabilities[slot].item()),
                })

    if not predictions:
        raise RuntimeError("CATT-ASD evaluation produced no final-frame predictions")
    labels = [row["label"] for row in predictions]
    scores = [row["score"] for row in predictions]
    result = {
        "schema_version": 1,
        "method": "CATT-ASD",
        "run_name": args.run_name,
        "variant": training_manifest["variant"],
        "architecture": architecture,
        "seed": training_manifest["seed"],
        "dataset": args.dataset,
        "protocol": "audited 50-frame cache; final-position predictions only",
        "window_length": 50,
        "stride": 25,
        "evaluated_position": LAST_FRAME,
        "cache_directory": str(args.cache_dir.resolve()),
        "cache_files": len(files),
        "windows_scanned": windows,
        "predictions": len(predictions),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_selection": training_manifest["checkpoint_selection"],
        "best_training_epoch": training_manifest["best_epoch"],
        "columbia_used_during_training_or_selection": training_manifest[
            "columbia_used_during_training_or_selection"
        ],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "latency_scope": "CATT-ASD prediction head with cached inputs; includes input transfer",
        "warmup_windows": 10,
        "latency_mean_ms_per_window": float(np.mean(elapsed) * 1000.0),
        "latency_median_ms_per_window": float(np.median(elapsed) * 1000.0),
    }
    if args.dataset == "ava":
        result["ava_average_precision_percent"] = float(
            average_precision_score(labels, scores) * 100.0
        )
    else:
        per_speaker = {}
        for speaker in SPEAKERS:
            selected = [row for row in predictions if row["speaker"] == speaker]
            if not selected:
                raise RuntimeError(f"No CATT-ASD predictions for Columbia speaker {speaker}")
            f1, threshold = best_f1(
                [row["label"] for row in selected],
                [row["score"] for row in selected],
            )
            per_speaker[speaker] = {
                "predictions": len(selected),
                "best_f1_percent": f1,
                "best_threshold": threshold,
            }
        result["columbia_per_speaker"] = per_speaker
        result["columbia_six_speaker_macro_best_f1_percent"] = float(
            np.mean([item["best_f1_percent"] for item in per_speaker.values()])
        )

    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("window_index", "slot", "speaker", "label", "score"),
        )
        writer.writeheader()
        writer.writerows(predictions)
    result["prediction_file"] = str(prediction_path.resolve())
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
