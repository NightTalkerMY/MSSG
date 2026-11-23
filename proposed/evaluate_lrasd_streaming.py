"""Evaluate official LR-ASD weights under the 50-frame final-position protocol."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from evaluate_lrasd_native import load_track, sha256


WINDOW_LENGTH = 50
STRIDE = 25
SPEAKERS = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")


def best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(values))
    threshold = float(thresholds[index]) if index < len(thresholds) else 1.0
    return float(values[index] * 100.0), threshold


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ava", "columbia"), required=True)
    parser.add_argument("--lrasd-root", type=Path, default=root.parent / "LR-ASD")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(), root


def resolve_inputs(root, dataset):
    if dataset == "ava":
        data_root = root / "data" / "AVA"
        return (
            data_root / "csv" / "val_loader.csv",
            data_root / "clips_audios" / "val",
            data_root / "clips_videos" / "val",
        )
    data_root = root / "data" / "Columbia"
    return (
        data_root / "csv" / "train_loader.csv",
        data_root / "clips_audios" / "train",
        data_root / "clips_videos" / "train",
    )


def main():
    args, root = parse_args()
    lrasd_root = args.lrasd_root.resolve()
    checkpoint = (
        args.checkpoint.resolve() if args.checkpoint
        else lrasd_root / "weight" / "pretrain_AVA.model"
    )
    if not (lrasd_root / "ASD.py").is_file() or not checkpoint.is_file():
        raise SystemExit(
            f"Missing official LR-ASD checkout or checkpoint: {lrasd_root}, {checkpoint}"
        )
    if args.output_dir is None:
        args.output_dir = (
            root / "realtime" / "baseline_runs" / f"lrasd_{args.dataset}_streaming"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")

    loader_file, audio_root, visual_root = resolve_inputs(root, args.dataset)
    for required in (loader_file, audio_root, visual_root):
        if not required.exists():
            raise SystemExit(f"Missing LR-ASD input: {required}")

    commit = subprocess.run(
        ["git", "-C", str(lrasd_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    sys.path.insert(0, str(lrasd_root))
    previous_cwd = Path.cwd()
    os.chdir(lrasd_root)
    try:
        # Imported here so LR-ASD's own ASD.py takes precedence over MSSG's.
        from ASD import ASD  # pylint: disable=import-outside-toplevel
        model_wrapper = ASD()
        model_wrapper.loadParameters(str(checkpoint))
    finally:
        os.chdir(previous_cwd)
    model_wrapper.eval()
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    elapsed = []
    short_tracks = 0
    lines = [
        line for line in loader_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with torch.no_grad():
        for line in tqdm(lines, desc=f"LR-ASD {args.dataset} 50-frame"):
            track = load_track(line, audio_root, visual_root, args.dataset)
            usable = len(track["labels"])
            if usable < WINDOW_LENGTH:
                short_tracks += 1
                continue
            speaker = (
                track["entity"].split("_")[0].capitalize()
                if args.dataset == "columbia" else ""
            )
            for start in range(0, usable - WINDOW_LENGTH + 1, STRIDE):
                end = start + WINDOW_LENGTH
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                audio = track["audio"][:, start * 4:end * 4].to(device)
                visual = track["visual"][:, start:end].to(device)
                audio_embed = model_wrapper.model.forward_audio_frontend(audio)
                visual_embed = model_wrapper.model.forward_visual_frontend(visual)
                fused = model_wrapper.model.forward_audio_visual_backend(
                    audio_embed, visual_embed
                )
                logits = model_wrapper.lossAV.FC(fused.squeeze(1))
                score = float(F.softmax(logits, dim=-1)[-1, 1].item())
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - started)

                final_index = end - 1
                label = int(track["labels"][final_index])
                prediction_rows.append([
                    track["video"], track["entity"],
                    track["timestamps"][final_index], start, final_index,
                    speaker, 0 if label == 2 else label, score,
                ])

    if not prediction_rows:
        raise RuntimeError("LR-ASD evaluation produced no 50-frame predictions")
    labels = np.asarray([row[6] for row in prediction_rows], dtype=np.int64)
    scores = np.asarray([row[7] for row in prediction_rows], dtype=np.float64)
    result = {
        "schema_version": 1,
        "method": "LR-ASD",
        "dataset": args.dataset,
        "protocol": "architecture-native inputs; 50-frame windows with final-position predictions only",
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "evaluated_position": WINDOW_LENGTH - 1,
        "official_repository": "https://github.com/Junhua-Liao/LR-ASD",
        "repository_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "loader_file": str(loader_file.resolve()),
        "tracks_scanned": len(lines),
        "tracks_shorter_than_window": short_tracks,
        "predictions": len(prediction_rows),
        "inference_parameters": (
            sum(parameter.numel() for parameter in model_wrapper.model.parameters())
            + sum(parameter.numel() for parameter in model_wrapper.lossAV.FC.parameters())
        ),
        "device": str(device),
        "latency_scope": "LR-ASD audio frontend, visual frontend, temporal backend, and classifier; includes input transfer",
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
            mask = np.asarray([row[5] == speaker for row in prediction_rows])
            if not mask.any():
                raise RuntimeError(f"No LR-ASD predictions for Columbia speaker {speaker}")
            f1, threshold = best_f1(labels[mask], scores[mask])
            per_speaker[speaker] = {
                "predictions": int(mask.sum()),
                "average_precision_percent": float(
                    average_precision_score(labels[mask], scores[mask]) * 100.0
                ),
                "best_f1_percent": f1,
                "best_threshold": threshold,
            }
        result["columbia_per_speaker"] = per_speaker
        result["columbia_six_speaker_macro_best_f1_percent"] = float(
            np.mean([item["best_f1_percent"] for item in per_speaker.values()])
        )

    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "video_id", "entity_id", "timestamp", "window_start_index",
            "final_frame_index", "speaker", "label", "score",
        ])
        writer.writerows(prediction_rows)
    result["prediction_file"] = str(prediction_path.resolve())
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
