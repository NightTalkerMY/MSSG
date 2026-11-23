"""Run the official pretrained LR-ASD model on local AVA or Columbia tracks.

The model and checkpoint come from Junhua-Liao/LR-ASD.  This evaluator uses
LR-ASD's native full-track audiovisual inference, not CATT-ASD's cached feature
head protocol.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_track_path(root, video, entity, suffix):
    candidates = (
        root / video / f"{entity}{suffix}",
        root / video / f"{entity.replace(':', '_')}{suffix}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing track input; tried: {candidates}")


def load_track(line, audio_root, visual_root, dataset):
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 4:
        raise RuntimeError(f"Malformed loader row: {line!r}")
    entity, declared_frames, fps, labels_text = fields[:4]
    video = entity[:11] if dataset == "ava" else "col"
    audio_path = resolve_track_path(audio_root, video, entity, ".wav")
    visual_folder = resolve_track_path(visual_root, video, entity, "")
    sample_rate, audio = wavfile.read(audio_path)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16 kHz audio: {audio_path} has {sample_rate}")
    fps_value = float(fps)
    audio_features = python_speech_features.mfcc(
        audio, 16000, numcep=13,
        winlen=0.025 * 25 / fps_value,
        winstep=0.010 * 25 / fps_value,
    )
    frames = sorted(
        visual_folder.glob("*.jpg"), key=lambda path: float(path.stem)
    )
    declared = int(declared_frames)
    labels = np.asarray(ast.literal_eval(labels_text), dtype=np.int64)
    usable = min(declared, len(frames), len(labels), len(audio_features) // 4)
    if usable <= 0:
        raise RuntimeError(f"Track has no usable aligned frames: {entity}")
    visuals = []
    timestamps = []
    for path in frames[:usable]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not decode {path}")
        visuals.append(cv2.resize(image, (112, 112)))
        timestamps.append(path.stem)
    return {
        "entity": entity,
        "video": video,
        "audio": torch.tensor(audio_features[: usable * 4], dtype=torch.float32).unsqueeze(0),
        "visual": torch.tensor(np.asarray(visuals), dtype=torch.float32).unsqueeze(0),
        "labels": labels[:usable],
        "timestamps": timestamps,
    }


def best_f1(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(values) * 100.0)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ava", "columbia"), required=True)
    parser.add_argument("--lrasd-root", type=Path, default=root.parent / "LR-ASD")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(), root


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
        args.output_dir = root / "realtime" / "baseline_runs" / f"lrasd_{args.dataset}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ava":
        data_root = root / "data" / "AVA"
        loader_file = data_root / "csv" / "val_loader.csv"
        audio_root = data_root / "clips_audios" / "val"
        visual_root = data_root / "clips_videos" / "val"
    else:
        data_root = root / "data" / "Columbia"
        loader_file = data_root / "csv" / "train_loader.csv"
        audio_root = data_root / "clips_audios" / "train"
        visual_root = data_root / "clips_videos" / "train"
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
        from ASD import ASD  # pylint: disable=import-outside-toplevel
        model_wrapper = ASD()
        model_wrapper.loadParameters(str(checkpoint))
    finally:
        os.chdir(previous_cwd)
    model_wrapper.eval()
    device = torch.device("cuda")

    prediction_rows = []
    lines = loader_file.read_text(encoding="utf-8").splitlines()
    with torch.no_grad():
        for line in tqdm(lines, desc=f"LR-ASD {args.dataset}"):
            track = load_track(line, audio_root, visual_root, args.dataset)
            audio_embed = model_wrapper.model.forward_audio_frontend(track["audio"].to(device))
            visual_embed = model_wrapper.model.forward_visual_frontend(track["visual"].to(device))
            fused = model_wrapper.model.forward_audio_visual_backend(audio_embed, visual_embed)
            logits = model_wrapper.lossAV.FC(fused.squeeze(1))
            scores = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            usable = min(len(scores), len(track["labels"]), len(track["timestamps"]))
            speaker = track["entity"].split("_")[0].capitalize() if args.dataset == "columbia" else ""
            for index in range(usable):
                label = int(track["labels"][index])
                if label == 2:
                    label = 0
                prediction_rows.append([
                    track["video"], track["entity"], track["timestamps"][index],
                    index, speaker, label, float(scores[index]),
                ])

    labels = np.asarray([row[5] for row in prediction_rows], dtype=np.int64)
    scores = np.asarray([row[6] for row in prediction_rows], dtype=np.float64)
    result = {
        "schema_version": 1,
        "method": "LR-ASD",
        "dataset": args.dataset,
        "protocol": "official pretrained LR-ASD model with native full-track inference",
        "official_repository": "https://github.com/Junhua-Liao/LR-ASD",
        "repository_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "loader_file": str(loader_file.resolve()),
        "prediction_frames": len(prediction_rows),
        "parameters": sum(parameter.numel() for parameter in model_wrapper.parameters()),
        "device": str(device),
    }
    if args.dataset == "ava":
        result["ava_sklearn_average_precision_percent"] = float(
            average_precision_score(labels, scores) * 100.0
        )
    else:
        expected = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")
        per_speaker = {}
        for speaker in expected:
            mask = np.asarray([row[4] == speaker for row in prediction_rows])
            if not mask.any():
                raise RuntimeError(f"No LR-ASD predictions for Columbia speaker {speaker}")
            per_speaker[speaker] = {
                "samples": int(mask.sum()),
                "average_precision_percent": float(
                    average_precision_score(labels[mask], scores[mask]) * 100.0
                ),
                "best_f1_percent": best_f1(labels[mask], scores[mask]),
            }
        result["columbia_per_speaker"] = per_speaker
        result["columbia_six_speaker_macro_best_f1_percent"] = float(
            np.mean([item["best_f1_percent"] for item in per_speaker.values()])
        )

    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "video_id", "entity_id", "timestamp", "frame_index",
            "speaker", "label", "score",
        ])
        writer.writerows(prediction_rows)
    result["predictions"] = str(prediction_path.resolve())
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
