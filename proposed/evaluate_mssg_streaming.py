"""Evaluate the retained MSSG teacher on CATT-ASD's 50-frame protocol.

Unlike ``evaluate_mssg_native.py``, this evaluator does not require native
2000-node graph files.  It reconstructs the teacher's seven BiGRU features and
same-frame speaker graph from the audited student caches, then scores only the
final position of each 50-frame window.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.data import Data
from tqdm import tqdm

from ASD import ASD
from model.models_gnn import MSSG


LAST_FRAME = 49
FRAME_WIDTH = 1280
ENCODER_CONFIGS = (
    ("face", "face_model", 0, 128, 0),
    ("face_body", "body_model", 256, 384, 128),
    ("face_large", "face_large_model", 384, 512, 256),
    ("background", "background_model.model", 512, 640, 384),
    ("face_small", "face_small_model", 640, 768, 512),
    ("face_down", "face_down_model", 768, 896, 640),
    ("face_body_large", "body_large_model", 128, 256, 768),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.argmax(values))
    threshold = float(thresholds[index]) if index < len(thresholds) else 1.0
    return float(values[index] * 100.0), threshold


def load_teacher(pretrain_dir, checkpoint, device):
    bigrus = {}
    encoder_sources = {}
    for name, relative_path, _start, _end, _mssg_start in ENCODER_CONFIGS:
        source = pretrain_dir / relative_path
        if not source.exists():
            raise SystemExit(f"Missing retained encoder weights: {source}")
        wrapper = ASD(encoder_struct=[32, 64, 128])
        if source.is_dir():
            wrapper.loadParameters_multi(str(source))
            encoder_sources[name] = sorted(
                (item.name for item in source.iterdir() if item.is_file()),
                reverse=True,
            )[:7]
        else:
            wrapper.loadParameters(str(source))
            encoder_sources[name] = [source.name]
        bigrus[name] = wrapper.model.GRU.to(device).eval()

    model = MSSG().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    model.eval()
    return bigrus, model, encoder_sources


@torch.no_grad()
def score_window(v, audio, spatial, labels, mask, names, bigrus, model, device):
    active = [index for index in range(mask.shape[0]) if mask[index, LAST_FRAME] > 0.5]
    if names is not None:
        active = [index for index in active if names[index] != "Ghost"]
    if not active:
        return [], 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    v = v.to(device)
    audio = audio.to(device)
    spatial = spatial.to(device)
    mask = mask.to(device)

    valid = [index for index in range(mask.shape[0]) if mask[index].sum() > 0]
    valid_lookup = {speaker: index for index, speaker in enumerate(valid)}
    features = torch.zeros(len(valid), v.shape[1], 896, device=device)
    for name, _path, cache_start, cache_end, mssg_start in ENCODER_CONFIGS:
        visual = v[valid, :, cache_start:cache_end]
        fused = visual + audio[valid]
        features[:, :, mssg_start:mssg_start + 128] = bigrus[name](fused)

    node_features = []
    for speaker in active:
        feature = features[valid_lookup[speaker], LAST_FRAME]
        location = spatial[speaker, LAST_FRAME]
        face_size = (location[2] * FRAME_WIDTH).reshape(1)
        speaker_count = torch.tensor([float(len(active))], device=device)
        node_features.append(torch.cat((feature, face_size, location, speaker_count)))

    source, target, attributes = [], [], []
    for left in range(len(active)):
        for right in range(len(active)):
            source.append(left)
            target.append(right)
            attributes.append(111 if left == right else 0)
    graph = Data(
        x=torch.stack(node_features),
        edge_index=torch.tensor([source, target], dtype=torch.long, device=device),
        edge_attr=torch.tensor(attributes, dtype=torch.float32, device=device),
    )
    probabilities = F.softmax(model(graph), dim=-1)[:, 1].detach().cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    rows = []
    for node_index, (speaker, score) in enumerate(zip(active, probabilities)):
        label = int(labels[speaker, LAST_FRAME].item())
        rows.append({
            "node_index": node_index,
            "slot": speaker,
            "speaker": names[speaker] if names is not None else "",
            "label": 0 if label == 2 else label,
            "score": float(score),
        })
    return rows, elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ava", "columbia"), required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pretrain-dir", type=Path, default=ROOT / "predata" / "pretrain_model")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "exps" / "graph_training" / "graph_2000" / "chckpoint_best.pt",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache_dir is None:
        name = "val_cached_chunks" if args.dataset == "ava" else "columbia_cached_chunks"
        args.cache_dir = ROOT / "realtime" / name
    if args.output_dir is None:
        args.output_dir = ROOT / "realtime" / "baseline_runs" / f"mssg_{args.dataset}_streaming"
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing MSSG checkpoint: {args.checkpoint}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    files = cache_files(args.cache_dir, args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bigrus, model, encoder_sources = load_teacher(
        args.pretrain_dir, args.checkpoint, device
    )
    predictions = []
    elapsed = []
    window_index = 0
    iterator = iter_cache(files, args.dataset)
    for v, audio, spatial, labels, mask, names in tqdm(iterator, desc=f"MSSG {args.dataset} streaming"):
        rows, duration = score_window(
            v, audio, spatial, labels, mask, names, bigrus, model, device
        )
        for row in rows:
            row["window_index"] = window_index
            predictions.append(row)
        if rows:
            elapsed.append(duration)
        window_index += 1
    if not predictions:
        raise RuntimeError("Evaluation produced no final-frame predictions")

    labels = [row["label"] for row in predictions]
    scores = [row["score"] for row in predictions]
    result = {
        "schema_version": 1,
        "method": "MSSG",
        "dataset": args.dataset,
        "protocol": "50-frame causal-window simulation; final-position predictions only",
        "window_length": 50,
        "evaluated_position": 49,
        "cache_directory": str(args.cache_dir.resolve()),
        "cache_files": len(files),
        "windows_scanned": window_index,
        "predictions": len(predictions),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "encoder_sources": encoder_sources,
        "device": str(device),
        "mssg_graph_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seven_bigru_parameters": sum(
            sum(parameter.numel() for parameter in bigru.parameters())
            for bigru in bigrus.values()
        ),
        "latency_scope": "seven retained BiGRUs, final-frame graph assembly, and MSSG inference; includes input transfer",
        "latency_mean_ms_per_window": float(np.mean(elapsed) * 1000.0),
        "latency_median_ms_per_window": float(np.median(elapsed) * 1000.0),
    }
    if args.dataset == "ava":
        result["ava_average_precision_percent"] = float(
            average_precision_score(labels, scores) * 100.0
        )
    else:
        per_speaker = {}
        for speaker in sorted({row["speaker"] for row in predictions}):
            selected = [row for row in predictions if row["speaker"] == speaker]
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
            fieldnames=("window_index", "node_index", "slot", "speaker", "label", "score"),
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
