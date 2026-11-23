"""Evaluate the retained MSSG graph checkpoint without retraining.

This is MSSG's native graph-node protocol.  It is intentionally kept separate
from CATT-ASD's cached 50-frame final-position protocol.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.data import Data
from tqdm import tqdm

from model.models_gnn import MSSG


SPEAKERS = ("Abbas", "Bell", "Boll", "Lieb", "Long", "Sick")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_f1(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(values) * 100.0)


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ava", "columbia"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "exps" / "graph_training" / "graph_2000" / "chckpoint_best.pt",
    )
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--columbia-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(), root


def main():
    args, root = parse_args()
    if args.graph_dir is None:
        if args.dataset == "ava":
            args.graph_dir = root / "graphs" / "graph_2000_0" / "val" / "processed"
        else:
            args.graph_dir = (
                root / "data" / "Columbia" / "graphs" /
                "graph_2000_0" / "train" / "processed"
            )
    if args.columbia_csv is None:
        args.columbia_csv = (
            root / "data" / "Columbia" / "predata" /
            "csv_split" / "train" / "col.csv"
        )
    if args.output_dir is None:
        args.output_dir = root / "realtime" / "baseline_runs" / f"mssg_{args.dataset}"

    checkpoint = args.checkpoint.resolve()
    graph_dir = args.graph_dir.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Missing MSSG checkpoint: {checkpoint}")
    graph_files = sorted(graph_dir.glob("*.pt")) if graph_dir.is_dir() else []
    if not graph_files:
        raise SystemExit(f"No MSSG graph files found in {graph_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSSG().to(device)
    state = torch.load(checkpoint, map_location=device)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    labels, scores, prediction_rows = [], [], []
    with torch.no_grad():
        for graph_path in tqdm(graph_files, desc=f"MSSG {args.dataset}"):
            graph = torch.load(graph_path, map_location="cpu")
            if not isinstance(graph, Data):
                raise RuntimeError(f"Unexpected graph object in {graph_path}")
            graph = graph.to(device)
            probabilities = F.softmax(model(graph), dim=-1)[:, 1].cpu().numpy()
            graph_labels = graph.y[:, 0].cpu().numpy()
            for node_index, (label, score) in enumerate(zip(graph_labels, probabilities)):
                binary_label = 0 if int(label) == 2 else int(label)
                labels.append(binary_label)
                scores.append(float(score))
                prediction_rows.append([
                    graph_path.name, node_index, binary_label, float(score)
                ])

    result = {
        "schema_version": 1,
        "method": "MSSG",
        "dataset": args.dataset,
        "protocol": "native MSSG graph-node inference over generated graph files",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "graph_directory": str(graph_dir),
        "graph_files": len(graph_files),
        "prediction_nodes": len(labels),
        "graph_head_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_scope": "MSSG graph prediction model only; upstream features are precomputed",
        "device": str(device),
    }

    if args.dataset == "ava":
        result["ava_average_precision_percent"] = float(
            average_precision_score(labels, scores) * 100.0
        )
    else:
        if not args.columbia_csv.is_file():
            raise SystemExit(f"Missing Columbia mapping CSV: {args.columbia_csv}")
        with args.columbia_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        rows.sort(key=lambda row: float(row[1]))
        if len(rows) != len(labels):
            raise RuntimeError(
                f"Columbia mapping mismatch: {len(rows)} CSV rows, "
                f"{len(labels)} predictions"
            )
        by_speaker = {speaker: {"labels": [], "scores": []} for speaker in SPEAKERS}
        for row, label, score, prediction in zip(rows, labels, scores, prediction_rows):
            speaker = row[7].split("_")[0].capitalize()
            if speaker not in by_speaker:
                raise RuntimeError(f"Unexpected Columbia speaker: {speaker}")
            by_speaker[speaker]["labels"].append(label)
            by_speaker[speaker]["scores"].append(score)
            prediction.extend([speaker, row[0], row[1], row[7]])
        per_speaker = {}
        for speaker in SPEAKERS:
            item = by_speaker[speaker]
            per_speaker[speaker] = {
                "samples": len(item["labels"]),
                "average_precision_percent": float(
                    average_precision_score(item["labels"], item["scores"]) * 100.0
                ),
                "best_f1_percent": best_f1(item["labels"], item["scores"]),
            }
        result["columbia_per_speaker"] = per_speaker
        result["columbia_six_speaker_macro_best_f1_percent"] = float(
            np.mean([item["best_f1_percent"] for item in per_speaker.values()])
        )

    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["graph_file", "node_index", "label", "score"]
        if args.dataset == "columbia":
            header += ["speaker", "video_id", "timestamp", "entity_id"]
        writer.writerow(header)
        writer.writerows(prediction_rows)
    result["predictions"] = str(prediction_path.resolve())
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
