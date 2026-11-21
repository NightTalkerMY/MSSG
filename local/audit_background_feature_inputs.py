"""Audit every input join used by ``graph_init.sh 3`` before GPU extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return value.replace(":", "_")


def read_loader(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                raise RuntimeError(f"Malformed loader row {path}:{line_number}")
            rows.append((fields[0], int(fields[1])))
    return rows


def inspect_split(data_root: Path, clip_root: Path, clip_info_root: Path, split: str) -> dict:
    loader_rows = read_loader(data_root / "csv" / f"{split}_loader.csv")
    clip_info = torch.load(clip_info_root / f"{split}_clip_info.pt", map_location="cpu")
    audio_root = data_root / ("clips_audios_aug" if split == "train" else "clips_audios") / split
    visual_root = clip_root / split
    missing_clip_info = []
    timestamp_mismatches = []
    missing_audio = []
    missing_visual_folder = []
    missing_visual_frames = []
    extra_or_short_visual_counts = []

    for clip_name, num_frames in loader_rows:
        safe_clip = safe_name(clip_name)
        video = clip_name[:11]
        timestamps = clip_info.get(clip_name)
        if timestamps is None:
            missing_clip_info.append(clip_name)
            continue
        if len(timestamps) != num_frames:
            timestamp_mismatches.append(
                {"clip": clip_name, "loader_frames": num_frames, "clip_info_frames": len(timestamps)}
            )

        audio = audio_root / video / f"{safe_clip}.wav"
        if not audio.is_file() or audio.stat().st_size == 0:
            missing_audio.append(audio.as_posix())

        visual = visual_root / video / safe_clip
        if not visual.is_dir():
            missing_visual_folder.append(visual.as_posix())
            continue
        actual_names = {path.name for path in visual.glob("*.jpg")}
        expected_names = {f"{float(timestamp):.2f}.jpg" for timestamp in timestamps}
        absent = sorted(expected_names - actual_names)
        if absent:
            missing_visual_frames.append(
                {"clip": clip_name, "missing_count": len(absent), "examples": absent[:5]}
            )
        if len(actual_names) != num_frames:
            extra_or_short_visual_counts.append(
                {"clip": clip_name, "loader_frames": num_frames, "actual_jpegs": len(actual_names)}
            )

    loader_keys = {name for name, _count in loader_rows}
    unused_clip_info = sorted(set(clip_info) - loader_keys)
    failures = (
        missing_clip_info
        or timestamp_mismatches
        or missing_audio
        or missing_visual_folder
        or missing_visual_frames
        or extra_or_short_visual_counts
        or unused_clip_info
    )
    return {
        "passed": not bool(failures),
        "loader_clips": len(loader_rows),
        "clip_info_entries": len(clip_info),
        "missing_clip_info": missing_clip_info,
        "timestamp_count_mismatches": timestamp_mismatches,
        "missing_or_empty_audio": missing_audio,
        "missing_visual_folders": missing_visual_folder,
        "missing_visual_frames": missing_visual_frames,
        "visual_count_mismatches": extra_or_short_visual_counts,
        "clip_info_entries_not_in_loader": unused_clip_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/AVA"))
    parser.add_argument(
        "--resized-root",
        type=Path,
        default=Path("data/AVA/clips_videos_background_resize_224/AVA/clips_videos_background"),
    )
    parser.add_argument("--clip-info-root", type=Path, default=Path("predata/clip_info"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("predata/pretrain_model/background_model.model"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("predata/background_feature_input_audit.json"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {args.checkpoint}")

    report = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "resized_root": args.resized_root.as_posix(),
        "splits": {
            split: inspect_split(args.data_root, args.resized_root, args.clip_info_root, split)
            for split in ("train", "val")
        },
    }
    report["passed"] = all(value["passed"] for value in report["splits"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
