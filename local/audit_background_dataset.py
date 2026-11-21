"""Audit AVA background crops before feature extraction or archival.

This script is read-only. It compares the train/val background JPEG tree with
the AVA annotation CSVs, accounts for unavailable source videos, checks the
expected ``split/video/entity/frame.jpg`` layout, and decodes deterministic
sample images from every populated leaf directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd


SPLITS = {"train": "trainval", "val": "trainval"}


def safe_name(value: object) -> str:
    return str(value).replace(":", "_")


def expected_counts(csv_path: Path, available_videos: set[str]) -> tuple[Counter, int]:
    frame_names: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    unavailable_rows = 0
    frame = pd.read_csv(
        csv_path,
        usecols=["video_id", "entity_id", "frame_timestamp"],
        dtype={"video_id": str, "entity_id": str},
    )
    for row in frame.itertuples(index=False):
        video = safe_name(row.video_id)
        if video not in available_videos:
            unavailable_rows += 1
            continue
        entity = safe_name(row.entity_id)
        frame_names[(video, entity)].add(f"{float(row.frame_timestamp):.2f}.jpg")
    return Counter({key: len(names) for key, names in frame_names.items()}), unavailable_rows


def actual_counts(split_root: Path, decode_samples: int) -> tuple[Counter, list[str], list[str], int]:
    counts: Counter = Counter()
    malformed: list[str] = []
    decode_failures: list[str] = []
    non_jpeg_files = 0
    if not split_root.exists():
        return counts, [f"missing split directory: {split_root}"], decode_failures, non_jpeg_files

    for root, _directories, files in os.walk(split_root):
        entity_dir = Path(root)
        relative = entity_dir.relative_to(split_root)
        images = sorted(entity_dir / name for name in files if Path(name).suffix.lower() == ".jpg")
        non_jpeg_files += sum(1 for name in files if Path(name).suffix.lower() != ".jpg")
        if not images:
            continue
        if len(relative.parts) != 2:
            malformed.append(relative.as_posix())
            continue
        counts[(relative.parts[0], relative.parts[1])] = len(images)
        if images:
            sample_indices = sorted({int(index * (len(images) - 1) / max(1, decode_samples - 1)) for index in range(decode_samples)})
            for index in sample_indices:
                image = cv2.imread(str(images[index]), cv2.IMREAD_GRAYSCALE)
                if image is None or image.size == 0:
                    decode_failures.append(str(images[index]))
    return counts, malformed, decode_failures, non_jpeg_files


def source_video_ids(directory: Path) -> set[str]:
    return {safe_name(path.stem) for path in directory.iterdir() if path.is_file()} if directory.exists() else set()


def hash_report_payload(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/AVA"))
    parser.add_argument("--background-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predata/background_audit.json"))
    parser.add_argument("--decode-samples-per-entity", type=int, default=1)
    args = parser.parse_args()
    background_root = args.background_root or args.data_root / "clips_videos_background"
    report = {
        "schema_version": 1,
        "data_root": str(args.data_root),
        "background_root": str(background_root),
        "splits": {},
    }
    passed = True

    for split, source_group in SPLITS.items():
        available = source_video_ids(args.data_root / "orig_videos" / source_group)
        expected, unavailable_rows = expected_counts(args.data_root / "csv" / f"{split}_orig.csv", available)
        actual, malformed, decode_failures, non_jpeg = actual_counts(
            background_root / split, args.decode_samples_per_entity
        )
        missing_entities = sorted(key for key in expected if key not in actual)
        unexpected_entities = sorted(key for key in actual if key not in expected)
        count_mismatches = sorted(
            [{
                "video_id": key[0],
                "entity_id": key[1],
                "expected": expected[key],
                "actual": actual[key],
            }
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]],
            key=lambda row: (row["video_id"], row["entity_id"]),
        )
        split_passed = not (
            missing_entities or unexpected_entities or count_mismatches or malformed or decode_failures or non_jpeg
        )
        passed = passed and split_passed
        report["splits"][split] = {
            "passed": split_passed,
            "available_source_videos": len(available),
            "annotation_rows_skipped_for_unavailable_source_videos": unavailable_rows,
            "expected_entities": len(expected),
            "actual_entities": len(actual),
            "expected_jpegs": sum(expected.values()),
            "actual_jpegs": sum(actual.values()),
            "missing_entities": [f"{video}/{entity}" for video, entity in missing_entities],
            "unexpected_entities": [f"{video}/{entity}" for video, entity in unexpected_entities],
            "count_mismatches": count_mismatches,
            "malformed_leaf_paths": malformed,
            "decode_failures": decode_failures,
            "non_jpeg_files": non_jpeg,
        }

    report["passed"] = passed
    report["content_sha256"] = hash_report_payload(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
