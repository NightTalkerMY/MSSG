"""Create per-video AVA label files required by the student cache loader."""

import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, default=root / "data" / "AVA" / "csv")
    parser.add_argument("--data-root", type=Path, default=root / "data" / "AVA")
    return parser.parse_args(), root


def convert(source, destination, expected_files):
    if not source.is_file():
        raise FileNotFoundError(f"Missing AVA annotation CSV: {source}")
    existing = sorted(destination.glob("*-activespeaker.csv")) if destination.is_dir() else []
    if existing:
        if len(existing) != expected_files or any(path.stat().st_size == 0 for path in existing):
            raise RuntimeError(
                f"Existing label directory is incomplete: {destination} "
                f"contains {len(existing)} files, expected {expected_files}"
            )
        print(f"Using existing label directory: {destination} ({len(existing)} files)")
        return {"source": str(source), "destination": str(destination),
                "files": len(existing), "rows": None, "reused": True}

    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    with source.open("r", newline="", encoding="utf-8") as input_handle, ExitStack() as stack:
        reader = csv.reader(input_handle)
        writers = {}
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if row_number == 1 and row[0].strip().lower() in {"video_id", "video"}:
                continue
            if len(row) < 8:
                raise RuntimeError(f"{source}: row {row_number} has {len(row)} columns, expected at least 8")
            video_id = row[0].strip()
            if not video_id or Path(video_id).name != video_id:
                raise RuntimeError(f"{source}: unsafe or empty video ID at row {row_number}: {video_id!r}")
            if video_id not in writers:
                path = destination / f"{video_id}-activespeaker.csv"
                handle = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
                writers[video_id] = csv.writer(handle)
                counts[video_id] = 0
            writers[video_id].writerow(row)
            counts[video_id] += 1
    if len(counts) != expected_files or any(count == 0 for count in counts.values()):
        raise RuntimeError(
            f"Converted {len(counts)} video files from {source}, expected {expected_files}"
        )
    return {"source": str(source), "destination": str(destination),
            "files": len(counts), "rows": sum(counts.values()), "reused": False}


def main():
    args, root = parse_args()
    report = {
        "train": convert(
            args.csv_root / "train_orig.csv",
            args.data_root / "ava_activespeaker_train_v1.0",
            120,
        ),
        "val": convert(
            args.csv_root / "val_orig.csv",
            args.data_root / "ava_activespeaker_test_v1.0",
            33,
        ),
        "passed": True,
    }
    output = root / "realtime" / "ava_cache_label_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Label manifest: {output}")


if __name__ == "__main__":
    main()
