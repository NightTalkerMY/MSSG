"""Validate CATT-ASD train and AVA-validation cache structure."""

import argparse
import json
from pathlib import Path
import torch

EXPECTED_TAILS = {"v": (5, 50, 896), "a": (5, 50, 128), "t": (5, 50, 128),
                  "s": (5, 50, 4), "l": (5, 50), "m": (5, 50)}

def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=root / "realtime" / "cached_chunks")
    parser.add_argument("--val-dir", type=Path, default=root / "realtime" / "val_cached_chunks")
    parser.add_argument("--output", type=Path, default=root / "realtime" / "student_cache_audit.json")
    return parser.parse_args()

def audit(directory, prefix):
    files = sorted(directory.glob(f"{prefix}_cache_part_*.pt")) if directory.is_dir() else []
    result = {"directory": str(directory.resolve()), "files": len(files), "windows": 0, "errors": []}
    for path in files:
        try:
            data = torch.load(path, map_location="cpu")
            if set(data) != set(EXPECTED_TAILS):
                result["errors"].append(f"{path.name}: keys={sorted(data)}")
                continue
            batch_sizes = set()
            for key, tail in EXPECTED_TAILS.items():
                shape = tuple(data[key].shape)
                if len(shape) != len(tail) + 1 or shape[1:] != tail:
                    result["errors"].append(f"{path.name}:{key} shape={shape}, expected=(*,{tail})")
                else:
                    batch_sizes.add(shape[0])
            if len(batch_sizes) != 1:
                result["errors"].append(f"{path.name}: inconsistent leading dimensions {sorted(batch_sizes)}")
            else:
                result["windows"] += next(iter(batch_sizes))
        except Exception as error:
            result["errors"].append(f"{path.name}: {type(error).__name__}: {error}")
    result["passed"] = bool(files) and result["windows"] > 0 and not result["errors"]
    return result

def main():
    args = parse_args()
    report = {"train": audit(args.train_dir, "train"), "val": audit(args.val_dir, "val")}
    report["passed"] = report["train"]["passed"] and report["val"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Student cache audit failed")

if __name__ == "__main__":
    main()
