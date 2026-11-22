"""Validate the held-out Columbia cache."""

import argparse
import json
from pathlib import Path
import torch

EXPECTED = {"v": (5, 50, 896), "a": (5, 50, 128), "t": (5, 50, 128),
            "s": (5, 50, 4), "l": (5, 50), "m": (5, 50)}

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "realtime" / "columbia_cached_chunks")
    parser.add_argument("--output", type=Path,
                        default=root / "realtime" / "columbia_cache_audit.json")
    args = parser.parse_args()
    files = sorted(args.cache_dir.glob("col_cache_part_*.pt")) if args.cache_dir.is_dir() else []
    report = {"directory": str(args.cache_dir.resolve()), "files": len(files),
              "windows": 0, "errors": []}
    for path in files:
        try:
            data = torch.load(path, map_location="cpu")
            if set(data) != set(EXPECTED) | {"names"}:
                report["errors"].append(f"{path.name}: keys={sorted(data)}")
                continue
            sizes = set()
            for key, tail in EXPECTED.items():
                shape = tuple(data[key].shape)
                if len(shape) != len(tail) + 1 or shape[1:] != tail:
                    report["errors"].append(f"{path.name}:{key} shape={shape}")
                else:
                    sizes.add(shape[0])
            if len(sizes) != 1:
                report["errors"].append(f"{path.name}: inconsistent batch dimensions")
            else:
                count = next(iter(sizes))
                if len(data["names"]) != count:
                    report["errors"].append(f"{path.name}: names length mismatch")
                report["windows"] += count
        except Exception as error:
            report["errors"].append(f"{path.name}: {type(error).__name__}: {error}")
    report["passed"] = bool(files) and report["windows"] > 0 and not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Columbia cache audit failed")

if __name__ == "__main__":
    main()
