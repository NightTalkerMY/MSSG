"""Audit the seven Columbia feature files before cache construction."""

import argparse
import csv
import json
from pathlib import Path

FAMILIES = (
    "face_feats_info", "face_body_large_feats_info", "face.body_feats_info",
    "face_large_feats_info", "background_feats_info", "face_small_feats_info",
    "face_down_feats_info",
)

def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--predata", type=Path,
                        default=root / "data" / "Columbia" / "predata")
    parser.add_argument("--output", type=Path,
                        default=root / "realtime" / "columbia_input_audit.json")
    return parser.parse_args()

def read_keys(path):
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                keys.add(row[0])
    return keys

def main():
    args = parse_args()
    label_file = args.predata / "csv_split" / "train" / "col.csv"
    report = {"predata": str(args.predata.resolve()), "label_file": str(label_file),
              "families": {}, "passed": label_file.is_file() and label_file.stat().st_size > 0}
    key_sets = {}
    for family in FAMILIES:
        path = args.predata / family / "train" / "col.csv"
        exists = path.is_file() and path.stat().st_size > 0
        keys = read_keys(path) if exists else set()
        key_sets[family] = keys
        report["families"][family] = {"path": str(path), "rows": len(keys), "nonempty": exists}
        report["passed"] = report["passed"] and exists and bool(keys)
    reference = key_sets["face_feats_info"]
    for family, keys in key_sets.items():
        missing = sorted(reference - keys)
        unexpected = sorted(keys - reference)
        report["families"][family]["missing_against_face"] = missing
        report["families"][family]["unexpected_against_face"] = unexpected
        report["passed"] = report["passed"] and not missing and not unexpected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Columbia feature input audit failed")

if __name__ == "__main__":
    main()
