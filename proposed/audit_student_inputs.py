"""Audit the seven AVA feature-CSV families required by CATT-ASD."""

import argparse
import json
from pathlib import Path

FEATURE_FAMILIES = (
    "face_feats_info", "background_feats_info", "face_small_feats_info",
    "face.body_feats_info", "face_large_feats_info", "face_down_feats_info",
    "face_body_large_feats_info",
)

def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--predata", type=Path, default=root / "predata")
    parser.add_argument("--output", type=Path, default=root / "realtime" / "student_input_audit.json")
    parser.add_argument("--expected-train", type=int, default=120)
    parser.add_argument("--expected-val", type=int, default=33)
    return parser.parse_args()

def main():
    args = parse_args()
    report = {"predata": str(args.predata.resolve()), "splits": {}, "passed": True}
    for split, expected in (("train", args.expected_train), ("val", args.expected_val)):
        family_files = {}
        for family in FEATURE_FAMILIES:
            folder = args.predata / family / split
            files = sorted(p for p in folder.glob("*.csv") if p.stat().st_size > 0) if folder.is_dir() else []
            family_files[family] = {p.name for p in files}
        reference = family_files["face_feats_info"]
        details, split_passed = {}, len(reference) == expected
        for family, names in family_files.items():
            missing = sorted(reference - names)
            unexpected = sorted(names - reference)
            passed = len(names) == expected and not missing and not unexpected
            split_passed = split_passed and passed
            details[family] = {"nonempty_csv_files": len(names), "missing_against_face": missing,
                               "unexpected_against_face": unexpected, "passed": passed}
        report["splits"][split] = {"expected_csv_files_per_family": expected,
                                    "families": details, "passed": split_passed}
        report["passed"] = report["passed"] and split_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Student feature input audit failed")

if __name__ == "__main__":
    main()
