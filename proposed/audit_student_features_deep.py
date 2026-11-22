"""Deep, streaming audit of all seven AVA feature-CSV families.

The audit reads every row without loading a complete CSV into memory. It checks
file membership, four-column schema, 128-dimensional numeric feature vectors,
finite values, row counts, and ordered frame-key agreement across families.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

FAMILIES = (
    "face_feats_info",
    "background_feats_info",
    "face_small_feats_info",
    "face.body_feats_info",
    "face_large_feats_info",
    "face_down_feats_info",
    "face_body_large_feats_info",
)
EXPECTED_FILES = {"train": 120, "val": 33}
EXPECTED_FEATURE_DIM = 128
MAX_RECORDED_ERRORS_PER_FILE = 20


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--predata", type=Path, default=root / "predata")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "realtime" / "student_feature_deep_audit.json",
    )
    parser.add_argument(
        "--skip-value-validation",
        action="store_true",
        help="Check dimensions but do not convert every token to float.",
    )
    return parser.parse_args()


def record_error(errors, message):
    if len(errors) < MAX_RECORDED_ERRORS_PER_FILE:
        errors.append(message)


def validate_vector(text, row_number, column_number, errors, check_values):
    stripped = text.strip().strip("[]")
    tokens = stripped.split(",") if stripped else []
    if len(tokens) != EXPECTED_FEATURE_DIM:
        record_error(
            errors,
            f"row {row_number}, column {column_number}: "
            f"dimension {len(tokens)}, expected {EXPECTED_FEATURE_DIM}",
        )
        return
    if not check_values:
        return
    try:
        values = (float(token) for token in tokens)
        if not all(math.isfinite(value) for value in values):
            record_error(errors, f"row {row_number}, column {column_number}: non-finite value")
    except ValueError as error:
        record_error(errors, f"row {row_number}, column {column_number}: {error}")


def scan_csv(path, check_values):
    digest = hashlib.sha256()
    errors = []
    rows = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                rows += 1
                if len(row) != 4:
                    record_error(errors, f"row {row_number}: {len(row)} columns, expected 4")
                    continue
                key = row[0].strip()
                if not key:
                    record_error(errors, f"row {row_number}: empty frame key")
                digest.update(key.encode("utf-8", errors="replace"))
                digest.update(b"\n")
                for column_number in (1, 2, 3):
                    validate_vector(
                        row[column_number], row_number, column_number, errors, check_values
                    )
    except Exception as error:
        record_error(errors, f"{type(error).__name__}: {error}")
    if rows == 0:
        record_error(errors, "file contains zero rows")
    return {"rows": rows, "ordered_key_sha256": digest.hexdigest(), "errors": errors}


def main():
    args = parse_args()
    check_values = not args.skip_value_validation
    report = {
        "predata": str(args.predata.resolve()),
        "numeric_value_validation": check_values,
        "expected_feature_dimension": EXPECTED_FEATURE_DIM,
        "splits": {},
        "issues": [],
        "passed": True,
    }

    jobs = []
    family_names = {}
    for split, expected_count in EXPECTED_FILES.items():
        family_names[split] = {}
        for family in FAMILIES:
            directory = args.predata / family / split
            names = {path.name for path in directory.glob("*.csv")} if directory.is_dir() else set()
            family_names[split][family] = names
            if len(names) != expected_count:
                report["issues"].append(
                    f"{split}/{family}: {len(names)} CSV files, expected {expected_count}"
                )
            for name in sorted(names):
                jobs.append((split, family, name, directory / name))

        all_names = set().union(*family_names[split].values())
        for name in sorted(all_names):
            for family in FAMILIES:
                if name not in family_names[split][family]:
                    report["issues"].append(f"{split}/{family}/{name}: missing file")

    scans = {split: {family: {} for family in FAMILIES} for split in EXPECTED_FILES}
    total = len(jobs)
    for number, (split, family, name, path) in enumerate(jobs, start=1):
        print(f"[{number:,}/{total:,}] {split}/{family}/{name}", flush=True)
        result = scan_csv(path, check_values)
        scans[split][family][name] = result
        for error in result["errors"]:
            report["issues"].append(f"{split}/{family}/{name}: {error}")

    for split in EXPECTED_FILES:
        reference = scans[split]["face_feats_info"]
        for name, reference_result in reference.items():
            for family in FAMILIES[1:]:
                result = scans[split][family].get(name)
                if result is None:
                    continue
                if result["rows"] != reference_result["rows"]:
                    report["issues"].append(
                        f"{split}/{family}/{name}: {result['rows']} rows; "
                        f"face reference has {reference_result['rows']}"
                    )
                if result["ordered_key_sha256"] != reference_result["ordered_key_sha256"]:
                    report["issues"].append(
                        f"{split}/{family}/{name}: frame-key sequence differs from face reference"
                    )

        report["splits"][split] = {
            "expected_files_per_family": EXPECTED_FILES[split],
            "families": {
                family: {
                    "files": len(scans[split][family]),
                    "rows": sum(item["rows"] for item in scans[split][family].values()),
                    "files_with_schema_errors": sum(
                        bool(item["errors"]) for item in scans[split][family].values()
                    ),
                }
                for family in FAMILIES
            },
        }

    report["passed"] = not report["issues"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit report: {args.output}")
    if not report["passed"]:
        raise SystemExit("Deep feature audit failed; inspect the report's issues list")


if __name__ == "__main__":
    main()
