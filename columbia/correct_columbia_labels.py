"""Create an audited Columbia label copy with known track identities corrected.

The downloaded LR-ASD label archive is retained unchanged.  Corrections are
applied only to a separate copy consumed by the local preprocessing pipeline.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


CORRECTIONS = (
    ("tracks_74680_75300/lieb.txt", "tracks_74680_75300/bell.txt"),
    ("tracks_74680_75300/long.txt", "tracks_74680_75300/boll.txt"),
    ("tracks_154000_155880/lieb.txt", "tracks_154000_155880/abbas.txt"),
    ("tracks_154000_155880/long.txt", "tracks_154000_155880/lieb.txt"),
    ("tracks_148120_149540/sick.txt", "tracks_148120_149540/long.txt"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*.txt"))
    }


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "colDataPath" / "col_labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "colDataPath" / "col_labels_corrected",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "colDataPath" / "col_labels_correction_manifest.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".tmp")

    if not source.is_dir():
        raise SystemExit(f"Missing downloaded label directory: {source}")
    if output.exists():
        raise SystemExit(
            f"Corrected label directory already exists: {output}\n"
            "Retain it if already audited, or remove only that generated directory "
            "before rebuilding it."
        )
    if temporary.exists():
        raise SystemExit(f"Remove incomplete temporary directory first: {temporary}")

    source_inventory = inventory(source)
    applied = []
    correction_sources = {old for old, _ in CORRECTIONS}
    for old_relative, new_relative in CORRECTIONS:
        if old_relative not in source_inventory:
            raise SystemExit(f"Expected source label is missing: {old_relative}")
        if new_relative in source_inventory and new_relative not in correction_sources:
            raise SystemExit(
                f"Correction target already exists in source archive: {new_relative}"
            )

    shutil.copytree(source, temporary)
    try:
        for old_relative, new_relative in CORRECTIONS:
            old_path = temporary / old_relative
            new_path = temporary / new_relative
            if new_path.exists():
                raise RuntimeError(f"Correction would overwrite a label: {new_relative}")
            old_hash = sha256(old_path)
            old_path.rename(new_path)
            applied.append(
                {
                    "from": old_relative,
                    "to": new_relative,
                    "content_sha256": old_hash,
                }
            )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    corrected_inventory = inventory(output)
    expected_count = len(source_inventory)
    expected_inventory = dict(source_inventory)
    for old_relative, new_relative in CORRECTIONS:
        expected_inventory[new_relative] = expected_inventory.pop(old_relative)
    report = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "source_files": expected_count,
        "corrected_files": len(corrected_inventory),
        "corrections": applied,
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(source_inventory, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "corrected_inventory_sha256": hashlib.sha256(
            json.dumps(corrected_inventory, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "passed": len(corrected_inventory) == expected_count
        and corrected_inventory == expected_inventory,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Columbia label correction audit failed")


if __name__ == "__main__":
    main()
