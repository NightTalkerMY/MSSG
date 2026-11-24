"""Perturb future inputs to verify the temporal support of a student head."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proposed"))

from student_model import CausalStudentASD  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expect", choices=("causal", "noncausal"), required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = ROOT / "realtime" / "experiments" / args.run_name
    manifest_path = run_dir / "run_manifest.json"
    checkpoint = run_dir / "best_ava.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise SystemExit(f"Missing retained run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    architecture = manifest.get("architecture", "full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalStudentASD(architecture=architecture).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cutoffs = (10, 25, 40)
    trial_results = []
    with torch.no_grad():
        for trial in range(args.trials):
            generator = torch.Generator(device=device).manual_seed(7100 + trial)
            visual = torch.randn(1, 5, 50, 896, generator=generator, device=device)
            audio = torch.randn(1, 5, 50, 128, generator=generator, device=device)
            spatial = torch.randn(1, 5, 50, 4, generator=generator, device=device)
            mask = torch.ones(1, 5, 50, device=device)
            reference, _ = model(visual, audio, spatial, mask)
            for cutoff in cutoffs:
                altered_visual = visual.clone()
                altered_audio = audio.clone()
                altered_spatial = spatial.clone()
                altered_visual[:, :, cutoff + 1:] = torch.randn(
                    altered_visual[:, :, cutoff + 1:].shape,
                    generator=generator, device=device,
                ) * 10.0
                altered_audio[:, :, cutoff + 1:] = torch.randn(
                    altered_audio[:, :, cutoff + 1:].shape,
                    generator=generator, device=device,
                ) * 10.0
                altered_spatial[:, :, cutoff + 1:] = torch.randn(
                    altered_spatial[:, :, cutoff + 1:].shape,
                    generator=generator, device=device,
                ) * 10.0
                changed, _ = model(
                    altered_visual, altered_audio, altered_spatial, mask
                )
                past_difference = float(
                    (reference[:, :, :cutoff + 1] - changed[:, :, :cutoff + 1])
                    .abs().max().item()
                )
                trial_results.append({
                    "trial": trial,
                    "cutoff": cutoff,
                    "maximum_past_logit_absolute_difference": past_difference,
                })

    maximum = max(
        item["maximum_past_logit_absolute_difference"] for item in trial_results
    )
    passed = maximum <= args.tolerance if args.expect == "causal" else maximum > args.tolerance
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "architecture": architecture,
        "expectation": args.expect,
        "test": "future inputs after each cutoff are replaced; logits through the cutoff are compared",
        "cutoffs": list(cutoffs),
        "trials": args.trials,
        "tolerance": args.tolerance,
        "maximum_past_logit_absolute_difference": maximum,
        "passed": passed,
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "trial_results": trial_results,
        "scope_note": (
            "This verifies the CATT-ASD prediction head with respect to its supplied "
            "features; upstream feature temporal support requires a separate audit."
        ),
    }
    output = args.output or (
        ROOT / "realtime" / "analysis" / "causality" / f"{args.run_name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Future-input perturbation test did not match its expectation")


if __name__ == "__main__":
    main()
