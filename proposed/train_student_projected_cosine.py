"""Train CATT-ASD with a training-only projected cosine distillation loss.

The projection head is optimized only to align the causal student's
intermediate feature direction with the retained offline-teacher feature.  It
is not part of the deployed student checkpoint or inference path.
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from student_model import CausalStudentASD
from train_student_seeded import CacheDataset, set_seed, validate_last_frame


class IdentityProjection(nn.Module):
    """Learnable 128-D mapping initialized as an identity transformation."""

    def __init__(self, dimension=128):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dimension))
            self.linear.bias.zero_()

    def forward(self, features):
        return self.linear(features)


class ProjectedCosineLoss(nn.Module):
    def __init__(self, projection, distill_weight):
        super().__init__()
        self.projection = projection
        self.distill_weight = distill_weight
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

    def forward(self, logits, features, teacher_features, labels, mask):
        valid = mask.reshape(-1) > 0.5
        ce_loss = self.ce(
            logits.reshape(-1, 2)[valid], labels.reshape(-1)[valid]
        ).mean()
        student = self.projection(features.reshape(-1, 128)[valid])
        teacher = teacher_features.reshape(-1, 128)[valid]
        student = F.normalize(student, p=2, dim=-1, eps=1e-8)
        teacher = F.normalize(teacher, p=2, dim=-1, eps=1e-8)
        cosine_loss = (1.0 - (student * teacher).sum(dim=-1)).mean()
        total = ce_loss + self.distill_weight * cosine_loss
        return total, ce_loss, cosine_loss


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--train-cache", type=Path,
                        default=root / "realtime" / "cached_chunks")
    parser.add_argument("--val-cache", type=Path,
                        default=root / "realtime" / "val_cached_chunks")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name", default="ce_projected_cosine_beta_01_seed_42")
    return parser.parse_args(), root


def main():
    args, root = parse_args()
    if args.distill_weight < 0:
        raise SystemExit("--distill-weight must be non-negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or root / "realtime" / "experiments" / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        CacheDataset(args.train_cache, "train", shuffle=True),
        batch_size=args.batch_size, num_workers=0,
    )
    val_loader = DataLoader(
        CacheDataset(args.val_cache, "val", shuffle=False),
        batch_size=args.val_batch_size, num_workers=0,
    )
    model = CausalStudentASD().to(device)
    projection = IdentityProjection().to(device)
    criterion = ProjectedCosineLoss(projection, args.distill_weight)
    optimized_parameters = list(model.parameters()) + list(projection.parameters())
    optimizer = AdamW(optimized_parameters, lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    log_path = output_dir / "training_log.csv"
    best_map, best_epoch = float("-inf"), None
    started = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "epoch", "avg_loss", "avg_ce", "avg_cosine", "lr",
            "ava_final_frame_map", "val_samples", "epoch_seconds",
        ))
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            model.train()
            projection.train()
            totals = np.zeros(3, dtype=np.float64)
            batches = 0
            progress = tqdm(
                train_loader, desc=f"{args.run_name} epoch {epoch}/{args.epochs}"
            )
            for visual, audio, teacher, spatial, labels, mask in progress:
                visual, audio = visual.to(device), audio.to(device)
                teacher, spatial = teacher.to(device), spatial.to(device)
                labels, mask = labels.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model(visual, audio, spatial, mask)
                loss, ce_loss, cosine_loss = criterion(
                    logits, features, teacher, labels, mask
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
                optimizer.step()
                totals += (loss.item(), ce_loss.item(), cosine_loss.item())
                batches += 1
                progress.set_postfix(
                    loss=f"{loss.item():.3f}", cosine=f"{cosine_loss.item():.3f}"
                )
            if batches == 0:
                raise RuntimeError("Training cache produced zero batches")
            scheduler.step()
            ava_map, val_samples = validate_last_frame(model, val_loader, device)
            elapsed = time.time() - epoch_started
            averages = totals / batches
            writer.writerow((
                epoch, *[f"{value:.8f}" for value in averages],
                f"{scheduler.get_last_lr()[0]:.8f}", f"{ava_map:.6f}",
                val_samples, f"{elapsed:.3f}",
            ))
            handle.flush()
            torch.save(model.state_dict(), output_dir / "last.pt")
            torch.save(projection.state_dict(), output_dir / "last_projection.pt")
            if ava_map > best_map:
                best_map, best_epoch = ava_map, epoch
                torch.save(model.state_dict(), output_dir / "best_ava.pt")
                torch.save(
                    projection.state_dict(), output_dir / "best_projection.pt"
                )
            print(
                f"Epoch {epoch}: AVA final-frame mAP={ava_map:.4f}; "
                f"best={best_map:.4f} at epoch {best_epoch}"
            )

    manifest = {
        "run_name": args.run_name,
        "variant": "ce_projected_cosine",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.val_batch_size,
        "classification_loss": "cross entropy with label smoothing 0.1",
        "distillation_loss": "cosine distance after identity-initialized training-only linear projection",
        "distill_weight": args.distill_weight,
        "projection_parameters_training_only": sum(
            parameter.numel() for parameter in projection.parameters()
        ),
        "projection_used_during_inference": False,
        "inference_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "checkpoint_selection": "highest AVA final-frame validation mAP only",
        "columbia_used_during_training_or_selection": False,
        "best_epoch": best_epoch,
        "best_ava_final_frame_map": best_map,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": time.time() - started,
        "train_cache": str(args.train_cache.resolve()),
        "val_cache": str(args.val_cache.resolve()),
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (output_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    evidence_dir = root / "realtime" / "revision_evidence" / args.run_name
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_path, evidence_dir / "training_log.csv")
    (evidence_dir / "run_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    print(manifest_text)


if __name__ == "__main__":
    main()
