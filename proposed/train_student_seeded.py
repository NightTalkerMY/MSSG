"""Reproducible CATT-ASD training with AVA-only checkpoint selection.

Columbia is deliberately absent. Evaluate it only after selecting the best
checkpoint using AVA final-frame validation mAP.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from student_model import CausalStudentASD


class CacheDataset(IterableDataset):
    def __init__(self, directory, prefix, shuffle):
        self.files = sorted(Path(directory).glob(f"{prefix}_cache_part_*.pt"))
        self.shuffle = shuffle
        if not self.files:
            raise FileNotFoundError(f"No {prefix} cache files found in {directory}")

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)
        for path in files:
            data = torch.load(path, map_location="cpu")
            indices = list(range(data["l"].shape[0]))
            if self.shuffle:
                random.shuffle(indices)
            for index in indices:
                yield tuple(data[key][index] for key in ("v", "a", "t", "s", "l", "m"))


class StudentLoss(nn.Module):
    def __init__(self, variant, alpha):
        super().__init__()
        self.variant = variant
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits, features, teacher_features, labels, mask):
        valid = mask.reshape(-1) > 0.5
        ce_loss = self.ce(logits.reshape(-1, 2)[valid], labels.reshape(-1)[valid]).mean()
        mse_loss = self.mse(
            features.reshape(-1, 128)[valid], teacher_features.reshape(-1, 128)[valid]
        ).mean()
        if self.variant == "ce_only":
            total = ce_loss
        else:
            total = self.alpha * ce_loss + (1.0 - self.alpha) * mse_loss
        return total, ce_loss, mse_loss


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("ce_mse", "ce_only"), default="ce_mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="CE weight for ce_mse; MSE weight is 1-alpha")
    parser.add_argument("--train-cache", type=Path,
                        default=root / "realtime" / "cached_chunks")
    parser.add_argument("--val-cache", type=Path,
                        default=root / "realtime" / "val_cached_chunks")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    return parser.parse_args(), root


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def validate_last_frame(model, loader, device):
    model.eval()
    scores, labels_out = [], []
    last_frame = 49
    for v, a, _teacher, spatial, labels, mask in tqdm(
        loader, desc="AVA final-frame validation"
    ):
        v, a = v.to(device), a.to(device)
        spatial, labels, mask = spatial.to(device), labels.to(device), mask.to(device)
        logits, _ = model(v, a, spatial, mask)
        valid = mask[:, :, last_frame] > 0.5
        probabilities = F.softmax(logits[:, :, last_frame, :], dim=-1)[..., 1]
        scores.extend(probabilities[valid].cpu().tolist())
        labels_out.extend(labels[:, :, last_frame][valid].cpu().tolist())
    if not labels_out:
        raise RuntimeError("AVA validation produced no valid final-frame labels")
    return float(average_precision_score(labels_out, scores) * 100.0), len(labels_out)


def main():
    args, root = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.variant}_seed_{args.seed}"
    output_dir = args.output_dir or root / "realtime" / "experiments" / run_name
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
    criterion = StudentLoss(args.variant, args.alpha)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    log_path = output_dir / "training_log.csv"
    best_map, best_epoch = float("-inf"), None
    started = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "avg_loss", "avg_ce", "avg_mse", "lr",
                         "ava_final_frame_map", "val_samples", "epoch_seconds"))
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            model.train()
            totals = np.zeros(3, dtype=np.float64)
            batches = 0
            progress = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{args.epochs}")
            for v, a, teacher, spatial, labels, mask in progress:
                v, a, teacher = v.to(device), a.to(device), teacher.to(device)
                spatial, labels, mask = spatial.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model(v, a, spatial, mask)
                loss, ce_loss, mse_loss = criterion(
                    logits, features, teacher, labels, mask
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals += (loss.item(), ce_loss.item(), mse_loss.item())
                batches += 1
                progress.set_postfix(loss=f"{loss.item():.3f}")
            if batches == 0:
                raise RuntimeError("Training cache produced zero batches")
            scheduler.step()
            ava_map, val_samples = validate_last_frame(model, val_loader, device)
            elapsed = time.time() - epoch_started
            averages = totals / batches
            writer.writerow((epoch, *[f"{value:.8f}" for value in averages],
                             f"{scheduler.get_last_lr()[0]:.8f}", f"{ava_map:.6f}",
                             val_samples, f"{elapsed:.3f}"))
            handle.flush()
            torch.save(model.state_dict(), output_dir / "last.pt")
            if ava_map > best_map:
                best_map, best_epoch = ava_map, epoch
                torch.save(model.state_dict(), output_dir / "best_ava.pt")
            print(f"Epoch {epoch}: AVA final-frame mAP={ava_map:.4f}; "
                  f"best={best_map:.4f} at epoch {best_epoch}")

    manifest = {
        "run_name": run_name,
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.val_batch_size,
        "ce_weight": 1.0 if args.variant == "ce_only" else args.alpha,
        "mse_weight": 0.0 if args.variant == "ce_only" else 1.0 - args.alpha,
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
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
