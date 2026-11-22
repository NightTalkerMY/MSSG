"""Build reusable AVA tensor caches from the seven feature-CSV families."""

import argparse
import gc
from pathlib import Path
import torch
from HoloPiStudentDataset import HoloPiStudentDataset

def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--predata", type=Path, default=root / "predata")
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-limit", type=int, default=2000)
    parser.add_argument("--buffer-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-speakers", type=int, default=5)
    return parser.parse_args(), root

def main():
    args, root = parse_args()
    if args.label_dir is None:
        label_name = "ava_activespeaker_train_v1.0" if args.split == "train" else "ava_activespeaker_test_v1.0"
        args.label_dir = root / "data" / "AVA" / label_name
    if args.output_dir is None:
        name = "cached_chunks" if args.split == "train" else "val_cached_chunks"
        args.output_dir = root / "realtime" / name
    if not args.label_dir.is_dir():
        raise SystemExit(f"Missing label directory: {args.label_dir}")
    prefix = "train" if args.split == "train" else "val"
    existing = sorted(args.output_dir.glob(f"{prefix}_cache_part_*.pt")) if args.output_dir.is_dir() else []
    if existing:
        raise SystemExit(f"Refusing to mix caches: {len(existing)} existing files in {args.output_dir}")
    dataset = HoloPiStudentDataset(str(args.predata), str(args.label_dir), mode=args.split,
                                   buffer_size=args.buffer_size, stride=args.stride,
                                   max_speakers=args.max_speakers)
    if len(dataset) == 0:
        raise SystemExit("Dataset produced zero windows; check features and labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffers = {key: [] for key in ("v", "a", "t", "s", "l", "m")}
    file_index = 0
    for index in range(len(dataset)):
        for key, value in zip(buffers, dataset[index]):
            buffers[key].append(value)
        if (index + 1) % args.chunk_limit == 0 or index + 1 == len(dataset):
            output = args.output_dir / f"{prefix}_cache_part_{file_index:04d}.pt"
            torch.save({key: torch.stack(value) for key, value in buffers.items()}, output)
            print(f"Saved {output} ({index + 1:,}/{len(dataset):,} windows)")
            buffers = {key: [] for key in buffers}
            file_index += 1
            gc.collect()
    print(f"Completed {args.split} cache: {file_index} files, {len(dataset):,} windows")

if __name__ == "__main__":
    main()
