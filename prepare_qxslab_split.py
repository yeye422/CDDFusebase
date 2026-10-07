"""Create a reproducible random subset split for QXS-SAROPT."""

import argparse
import csv
import json
import random
from pathlib import Path

from utils.dataset_os import digest_splits, paired_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("../datasets/QXSLAB_SAROPT")
    )
    parser.add_argument("--optical-dir", default="opt_256_oc_0.2")
    parser.add_argument("--sar-dir", default="sar_256_oc_0.2")
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def numeric_key(value):
    return int(value) if value.isdigit() else value


def main():
    args = parse_args()
    pairs = paired_paths(args.data_root, args.optical_dir, args.sar_dir)
    requested = args.train_count + args.val_count + args.test_count
    if requested > len(pairs):
        raise ValueError(f"Requested {requested} pairs, only {len(pairs)} available")
    if min(args.train_count, args.val_count, args.test_count) <= 0:
        raise ValueError("All split counts must be positive")

    stems = [optical.stem for optical, _ in pairs]
    random.Random(args.seed).shuffle(stems)
    train_end = args.train_count
    val_end = train_end + args.val_count
    test_end = val_end + args.test_count
    splits = {
        "train": sorted(stems[:train_end], key=numeric_key),
        "val": sorted(stems[train_end:val_end], key=numeric_key),
        "test": sorted(stems[val_end:test_end], key=numeric_key),
    }

    output = args.output
    if output is None:
        output = (
            args.data_root
            / "splits"
            / (
                f"cddfuse_{args.train_count}_{args.val_count}_{args.test_count}"
                f"_seed{args.seed}.json"
            )
        )
    if output.exists() and not args.force:
        raise FileExistsError(f"Split already exists: {output}; use --force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "QXS-SAROPT",
        "data_root_at_creation": str(args.data_root.resolve()),
        "optical_dir": args.optical_dir,
        "sar_dir": args.sar_dir,
        "seed": args.seed,
        "available_pairs": len(pairs),
        "used_pairs": requested,
        "unused_pairs": len(pairs) - requested,
        "split_strategy": "Random sampling without replacement; split lists sorted numerically after allocation",
        "selection_reason": (
            "CDDFuse has about 1.19M parameters and starts from pretrained IVF weights. "
            "8000 training pairs provide a similar epoch size to the prior 7381-pair "
            "Dongying run while retaining 1000-pair validation and test sets."
        ),
        "references": [
            "https://arxiv.org/abs/2103.08259",
            "https://doi.org/10.3390/rs18070975",
            "https://www.mdpi.com/2072-4292/15/14/3528",
        ],
        "splits_sha256": digest_splits(splits),
        "splits": splits,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "split"))
        for split_name in ("train", "val", "test"):
            writer.writerows((stem, split_name) for stem in splits[split_name])

    print(f"available: {len(pairs)}")
    print(
        f"train/val/test: {len(splits['train'])}/"
        f"{len(splits['val'])}/{len(splits['test'])}"
    )
    print(f"unused: {len(pairs) - requested}")
    print(f"sha256: {manifest['splits_sha256']}")
    print(f"json: {output.resolve()}")
    print(f"csv: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
