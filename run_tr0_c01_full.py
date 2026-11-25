"""Train the C01 recipe from scratch on QXS-SAROPT train8000."""

import argparse
import os
from pathlib import Path

import torch

from run_tr0_factorial import checkpoint_path, read_json, sha256_file, train
from tr0_evaluation import register_trainings
from utils.dataset_os import manifest_pairs, validate_manifest_splits


ARM = "C01"
RUN_ID = "train8000-seed42"
ROOT = Path("models/tr0_v1_train8000/seed42")
EXPECTED_COUNTS = {"train": 8000, "val": 1000, "test": 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--check", action="store_true", help="Verify the split and CUDA without model execution.")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)

    config = read_json("configs/tr0_v1.json")
    config.update(train_count=EXPECTED_COUNTS["train"], screen_count=EXPECTED_COUNTS["val"])
    source = Path(config["source_manifest"])
    manifest = read_json(source)
    validate_manifest_splits(manifest)
    if {key: len(values) for key, values in manifest["splits"].items()} != EXPECTED_COUNTS:
        raise ValueError(f"Expected the fixed 8000/1000/1000 split: {source}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("C01 requires CUDA with BF16 support")

    locked = ROOT / "screen_manifest.json"
    digest = sha256_file(source)
    if locked.exists() and sha256_file(locked) != digest:
        raise ValueError(f"Existing run uses a different split: {locked}")
    print(f"C01 train8000/val1000, seed {config['seed']}, CUDA BF16; split SHA-256 {digest}", flush=True)
    print(f"Final checkpoint: {checkpoint_path(config, ROOT, ARM)}", flush=True)
    if args.check:
        pairs, _ = manifest_pairs(Path(config["data_root"]), source)
        print(f"Verified source files for {sum(len(split) for split in pairs.values())} paired scenes", flush=True)
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    if not locked.exists():
        locked.write_bytes(source.read_bytes())
    final = train(config, ROOT, ARM, locked, torch.device(args.device), resume=True)
    register_trainings(config, RUN_ID, {ARM: final}, locked)


if __name__ == "__main__":
    main()
