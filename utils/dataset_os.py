"""Optical-SAR dataset used by the CDDFuse domain-adaptation trainer."""

import random
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def natural_key(path):
    return int(path.stem) if path.stem.isdigit() else path.stem


def digest_splits(splits):
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest_splits(manifest):
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Split manifest is missing the splits object")
    required = ("train", "val", "test")
    if any(name not in splits for name in required):
        raise ValueError("Split manifest must contain train, val, and test lists")
    all_stems = [str(stem) for name in required for stem in splits[name]]
    if len(all_stems) != len(set(all_stems)):
        raise ValueError("Split manifest contains duplicate or overlapping image stems")
    expected = manifest.get("splits_sha256")
    actual = digest_splits({name: splits[name] for name in required})
    if expected and expected != actual:
        raise ValueError(
            f"Split manifest digest mismatch: expected {expected}, computed {actual}"
        )
    return actual


def paired_paths(data_root, optical_dir="GF2", sar_dir="GF3"):
    data_root = Path(data_root)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    optical_paths = []
    sar_paths = []
    for extension in extensions:
        optical_paths.extend((data_root / optical_dir).glob(extension))
        sar_paths.extend((data_root / sar_dir).glob(extension))
    optical = {path.stem: path for path in optical_paths}
    sar = {path.stem: path for path in sar_paths}
    common = sorted(set(optical) & set(sar), key=lambda value: int(value) if value.isdigit() else value)
    if not common:
        raise RuntimeError(
            f"No paired images found in {data_root / optical_dir} and "
            f"{data_root / sar_dir}"
        )
    if len(common) != len(optical) or len(common) != len(sar):
        raise RuntimeError(
            f"Unpaired data: GF2={len(optical)}, GF3={len(sar)}, paired={len(common)}"
        )
    return [(optical[stem], sar[stem]) for stem in common]


def split_pairs(
    data_root,
    test_count=150,
    val_count=300,
    optical_dir="GF2",
    sar_dir="GF3",
):
    pairs = paired_paths(data_root, optical_dir, sar_dir)
    if len(pairs) <= test_count + val_count:
        raise RuntimeError(
            f"Need more than {test_count + val_count} pairs, found {len(pairs)}"
        )
    return {
        "test": pairs[:test_count],
        "val": pairs[test_count : test_count + val_count],
        "train": pairs[test_count + val_count :],
    }


def manifest_pairs(data_root, manifest_path):
    data_root = Path(data_root)
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest_splits(manifest)
    optical_dir = manifest["optical_dir"]
    sar_dir = manifest["sar_dir"]
    available = {
        optical.stem: (optical, data_root / sar_dir / optical.name)
        for optical in (data_root / optical_dir).iterdir()
        if optical.is_file()
    }
    splits = {}
    for split_name in ("train", "val", "test"):
        stems = manifest["splits"][split_name]
        pairs = []
        for stem in stems:
            if stem not in available:
                raise FileNotFoundError(
                    f"Optical image for stem {stem} is missing from {data_root / optical_dir}"
                )
            optical, expected_sar = available[stem]
            if not expected_sar.exists():
                candidates = list((data_root / sar_dir).glob(f"{stem}.*"))
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f"SAR image for stem {stem} is missing or ambiguous"
                    )
                expected_sar = candidates[0]
            pairs.append((optical, expected_sar))
        splits[split_name] = pairs
    return splits, manifest


class OpticalSARDataset(Dataset):
    def __init__(
        self,
        pairs,
        patch_size=128,
        training=False,
        invalid_threshold=0,
        min_valid_fraction=0.90,
        crop_attempts=12,
        augment=True,
        limit=None,
    ):
        self.pairs = list(pairs[:limit] if limit else pairs)
        self.patch_size = patch_size
        self.training = training
        self.invalid_threshold = invalid_threshold
        self.min_valid_fraction = min_valid_fraction
        self.crop_attempts = crop_attempts
        self.augment = augment and training
        if not self.pairs:
            raise RuntimeError("Dataset split is empty")

    def __len__(self):
        return len(self.pairs)

    def _choose_crop(self, mask):
        height, width = mask.shape
        size = self.patch_size
        if size > height or size > width:
            raise RuntimeError(
                f"Patch size {size} exceeds image size {(height, width)}"
            )
        if not self.training:
            return (height - size) // 2, (width - size) // 2

        best = None
        for _ in range(self.crop_attempts):
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            fraction = float(mask[top : top + size, left : left + size].mean())
            if best is None or fraction > best[0]:
                best = (fraction, top, left)
            if fraction >= self.min_valid_fraction:
                return top, left
        return best[1], best[2]

    def __getitem__(self, index):
        optical_path, sar_path = self.pairs[index]
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        if optical_bgr is None or sar_u8 is None:
            raise RuntimeError(f"Could not read pair {optical_path.name}")
        if optical_bgr.shape[:2] != sar_u8.shape:
            raise RuntimeError(
                f"Shape mismatch for {optical_path.name}: "
                f"GF2={optical_bgr.shape[:2]}, GF3={sar_u8.shape}"
            )

        optical_y = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        valid = sar_u8 > self.invalid_threshold
        top, left = self._choose_crop(valid)
        size = self.patch_size
        optical_y = optical_y[top : top + size, left : left + size]
        sar_u8 = sar_u8[top : top + size, left : left + size]
        valid = valid[top : top + size, left : left + size]

        if self.augment:
            if random.random() < 0.5:
                optical_y = np.fliplr(optical_y)
                sar_u8 = np.fliplr(sar_u8)
                valid = np.fliplr(valid)
            if random.random() < 0.5:
                optical_y = np.flipud(optical_y)
                sar_u8 = np.flipud(sar_u8)
                valid = np.flipud(valid)
            rotations = random.randint(0, 3)
            if rotations:
                optical_y = np.rot90(optical_y, rotations)
                sar_u8 = np.rot90(sar_u8, rotations)
                valid = np.rot90(valid, rotations)

        optical = torch.from_numpy(np.ascontiguousarray(optical_y))[None].float() / 255.0
        sar = torch.from_numpy(np.ascontiguousarray(sar_u8))[None].float() / 255.0
        mask = torch.from_numpy(np.ascontiguousarray(valid))[None].float()
        return optical, sar, mask, optical_path.stem
