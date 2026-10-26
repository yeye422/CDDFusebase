"""Paired noisy/reference SAR dataset for standalone SPD training."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def _image_map(directory):
    directory = Path(directory)
    paths = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(directory.glob(pattern))
    return {path.stem: path for path in paths}


def paired_sar_paths(noisy_dir, reference_dir):
    noisy = _image_map(noisy_dir)
    reference = _image_map(reference_dir)
    common = sorted(
        set(noisy) & set(reference),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if not common:
        raise RuntimeError("No paired noisy/reference SAR images were found")
    if len(common) != len(noisy) or len(common) != len(reference):
        raise RuntimeError(
            "Noisy/reference directories must contain exactly matching stems: "
            f"noisy={len(noisy)}, reference={len(reference)}, paired={len(common)}"
        )
    return [(noisy[stem], reference[stem]) for stem in common]


class SARDenoisingDataset(Dataset):
    def __init__(
        self,
        pairs,
        patch_size=128,
        training=False,
        line_mask_dir=None,
        point_mask_dir=None,
        reference_mask_dir=None,
        invalid_threshold=-1,
        min_valid_fraction=0.90,
        crop_attempts=12,
        augment=True,
        limit=None,
    ):
        self.pairs = list(pairs[:limit] if limit else pairs)
        if not self.pairs:
            raise RuntimeError("SPD dataset is empty")
        self.patch_size = int(patch_size)
        self.training = bool(training)
        self.invalid_threshold = int(invalid_threshold)
        self.min_valid_fraction = float(min_valid_fraction)
        self.crop_attempts = int(crop_attempts)
        self.augment = bool(augment and training)
        self.line_masks = _image_map(line_mask_dir) if line_mask_dir else {}
        self.point_masks = _image_map(point_mask_dir) if point_mask_dir else {}
        self.reference_masks = (
            _image_map(reference_mask_dir) if reference_mask_dir else {}
        )
        pair_stems = {noisy.stem for noisy, _ in self.pairs}
        for label, mapping in (
            ("line", self.line_masks),
            ("point", self.point_masks),
            ("reference-quality", self.reference_masks),
        ):
            if mapping:
                missing = sorted(pair_stems - set(mapping))
                if missing:
                    raise FileNotFoundError(
                        f"{label} masks are missing {len(missing)} paired "
                        f"images, including {missing[:5]}"
                    )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read_gray(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        if image.ndim == 3:
            first = image[:, :, 0]
            if not np.all(image == first[:, :, None]):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = first
        if image.dtype != np.uint8:
            raise TypeError(
                f"SPD currently expects uint8 display-domain images; "
                f"{path} has dtype {image.dtype}"
            )
        return image

    def _optional_mask(self, mapping, stem, shape, default):
        if not mapping:
            return np.full(shape, default, dtype=np.float32)
        path = mapping.get(stem)
        if path is None:
            raise FileNotFoundError(f"Mask for stem {stem} is missing")
        mask = self._read_gray(path)
        if mask.shape != shape:
            raise RuntimeError(
                f"Mask shape mismatch for {stem}: {mask.shape} versus {shape}"
            )
        return mask.astype(np.float32) / 255.0

    def _choose_crop(self, valid):
        height, width = valid.shape
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
            fraction = float(valid[top : top + size, left : left + size].mean())
            if best is None or fraction > best[0]:
                best = (fraction, top, left)
            if fraction >= self.min_valid_fraction:
                return top, left
        return best[1], best[2]

    def __getitem__(self, index):
        noisy_path, reference_path = self.pairs[index]
        noisy = self._read_gray(noisy_path)
        reference = self._read_gray(reference_path)
        if noisy.shape != reference.shape:
            raise RuntimeError(
                f"Shape mismatch for {noisy_path.stem}: "
                f"noisy={noisy.shape}, reference={reference.shape}"
            )
        stem = noisy_path.stem
        valid = (noisy > self.invalid_threshold) & (
            reference > self.invalid_threshold
        )
        line = self._optional_mask(self.line_masks, stem, noisy.shape, 0.0)
        point = self._optional_mask(self.point_masks, stem, noisy.shape, 0.0)
        reference_quality = self._optional_mask(
            self.reference_masks, stem, noisy.shape, 1.0
        )

        top, left = self._choose_crop(valid)
        size = self.patch_size
        arrays = [noisy, reference, valid, line, point, reference_quality]
        arrays = [
            value[top : top + size, left : left + size] for value in arrays
        ]

        if self.augment:
            if random.random() < 0.5:
                arrays = [np.fliplr(value) for value in arrays]
            if random.random() < 0.5:
                arrays = [np.flipud(value) for value in arrays]
            rotations = random.randint(0, 3)
            if rotations:
                arrays = [np.rot90(value, rotations) for value in arrays]

        noisy, reference, valid, line, point, reference_quality = arrays

        def tensor(value, scale=1.0):
            return (
                torch.from_numpy(np.ascontiguousarray(value))[None].float()
                / scale
            )

        return {
            "noisy": tensor(noisy, 255.0),
            "reference": tensor(reference, 255.0),
            "valid": tensor(valid.astype(np.float32)),
            "line_mask": tensor(line),
            "point_mask": tensor(point),
            "reference_quality": tensor(reference_quality),
            "stem": stem,
        }
