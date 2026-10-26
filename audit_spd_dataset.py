"""Read-only audit for paired SAR despeckling TIFF datasets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")


def image_map(directory):
    paths = []
    for pattern in PATTERNS:
        paths.extend(Path(directory).glob(pattern))
    return {path.stem: path for path in paths}


def read_unchanged(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    return image


def sample_statistics(noisy_paths, reference_paths, sample_count):
    stems = sorted(set(noisy_paths) & set(reference_paths))
    if not stems:
        return []
    indices = np.linspace(0, len(stems) - 1, min(sample_count, len(stems)))
    indices = sorted(set(int(round(value)) for value in indices))
    records = []
    for index in indices:
        stem = stems[index]
        noisy = read_unchanged(noisy_paths[stem])
        reference = read_unchanged(reference_paths[stem])
        if noisy.shape != reference.shape:
            raise RuntimeError(
                f"Shape mismatch for {stem}: {noisy.shape} versus {reference.shape}"
            )
        noisy_float = noisy.astype(np.float64)
        reference_float = reference.astype(np.float64)
        noisy_channel_max_difference = None
        reference_channel_max_difference = None
        if noisy.ndim == 3:
            noisy_channel_max_difference = float(
                np.max(np.abs(noisy_float - noisy_float[:, :, :1]))
            )
        if reference.ndim == 3:
            reference_channel_max_difference = float(
                np.max(
                    np.abs(reference_float - reference_float[:, :, :1])
                )
            )
        denominator = float(np.iinfo(noisy.dtype).max) if np.issubdtype(
            noisy.dtype, np.integer
        ) else 1.0
        records.append(
            {
                "stem": stem,
                "shape": list(noisy.shape),
                "noisy_dtype": str(noisy.dtype),
                "reference_dtype": str(reference.dtype),
                "noisy_min": float(noisy_float.min()),
                "noisy_max": float(noisy_float.max()),
                "noisy_mean": float(noisy_float.mean()),
                "reference_min": float(reference_float.min()),
                "reference_max": float(reference_float.max()),
                "reference_mean": float(reference_float.mean()),
                "noisy_channel_max_difference": noisy_channel_max_difference,
                "reference_channel_max_difference": (
                    reference_channel_max_difference
                ),
                "normalized_mae": float(
                    np.mean(np.abs(noisy_float - reference_float)) / denominator
                ),
                "exact_equal_fraction": float(np.mean(noisy == reference)),
            }
        )
    return records


def coordinate_summary(stems):
    coordinates = []
    for stem in stems:
        fields = stem.split("_")
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            coordinates.append((int(fields[0]), int(fields[1])))
    if len(coordinates) != len(stems):
        return {"coordinate_stems": len(coordinates)}
    xs = sorted({value[0] for value in coordinates})
    ys = sorted({value[1] for value in coordinates})
    x_steps = sorted({b - a for a, b in zip(xs, xs[1:])})
    y_steps = sorted({b - a for a, b in zip(ys, ys[1:])})
    return {
        "coordinate_stems": len(coordinates),
        "unique_x": len(xs),
        "unique_y": len(ys),
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
        "y_min": min(ys) if ys else None,
        "y_max": max(ys) if ys else None,
        "x_steps": x_steps,
        "y_steps": y_steps,
    }


def audit_split(noisy_dir, reference_dir, sample_count):
    noisy = image_map(noisy_dir)
    reference = image_map(reference_dir)
    common = sorted(set(noisy) & set(reference))
    return {
        "noisy_dir": str(Path(noisy_dir).resolve()),
        "reference_dir": str(Path(reference_dir).resolve()),
        "noisy_count": len(noisy),
        "reference_count": len(reference),
        "paired_count": len(common),
        "missing_reference": sorted(set(noisy) - set(reference))[:20],
        "missing_noisy": sorted(set(reference) - set(noisy))[:20],
        "coordinates": coordinate_summary(common),
        "samples": sample_statistics(noisy, reference, sample_count),
        "stems": common,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-noisy", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--val-noisy", required=True)
    parser.add_argument("--val-reference", required=True)
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = audit_split(
        args.train_noisy, args.train_reference, args.sample_count
    )
    val = audit_split(args.val_noisy, args.val_reference, args.sample_count)
    overlap = sorted(set(train.pop("stems")) & set(val.pop("stems")))
    report = {
        "train": train,
        "val": val,
        "train_val_stem_overlap_count": len(overlap),
        "train_val_stem_overlap_examples": overlap[:20],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {
        "train": {
            "noisy": train["noisy_count"],
            "reference": train["reference_count"],
            "paired": train["paired_count"],
            "coordinates": train["coordinates"],
            "samples": train["samples"],
        },
        "val": {
            "noisy": val["noisy_count"],
            "reference": val["reference_count"],
            "paired": val["paired_count"],
            "coordinates": val["coordinates"],
            "samples": val["samples"],
        },
        "train_val_stem_overlap_count": len(overlap),
        "report": str(output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
