"""D0: non-training C0 feature decomposition and stability audit.

This is a diagnostic, not a speckle-identification algorithm.
All models use identical validation images and unchanged checkpoints.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import load_variant_models
from utils.dataset_os import manifest_pairs


def pearson_by_channel(a, b):
    x = a.float().flatten(2)
    y = b.float().flatten(2)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    covariance = (x * y).sum(-1)
    denom = x.square().sum(-1).sqrt() * y.square().sum(-1).sqrt()
    valid = denom > 1e-8
    corr = torch.where(valid, covariance / denom.clamp_min(1e-8), 0)
    return corr.clamp(-1, 1), valid


def high_frequency_fraction(feature, frequency_cutoff=0.25):
    # Radial frequency in cycles/pixel. Mean removed, avoiding DC domination.
    x = feature.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    h, w = x.shape[-2:]
    ky = torch.fft.fftfreq(h, device=x.device).reshape(h, 1)
    kx = torch.fft.fftfreq(w, device=x.device).reshape(1, w)
    radial = torch.sqrt(ky.square() + kx.square())
    energy = torch.fft.fft2(x, dim=(-2, -1)).abs().square()
    return (
        energy[..., radial >= frequency_cutoff].sum()
        / energy.sum().clamp_min(1e-10)
    ).item()


def relative_change(x, ref):
    d = (x.float() - ref.float()).square().sum().sqrt()
    norm = ref.float().square().sum().sqrt().clamp_min(1e-8)
    cos = torch.nn.functional.cosine_similarity(
        x.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0),
    ).item()
    return float(d / norm), float(cos)


def extract(models, variant, optical, sar):
    encoder = models["encoder"]
    if variant == "e1_dual_stem":
        ob, od, os = encoder(optical, "optical")
        sb, sd, ss = encoder(sar, "sar")
    else:
        ob, od, os = encoder(optical)
        sb, sd, ss = encoder(sar)
    fb = models["base_fuse"](ob + sb)
    fd = models["detail_fuse"](od + sd)
    fused, _ = models["decoder"](optical, fb, fd)
    optical_hat, _ = models["decoder"](optical, ob, od)
    sar_hat, _ = models["decoder"](sar, sb, sd)
    return {
        "O_Shallow": os, "S_Shallow": ss,
        "O_Base": ob, "S_Base": sb,
        "O_Detail": od, "S_Detail": sd,
        "F_Base": fb, "F_Detail": fd,
        "Fused": fused,
        "O_Recon": optical_hat, "S_Recon": sar_hat,
    }


def read_pair(optical_path, sar_path, device):
    bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    gray_s = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or gray_s is None:
        raise RuntimeError(f"Cannot read paired images: {optical_path}, {sar_path}")
    gray_o = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if gray_o.shape != gray_s.shape:
        raise ValueError(f"Optical/SAR shape mismatch: {optical_path.stem}")
    o = torch.from_numpy(gray_o.copy())[None, None].to(device).float() / 255.0
    s = torch.from_numpy(gray_s.copy())[None, None].to(device).float() / 255.0
    return o, s


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1 or args.noise_std < 0:
        parser.error("--limit must be positive and --noise-std nonnegative")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_pairs, _ = manifest_pairs(args.data_root, args.split_manifest)
    selected = split_pairs[args.split][: args.limit]
    if not selected:
        raise RuntimeError("Split is empty")
    models = {}
    checksums = {}
    for item in args.model:
        name, sep, filename = item.partition("=")
        if not sep or not name or not filename or name in models:
            parser.error(f"Use unique NAME=PATH arguments; invalid {item!r}")
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        models[name] = load_variant_models(path, device)
        checksums[name] = digest(path)
    if "C0" not in models:
        parser.error("--model must include C0=path")
    if models["C0"][1] != "frequency_minimax_baseline":
        raise ValueError("The C0 reference checkpoint must be frequency_minimax_baseline")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    channel_rows = []
    with torch.no_grad():
        for i, (optical_path, sar_path) in enumerate(selected, start=1):
            optical, sar = read_pair(optical_path, sar_path, device)
            # One perturbation is reused across models for a paired comparison.
            rng = torch.Generator(device="cpu").manual_seed(args.seed + i)
            noise = torch.randn(sar.shape, generator=rng).to(device) * args.noise_std
            sar_pert = (sar + noise * (sar > 0).float()).clamp(0, 1)

            reference, _ = models["C0"]
            c0 = extract(reference, "frequency_minimax_baseline", optical, sar)
            for name, (model, variant) in models.items():
                if variant not in ("frequency_minimax_baseline", "e1_dual_stem"):
                    raise ValueError(f"D0 does not support adapted fusion paths: {name}")
                f = c0 if name == "C0" else extract(model, variant, optical, sar)
                f_pert = extract(model, variant, optical, sar_pert)
                row = {
                    "image": optical_path.stem,
                    "model": name,
                    "c0_checkpoint_sha256": checksums["C0"],
                    "checkpoint_sha256": checksums[name],
                    "o_recon_mae": float((f["O_Recon"] - optical).abs().mean()),
                    "s_recon_mae": float((f["S_Recon"] - sar).abs().mean()),
                    "o_recon_mse": float((f["O_Recon"] - optical).square().mean()),
                    "s_recon_mse": float((f["S_Recon"] - sar).square().mean()),
                    "sar_input_perturb_rel": relative_change(sar_pert, sar)[0],
                }
                for branch in ("Base", "Detail"):
                    correlation, valid = pearson_by_channel(
                        f["O_" + branch], f["S_" + branch]
                    )
                    usable = correlation[valid]
                    row[f"{branch.lower()}_cc_mean"] = float(
                        usable.mean() if usable.numel() else torch.tensor(float("nan"))
                    )
                    row[f"{branch.lower()}_cc_abs_mean"] = float(
                        usable.abs().mean() if usable.numel() else torch.tensor(float("nan"))
                    )
                    row[f"{branch.lower()}_cc_valid_fraction"] = float(
                        valid.float().mean()
                    )
                    for channel, (value, is_valid) in enumerate(
                        zip(correlation.flatten().tolist(), valid.flatten().tolist())
                    ):
                        channel_rows.append({
                            "image": optical_path.stem, "model": name,
                            "branch": branch, "channel": channel,
                            "correlation": value, "valid": int(is_valid),
                        })
                for key in (
                    "O_Shallow", "S_Shallow", "O_Base", "S_Base",
                    "O_Detail", "S_Detail", "F_Base", "F_Detail"
                ):
                    row[f"{key}_highfreq_frac"] = high_frequency_fraction(f[key])
                    row[f"{key}_rms"] = float(f[key].float().square().mean().sqrt())
                    drift, cosine = relative_change(f[key], c0[key])
                    row[f"{key}_drift_rel"] = drift
                    row[f"{key}_c0_cosine"] = cosine
                for key in ("S_Shallow", "S_Base", "S_Detail", "F_Base", "F_Detail", "Fused"):
                    sensitivity, _ = relative_change(f_pert[key], f[key])
                    row[f"{key}_perturb_rel"] = sensitivity
                row["fused_c0_drift_rel"] = relative_change(f["Fused"], c0["Fused"])[0]
                rows.append(row)
            print(f"processed {i}/{len(selected)}: {optical_path.name}", flush=True)

    write_csv(args.output_dir / "per_image.csv", rows)
    write_csv(args.output_dir / "channel_correlations.csv", channel_rows)
    metric_keys = [
        key for key in rows[0]
        if key not in ("image", "model", "c0_checkpoint_sha256", "checkpoint_sha256")
    ]
    summary = []
    for name in models:
        subset = [row for row in rows if row["model"] == name]
        summary.append({
            "model": name,
            "samples": len(subset),
            **{k: float(np.nanmean([float(row[k]) for row in subset])) for k in metric_keys},
        })
    write_csv(args.output_dir / "summary.csv", summary)
    metadata = {
        "data_root": str(args.data_root),
        "split": args.split,
        "split_manifest_sha256": digest(args.split_manifest),
        "noise_std": args.noise_std,
        "noise_model": "additive Gaussian on normalized 8-bit SAR; not physical speckle",
        "seed": args.seed,
        "samples": len(selected),
        "image_ids": [optical.stem for optical, _ in selected],
        "checkpoint_sha256": checksums,
        "highfreq_cutoff_cycles_per_pixel": 0.25,
        "feature_drift_reference": "C0 on identical image pairs",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("D0 completed:", args.output_dir.resolve())
    for row in summary:
        print(
            row["model"],
            "BaseCC=", round(row["base_cc_mean"], 4),
            "DetailCC=", round(row["detail_cc_mean"], 4),
            "SDetailDrift=", round(row["S_Detail_drift_rel"], 4),
            "SDetailNoise=", round(row["S_Detail_perturb_rel"], 4),
            "SReconMAE=", round(row["s_recon_mae"], 4),
        )


if __name__ == "__main__":
    main()
