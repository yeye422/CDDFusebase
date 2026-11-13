"""Diagnose the PAC residual on one explicitly selected validation scene."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import evaluate, load_variant_models, sha256_file
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator

DATA = Path("../datasets/QXSLAB_SAROPT")
MANIFEST = DATA / "splits/cddfuse_8000_1000_1000_seed20260924.json"
BASE = Path("models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth")
PAC = Path("models/e2_pac_mamba_v1/seed42/epoch_012.pth")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="4")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve().relative_to(Path.cwd())
    if out.exists():
        raise FileExistsError(out)
    splits, _ = manifest_pairs(DATA, MANIFEST)
    offset, (optical_path, sar_path) = next(
        (i, pair) for i, pair in enumerate(splits["val"]) if pair[0].stem == args.scene_id
    )
    bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or sar is None or bgr.shape[:2] != sar.shape:
        raise ValueError("Invalid paired input")
    optical = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, s = (torch.from_numpy(a.astype(np.float32) / 255)[None, None].to(device) for a in (optical, sar))
    base, _ = load_variant_models(BASE, device, requested="lwpd_e2")
    models, _ = load_variant_models(PAC, device, requested="pac_mamba_e2")
    model = models["pac_mamba"]
    with torch.inference_mode():
        original = base["direct"](x, s)
        on, features = model(x, s, return_features=True)
        pac = features["pac"]
        residual = pac["update_scale"] * pac["update"]
        fused_base = features["fused_base"]
        torch.testing.assert_close(pac["keep"] + residual, features["fused_detail"])
        half = model.decoder(None, fused_base, pac["keep"] + 0.5 * residual)[0]
        off = model.decoder(None, fused_base, pac["keep"])[0]
        tensors = dict(original_e2=original, pac_off=off, pac_half=half, pac_on=on)
        images = {name: np.rint(t.clamp(0, 1).squeeze().cpu().numpy() * 255).astype(np.uint8)
                  for name, t in tensors.items()}
        on_off_mae = float((on - off).abs().mean())
        residual_ratio = float(residual.square().mean().sqrt() / pac["keep"].square().mean().sqrt())
    rows = {}
    for name, img in images.items():
        values = evaluate(img.astype(np.float32), optical.astype(np.float32), sar.astype(np.float32),
                          metric_profile="paper")
        rows[name] = dict(zip(PaperEvaluator.METRIC_NAMES, map(float, values)))

    optical_gradient, _ = PaperEvaluator.Qabf_getArray(optical.astype(np.float64))
    sar_gradient, _ = PaperEvaluator.Qabf_getArray(sar.astype(np.float64))
    upper = np.maximum(optical_gradient, sar_gradient)
    masks = {}
    for name, img in images.items():
        gradient, _ = PaperEvaluator.Qabf_getArray(img.astype(np.float64))
        masks[name] = gradient > upper
        rows[name]["over_source_gradient_fraction"] = float(masks[name].mean())

    out.mkdir(parents=True)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", *rows["pac_on"]])
        writer.writeheader()
        writer.writerows(dict(condition=name, **row) for name, row in rows.items())
    tiles = []
    for name, img in {"optical": optical, "sar": sar, **images}.items():
        tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, name, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    overlay = cv2.cvtColor(images["pac_on"], cv2.COLOR_GRAY2BGR)
    overlay[masks["pac_on"] & ~masks["pac_off"]] = (0, 0, 255)
    overlay[masks["pac_off"] & ~masks["pac_on"]] = (255, 190, 0)
    for name, img in {"overview": np.concatenate(tiles, axis=1), "gradient_threshold_changes": overlay}.items():
        if not cv2.imwrite(str(out / f"{name}.png"), img):
            raise OSError(f"Cannot save {name}")
    summary = {f"{name}_nabf": row["Nabf"] for name, row in rows.items()}
    summary.update(on_off_raw_mae=on_off_mae, applied_residual_rms_ratio=residual_ratio)
    write_json(out / "summary.json", summary)
    write_json(out / "run_metadata.json", dict(
        scene_id=args.scene_id, split="val", offset=offset, evaluated_pairs=1, device=device,
        precision="fp32", metric_profile="paper", output_policy="clamp_then_round_uint8",
        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=sha256_file(MANIFEST),
        checkpoints={path.as_posix(): sha256_file(path) for path in (BASE, PAC)},
        inputs={path.as_posix(): sha256_file(path) for path in (optical_path, sar_path)},
        intervention="Multiply the trained PAC residual by 1, 0.5 or 0 before the same decoder.",
        gradient_overlay="Red: crosses above source maximum when PAC is enabled. Cyan: reverse.",
    ))
    print(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
