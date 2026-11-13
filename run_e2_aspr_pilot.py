"""Paired fixed-ASPR validation and matched QXS train1024/val64 pilot."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from e2_aspr import SPEC, fixed_weights, gradient, solve
from e2_base_anchor import OpticalBaseAnchoredE2
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import FIXED_ASPR_PILOT, ORIGINAL_LOSS_PILOT, train
from run_qxs_reference_test1000 import DATA, read_image, read_json, write_json
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("test_result/e2_aspr_fixed_pilot_v1/val64")
METRICS = FusionEvaluator.METRIC_NAMES
SOURCE_METRICS = ("SSIM_O", "SSIM_S", "MS_SSIM_O", "MS_SSIM_S")
DIAGNOSTICS = ("relative_initial_residual", "correction_mae", "mean_brightness_change",
               "sar_evidence_gradient_change", "ws_mean", "outside_range_fraction")
LABELS = dict(e2="E2", lowpass="E2 + optical lowpass", optical="E2 + optical constraints",
              fixed="E2 + fixed ASPR", shifted="E2 + shifted SAR protection",
              trained_f0="ASPR-trained E2 before PRS", trained="ASPR-trained E2 + PRS")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


def load_model(checkpoint, device, expected_aspr_hash=None):
    saved = torch.load(checkpoint, map_location="cpu")
    args = saved["args"]
    if args["variant"] != OpticalBaseAnchoredE2.VARIANT or args.get("aspr_source_sha256") != expected_aspr_hash:
        raise ValueError(f"Unexpected checkpoint variant or ASPR source: {checkpoint}")
    model = OpticalBaseAnchoredE2().to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


@torch.inference_mode()
def evaluate_scene(pair, models, device, output=None):
    optical_path, sar_path = pair
    optical_color = read_image(optical_path, cv2.IMREAD_COLOR)
    optical_u8 = cv2.cvtColor(optical_color, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sar_u8 = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
    if optical_u8.shape != sar_u8.shape:
        raise ValueError(f"Source shape mismatch: {optical_path.stem}")
    optical, sar = (torch.from_numpy(im.astype(np.float32) / 255)[None, None].to(device)
                    for im in (optical_u8, sar_u8))
    initial = models["e2"](optical, sar)
    if "trained" in models:
        initial_aspr = models["trained"](optical, sar)
    else:
        initial_aspr = initial
    wo, ws = fixed_weights(optical, sar, initial_aspr, (sar > 0).float())
    if output is not None:
        np.savez_compressed(output / "evidence" / f"{optical_path.stem}.npz",
                            optical_weight=wo[0, 0].cpu().numpy(), sar_weight=ws[0, 0].cpu().numpy())
    full, residual = solve(initial_aspr, optical, wo, ws, diagnostics=True)
    variants = {"e2": (initial, initial, 0.0)}
    if "trained" in models:
        variants.update(trained_f0=(initial_aspr, initial_aspr, 0.0), trained=(full, initial_aspr, float(residual)))
    else:
        zero = torch.zeros_like(wo)
        # Move the map by half the image in each direction. Keep its values and spatial structure.
        shift = tuple(size // 2 for size in ws.shape[-2:])
        for key, w_opt, w_sar in (("lowpass", zero, zero), ("optical", wo, zero),
                                  ("shifted", wo, ws.roll(shift, dims=(-2, -1)))):
            image, res = solve(initial, optical, w_opt, w_sar, diagnostics=True)
            variants[key] = (image, initial, float(res))
        variants["fixed"] = (full, initial, float(residual))

    result = {}
    for key, (image, anchor, residual) in variants.items():
        if not torch.isfinite(image).all() or not np.isfinite(residual):
            raise FloatingPointError(f"Non-finite ASPR output for {optical_path.stem}/{key}")
        fused = image[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
        if output is not None and not cv2.imwrite(str(output / key / "gray" / f"{optical_path.stem}.png"), fused):
            raise OSError(f"Cannot save {key}/{optical_path.name}")
        values = FusionEvaluator.metric_vector(fused, optical_u8, sar_u8)
        source_values = [metric(fused, source)
                         for metric in (FusionEvaluator._ssim_pair, FusionEvaluator._ms_ssim_pair)
                         for source in (optical_u8, sar_u8)]
        change = image - anchor
        sar_region_change = (ws * gradient(change).abs().mean(dim=1, keepdim=True)).sum()
        diagnostics = (residual, float(change.abs().mean()), float(change.mean()),
                       float(sar_region_change / ws.sum().clamp_min(1e-8)), float(ws.mean()),
                       float(((image < 0) | (image > 1)).float().mean()))
        result[key] = (values, source_values, diagnostics)
    return result


def summarize(output, scene_ids, results, contrasts, diagnostic_fields=DIAGNOSTICS,
              source_metrics=SOURCE_METRICS):
    names = (*METRICS, *source_metrics)
    values = {key: np.asarray([np.r_[row[0], row[1]] for row in rows]) for key, rows in results.items()}
    common = np.isfinite(np.stack(list(values.values()))).all(axis=0)
    summaries = {}
    for key, rows in results.items():
        folder = output / key
        means, stds = [], []
        for col, mask in enumerate(common.T):
            valid = values[key][mask, col]
            means.append(float(valid.mean()) if valid.size else None)
            stds.append(float(valid.std()) if valid.size else None)
        summaries[key] = {name: value for name, value in zip(METRICS, means) if value is not None}
        write_csv(folder / "metrics.csv", ("image", *METRICS), (
            *((scene, *row[0]) for scene, row in zip(scene_ids, rows)),
            ("mean", *means[:len(METRICS)]), ("std", *stds[:len(METRICS)]),
        ))
        if source_metrics or diagnostic_fields:
            write_csv(folder / "diagnostics.csv", ("image", *source_metrics, *diagnostic_fields),
                      ((scene, *row[1], *row[2]) for scene, row in zip(scene_ids, rows)))
    write_json(output / "excluded_scene_ids.json", {
        name: [scene for scene, keep in zip(scene_ids, mask) if not keep]
        for name, mask in zip(names, common.T)
    })
    rng = np.random.default_rng(42)
    paired = []
    for control, candidate in contrasts:
        for col, name in enumerate(names):
            before, after = (values[key][common[:, col], col] for key in (control, candidate))
            delta = after - before
            count = delta.size
            low, high = (np.quantile(delta[rng.integers(count, size=(2000, count))].mean(axis=1), [.025, .975])
                         if count > 1 else (None, None))
            row = ((before.mean(), after.mean(), delta.mean()) if count else (None, None, None))
            paired.append((control, candidate, name, count, *row, low, high))
    write_csv(output / "paired_comparison.csv", ("control", "candidate", "metric", "common_count",
              "control_mean", "candidate_mean", "candidate_minus_control", "ci_low", "ci_high"), paired)
    return summaries


def validate(mode):
    checkpoint = ORIGINAL_LOSS_PILOT.final
    manifest = ORIGINAL_LOSS_PILOT.manifest
    source_hash = sha256_file(Path(__file__).with_name("e2_aspr.py"))
    runner_hash = sha256_file(Path(__file__))
    trained = mode == "trained-val"
    checkpoints = {"e2": checkpoint}
    if trained:
        checkpoints["trained"] = FIXED_ASPR_PILOT.final
    checkpoint_hashes = {key: sha256_file(path) for key, path in checkpoints.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, _ = manifest_pairs(DATA, manifest)
    if (len(splits["train"]), len(splits["val"]), len(splits["test"])) != (1024, 64, 0):
        raise ValueError("Expected the locked train1024/val64 pilot manifest")
    selected = splits["val"]
    output = OUTPUT / mode
    group = f"QXS-E2-ASPR-PILOT-V1-{mode.upper()}64-REFERENCE-FMI"
    scene_ids = [pair[0].stem for pair in selected]
    metadata = dict(mode=mode, device=str(device), scene_ids=scene_ids, aspr=SPEC,
                    aspr_source_sha256=source_hash, evaluator_sha256=runner_hash,
                    split_manifest=manifest.as_posix(), split_manifest_sha256=sha256_file(manifest),
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8",
                    checkpoints={key: path.as_posix() for key, path in checkpoints.items()},
                    checkpoint_hashes=checkpoint_hashes)
    metadata = json.loads(json.dumps(metadata))
    if output.exists():
        if read_json(output / "run_metadata.json") != metadata:
            raise FileExistsError(f"Preserve output with a different protocol: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        models = {key: load_model(path, device, source_hash if key == "trained" else None)
                  for key, path in checkpoints.items()}
        if mode == "smoke":
            for key, row in evaluate_scene(selected[0], models, device).items():
                print(f"{scene_ids[0]} {key}: SSIM={row[0][METRICS.index('SSIM')]:.6f} residual={row[2][0]:.3g}")
            return
        keys = ("e2", "trained_f0", "trained") if trained else ("e2", "lowpass", "optical", "shifted", "fixed")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{mode}-", dir=output.parent) as temporary:
            work = Path(temporary) / "results"
            for key in keys:
                (work / key / "gray").mkdir(parents=True)
            (work / "evidence").mkdir()
            results = {key: [] for key in keys}
            for index, pair in enumerate(selected, 1):
                for key, row in evaluate_scene(pair, models, device, work).items():
                    results[key].append(row)
                print(f"{mode} {index}/{len(selected)} {pair[0].stem}", flush=True)
            contrasts = (("e2", "trained"), ("trained_f0", "trained")) if trained else (
                ("e2", "lowpass"), ("lowpass", "optical"), ("optical", "fixed"),
                ("shifted", "fixed"), ("lowpass", "fixed"), ("e2", "fixed"),
            )
            summaries = summarize(work, scene_ids, results, contrasts)
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "registry_metrics.json", summaries)
            work.rename(output)
    entries = []
    for key, metrics in summaries.items():
        origin = "trained" if key.startswith("trained") else "e2"
        folder = output / key
        entries.append(dict(
            id=f"{group}-{key}", method=LABELS[key], dataset="QXS-SAROPT", task="fusion",
            status="diagnostic_fixed_subset", comparable_group=group, split="val", offset=0,
            sample_count=len(scene_ids), split_manifest=manifest.as_posix(),
            split_manifest_sha256=metadata["split_manifest_sha256"], checkpoint=checkpoints[origin].as_posix(),
            checkpoint_sha256=checkpoint_hashes[origin], result_dir=folder.as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(), metric_profile=DEFAULT_PROFILE,
            output_policy="clamp_then_uint8", metrics=metrics,
            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in metrics}),
        ))
    register_many(entries)
    print(f"Paired comparison: {output / 'paired_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "fixed-val", "train", "trained-val"))
    action = parser.parse_args().action
    os.chdir(ROOT)
    if action == "train":
        train(FIXED_ASPR_PILOT)
    else:
        validate(action)


if __name__ == "__main__":
    main()
