"""Train RCCR-L2 once and compare epoch 16 with the existing E2 on val1000."""

import argparse
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

import e2_rccr
from e2_rccr import RCCRL2E2
from lwpd import HaarDWT2D
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import BASE_ANCHOR, TOTAL_EPOCHS, Experiment, train
from run_qxs_reference_test1000 import DATA, read_image, read_json, verify_images, write_json
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


EXPERIMENT = Experiment(
    "e2_scratch_rccr_l2_v1", "E2-SCRATCH-RCCR-L2-V1", RCCRL2E2.VARIANT,
    "run_e2_rccr.py", "LWPD E2 + RCCR-L2",
    "从头训练4轮重建和12轮融合，仅学习关系引导的L2门控。",
)
GROUP = "QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI"
METHODS = dict(e2="E2 光学 Base（已有）", f0="RCCR-L2 校正前", fused="RCCR-L2 校正后")
DIAGNOSTICS = ("gate_mean", "correction_rms", "outside_range_fraction",
               "h1_float_max_error", "h2_float_max_error", "h1_uint8_max_error", "h2_uint8_max_error")


def uint8_image(image):
    return np.clip(np.rint(image[0, 0].cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


def highpass_errors(haar, difference):
    (low, *high_1), _ = haar(difference)
    (_, *high_2), _ = haar(low)
    return tuple(float(torch.stack(bands).abs().amax()) for bands in (high_1, high_2))


@torch.inference_mode()
def evaluate_scene(pair, model, haar, archive, device, output=None):
    optical_path, sar_path = pair
    name = optical_path.stem + ".png"
    optical = cv2.cvtColor(read_image(optical_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sar = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
    baseline = read_image(archive / name, cv2.IMREAD_UNCHANGED)
    if baseline.dtype != np.uint8 or baseline.shape != optical.shape or sar.shape != optical.shape:
        raise ValueError(f"Image shape or baseline dtype differs for {name}")
    tensors = [torch.from_numpy(image.astype(np.float32) / 255)[None, None].to(device) for image in (optical, sar)]
    initial, fused, gate = model.forward_pair(*tensors)
    if not torch.isfinite(fused).all() or not torch.isfinite(initial).all():
        raise FloatingPointError(f"Non-finite RCCR output for {name}")
    initial_u8, fused_u8 = uint8_image(initial), uint8_image(fused)
    difference = fused - initial
    quantized_difference = (fused_u8.astype(np.float32) - initial_u8) / 255
    diagnostics = (float(gate.mean()), float(difference.square().mean().sqrt()),
                   float(((fused < 0) | (fused > 1)).float().mean()),
                   *highpass_errors(haar, difference),
                   *highpass_errors(haar, torch.from_numpy(quantized_difference)[None, None].to(device)))
    rows = {}
    for key, image in zip(METHODS, (baseline, initial_u8, fused_u8)):
        if output is not None and key != "e2":
            if not cv2.imwrite(str(output / key / "gray" / name), image):
                raise OSError(f"Cannot save {key}/{name}")
        values = FusionEvaluator.metric_vector(image, optical, sar)
        source_values = [metric(image, source)
                         for metric in (FusionEvaluator._ssim_pair, FusionEvaluator._ms_ssim_pair)
                         for source in (optical, sar)]
        rows[key] = (values, source_values, diagnostics if key == "fused" else (None,) * len(DIAGNOSTICS))
    return rows


def validate():
    output = EXPERIMENT.result
    checkpoints = {"e2": BASE_ANCHOR.final, "f0": EXPERIMENT.final, "fused": EXPERIMENT.final}
    checkpoint_hashes = {key: sha256_file(path) for key, path in checkpoints.items()}
    manifest_hash = sha256_file(EXPERIMENT.manifest)
    source_hash = sha256_file(Path(e2_rccr.__file__))
    splits, _ = manifest_pairs(DATA, EXPERIMENT.manifest)
    pairs = splits["val"]
    scene_ids = [pair[0].stem for pair in pairs]
    archive = BASE_ANCHOR.result / "gray"
    original_meta = BASE_ANCHOR.result / "run_metadata.json"
    baseline_meta = read_json(original_meta)
    expected = dict(checkpoint_sha256=checkpoint_hashes["e2"], split_manifest_sha256=manifest_hash,
                    variant=BASE_ANCHOR.variant, split="val", offset=0, output_range_policy="clamp",
                    selected_scene_ids=scene_ids, evaluated_pairs=EXPERIMENT.val_count)
    if len(pairs) != EXPERIMENT.val_count or any(baseline_meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived E2 validation differs from the fixed RCCR comparison protocol")
    verify_images(archive, scene_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = dict(split="val", offset=0, device=str(device), scene_ids=scene_ids,
                    split_manifest=EXPERIMENT.manifest.as_posix(), split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", rccr=e2_rccr.SPEC,
                    rccr_source_sha256=source_hash, evaluator_sha256=sha256_file(Path(__file__)),
                    baseline_gray=archive.as_posix(), baseline_metadata=original_meta.as_posix(),
                    baseline_metadata_sha256=sha256_file(original_meta), checkpoint_hashes=checkpoint_hashes,
                    checkpoints={key: path.as_posix() for key, path in checkpoints.items()})
    if output.exists():
        if read_json(output / "run_metadata.json") != metadata:
            raise FileExistsError(f"Preserve output with a different protocol: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        saved = torch.load(EXPERIMENT.final, map_location="cpu")
        args = saved["args"]
        if (saved["variant"], saved["epoch"], args.get("rccr_source_sha256"), args.get("rccr")) != (
            EXPERIMENT.variant, TOTAL_EPOCHS - 1, source_hash, e2_rccr.SPEC
        ):
            raise ValueError("Checkpoint differs from the RCCR-L2 protocol")
        model = RCCRL2E2().to(device).eval()
        model.load_state_dict(saved["model"])
        haar = HaarDWT2D().to(device)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rccr-val-", dir=output.parent) as temporary:
            work = Path(temporary) / "results"
            for key in METHODS:
                (work / key).mkdir(parents=True)
                if key != "e2":
                    (work / key / "gray").mkdir()
            results = {key: [] for key in METHODS}
            for index, pair in enumerate(pairs, 1):
                for key, row in evaluate_scene(pair, model, haar, archive, device, work).items():
                    results[key].append(row)
                if index == 1 or index % 25 == 0:
                    print(f"val {index}/{len(pairs)}", flush=True)
            contrasts = (("e2", "f0"), ("e2", "fused"), ("f0", "fused"))
            summaries = summarize(work, scene_ids, results, contrasts, diagnostic_fields=DIAGNOSTICS)
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "registry_metrics.json", summaries)
            work.rename(output)
    entries = []
    for key, metrics in summaries.items():
        folder = output / key
        entries.append(dict(
            id=f"{GROUP}-{key}", method=METHODS[key], dataset="QXS-SAROPT", task="fusion",
            status="development_validation", comparable_group=GROUP, split="val", offset=0,
            sample_count=len(scene_ids), split_manifest=metadata["split_manifest"],
            split_manifest_sha256=manifest_hash, checkpoint=checkpoints[key].as_posix(),
            checkpoint_sha256=checkpoint_hashes[key], result_dir=folder.as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(), metric_profile=DEFAULT_PROFILE,
            output_policy=metadata["output_policy"], metrics=metrics,
            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in metrics}),
        ))
    register_many(entries)
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train-val", "val"))
    action = parser.parse_args().action
    os.chdir(Path(__file__).resolve().parent)
    if action == "train-val":
        train(EXPERIMENT)
    validate()


if __name__ == "__main__":
    main()
