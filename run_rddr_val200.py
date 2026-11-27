"""Screen four fixed radiometric mixes on the first 200 locked QXS val scenes."""

import argparse
import csv
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

from e2_cfdr import CFDRE2, VARIANTS, CenterPreservingReconstruction
from hcfdr_comparison import EVALUATOR_FILES, OUTPUT as CF1_ARCHIVE, reuse_metrics
from run_cfdr_frozen import checkpoint_path
from run_e2_aspr_pilot import METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_ipmf_frozen import validation_entries
from run_qxs_reference_test1000 import DATA, MANIFEST, read_image, read_json, write_json
from test_OS import evaluate
from train_direct_fusion import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


PARENT = checkpoint_path("CF1", 4)
OUTPUT = Path("test_result/e2_rddr_fixed_v1/val200/first200")
GROUP = "QXS-RDDR-FIXED-V1-CF1-E04-VAL200-FIRST200-REF-FMI"
MIXES = {"R0": 0.0, "R1": 0.25, "R2": 0.5, "R3": 1.0}
METHODS = {"R0": "CF1 第 4 轮", "R1": "温和协调", "R2": "中等协调", "R3": "完全协调"}
CONTRASTS = [("R0", arm) for arm in MIXES if arm != "R0"]
STRUCTURE = {"SSIM": .003, "MS_SSIM": .003}
WEAK = {"CC": .003, "SCD": .003, "PSNR": .1}
RETENTION = {"MI": .030, "FMI": .005, "Qabf": .002, "VIF": .003}


class FixedRadiometricReadout(CenterPreservingReconstruction):
    def __init__(self):
        super().__init__(dynamic=False)

    def components(self, prediction, hidden, fused_detail, optical, sar):
        context = self.condition(torch.cat((hidden, fused_detail), dim=1))
        gain = self.gain(context).softmax(dim=1, dtype=torch.float32)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            inputs = torch.cat((prediction.float(), optical.float(), sar.float()), dim=1)
            base, means = self.smooth(inputs).split((1, 2), dim=1)
            detail = (gain[:, :2] * (inputs[:, 1:] - means)).sum(dim=1, keepdim=True)
            target = means.mean(dim=1, keepdim=True)
        return base, target, detail

    def forward(self, prediction, hidden, fused_detail, optical, sar, *, return_stats=False):
        base, target, detail = self.components(prediction, hidden, fused_detail, optical, sar)
        return {
            "R0": base + detail,
            "R1": torch.lerp(base, target, MIXES["R1"]) + detail,
            "R2": torch.lerp(base, target, MIXES["R2"]) + detail,
            "R3": target + detail,
        }


def load_model(device):
    saved = torch.load(PARENT, map_location="cpu", weights_only=True)
    if (saved["arm"], saved["epoch"], saved["variant"], saved["config_sha256"]) != (
        "CF1", 4, VARIANTS["CF1"], sha256_file(PARENT.parent / "config.json"),
    ):
        raise ValueError(f"Unexpected CF1 checkpoint: {PARENT}")
    model = CFDRE2("CF1")
    model.cfdr = FixedRadiometricReadout()
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)


def archived_r0(all_ids, checkpoint_digest, manifest_digest):
    raw = CF1_ARCHIVE / "CF1" / "evaluator_metrics.csv"
    if not raw.is_file():
        return None, None
    metadata = reuse_metrics(CF1_ARCHIVE, all_ids, manifest_digest, {"CF1": checkpoint_digest}, 4)
    provenance = dict(metrics=raw.as_posix(), metrics_sha256=sha256_file(raw),
                      metadata=metadata.as_posix(), metadata_sha256=sha256_file(metadata))
    return read_metric_rows(raw, all_ids), provenance


def quantize(image):
    return np.rint(image[0, 0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)


@torch.inference_mode()
def evaluate_pairs(model, pairs, device, archived):
    results = {arm: [] for arm in MIXES}
    for index, (optical_path, sar_path) in enumerate(pairs):
        optical = cv2.cvtColor(read_image(optical_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)[:, :, 0]
        sar = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
        if optical.shape != sar.shape:
            raise ValueError(f"Source shape mismatch: {optical_path.stem}")
        inputs = [torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)
                  for image in (optical, sar)]
        outputs = model(*inputs)
        if index == 0:
            original = CFDRE2("CF1").to(device).eval()
            original.load_state_dict(model.state_dict())
            torch.testing.assert_close(outputs["R0"], original(*inputs), rtol=0, atol=0)
        for arm, image in outputs.items():
            if not torch.isfinite(image).all():
                raise FloatingPointError(f"Non-finite image: {optical_path.stem}/{arm}")
            if arm == "R0" and archived is not None and index != 0:
                results[arm].append(archived[index])
                continue
            values = evaluate(quantize(image), optical, sar, DEFAULT_PROFILE)
            if arm == "R0" and archived is not None:
                np.testing.assert_allclose(values, archived[index][0], rtol=1e-7, atol=1e-9)
                results[arm].append(archived[index])
            else:
                results[arm].append((values, (), ()))
        if index == 0 or (index + 1) % 20 == 0:
            print(f"Evaluated {index + 1}/{len(pairs)} scenes", flush=True)
    return results


def write_screening(folder):
    with (folder / "paired_comparison.csv").open(newline="", encoding="utf-8-sig") as file:
        rows = {(row["candidate"], row["metric"]): row for row in csv.DictReader(file)}

    def matched(arm, thresholds):
        return [name for name, floor in thresholds.items()
                if (row := rows[arm, name])["ci_low"] and float(row["ci_low"]) > 0
                and float(row["candidate_minus_control"]) >= floor]

    screening = {}
    for _, arm in CONTRASTS:
        structure, weak = matched(arm, STRUCTURE), matched(arm, WEAK)
        losses = [name for name, limit in RETENTION.items()
                  if (value := rows[arm, name]["candidate_minus_control"]) and float(value) < -limit]
        unknown = [name for name in (*RETENTION, "Nabf")
                   if not rows[arm, name]["candidate_minus_control"]]
        nabf = rows[arm, "Nabf"]["candidate_minus_control"]
        screening[arm] = dict(structure_improved=structure, weak_improved=weak, cf1_retention_flags=losses,
                              unavailable_metrics=unknown, nabf_delta=float(nabf) if nabf else None,
                              positive_trend=bool(structure) and len(weak) >= 2 and not losses and not unknown)
    write_json(folder / "screening.json", dict(
        structure_min_deltas=STRUCTURE, weak_min_deltas=WEAK, cf1_retention_tolerances=RETENTION,
        external_advantage="not_assessed", candidates=screening,
    ))


def run(device, smoke):
    splits, manifest = manifest_pairs(DATA, MANIFEST)
    all_ids = manifest["splits"]["val"]
    if manifest["seed"] != 20260924 or len(all_ids) != 1000:
        raise ValueError("Expected the locked QXS val1000 manifest")
    count = 1 if smoke else 200
    ids = all_ids[:count]
    checkpoint_digest, manifest_digest = sha256_file(PARENT), sha256_file(MANIFEST)
    archived, provenance = archived_r0(all_ids, checkpoint_digest, manifest_digest)
    if smoke:
        results = evaluate_pairs(load_model(device), splits["val"][:count], device, archived)
        assert all(len(rows[0][0]) == len(METRICS) == 14 for rows in results.values())
        print("Smoke passed: exact CF1 R0 parity, four outputs, 14 metrics each.", flush=True)
        return

    sources = (*EVALUATOR_FILES, "e2_cfdr.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py",
               "utils/dataset_os.py", "run_e2_aspr_pilot.py", Path(__file__).name)
    protocol = dict(
        split="val", offset=0, scene_ids=ids, selection="first_200_in_locked_val_manifest_order",
        device=str(device), checkpoints=dict.fromkeys(MIXES, PARENT.as_posix()),
        checkpoint_hashes=dict.fromkeys(MIXES, checkpoint_digest), split_manifest=MANIFEST.as_posix(),
        split_manifest_sha256=manifest_digest, metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8",
        metrics_only=True, fixed_mix=MIXES, target="mean_of_cf1_smoothed_optical_and_sar", cached_r0=provenance,
        source_sha256={name: sha256_file(Path(name)) for name in sources},
    )
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous RDDR results: {OUTPUT}")
    else:
        work = OUTPUT.with_name(f".{OUTPUT.name}-pending-{uuid4().hex}")
        work.mkdir(parents=True)
        for arm in MIXES:
            (work / arm).mkdir()
        results = evaluate_pairs(load_model(device), splits["val"][:count], device, archived)
        summaries = summarize(work, ids, results, CONTRASTS, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        write_screening(work)
        work.rename(OUTPUT)
    register_many(validation_entries(OUTPUT, GROUP, protocol, METHODS,
                                      "固定辐射协调免训练筛选，四组共用 CF1 原位细节及共同有限样本。"))
    print(f"RDDR screening: {OUTPUT / 'screening.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "val200"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDDR validation")
    run(torch.device(args.device), args.stage == "smoke")


if __name__ == "__main__":
    main()
