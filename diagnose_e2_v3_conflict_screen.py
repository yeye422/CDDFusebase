"""Apply the fixed V3 training weights to the archived single-scene V2 edit test."""
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_render import (
    CENTER, ROOT, boundary, endpoints, pixel_grid, sample_pixels, sampling_grid, source_contrast, write_csv,
)
from diagnose_e2_v2_polarity_transition import OUTPUT as PRIOR, DOSES, EPOCHS, SCENE, symmetric_step
from e2_local_correction_v3 import LocalFusionObjective, RULE, SPEC
from maintain_experiment_index import sha256_file
from run_e2_local_correction_v2 import NEGATIVE, PILOT as V2
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json, write_json
from train_direct_fusion import DirectFusionLosses
from utils.fusion_metrics import DEFAULT_PROFILE


OUTPUT = Path("test_result/e2_local_correction_v3_screen_v1/scene4880_B")
LINES = Path("test_result/e2_blind_lines_v1/remaining7_hough/selection_4880.json")


@torch.inference_mode()
def main():
    os.chdir(ROOT)
    if OUTPUT.exists():
        raise FileExistsError(f"Saved screen: {OUTPUT}")
    prior = read_json(PRIOR / "run_metadata.json")
    manifest = read_json(V2.manifest)
    pair = locate_scene(DATA, manifest, SCENE)
    if [sha256_file(p) for p in pair] != prior["input_hashes"] or sha256_file(V2.manifest) != prior["manifest_sha256"]:
        raise ValueError("V2 input mismatch")
    optical, sar = read_pair(pair)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o, s = (to_tensor(im, device) for im in (optical, sar))
    criterion = DirectFusionLosses().to(device)
    objective = LocalFusionObjective(criterion, o, s, s > 0)
    weight, mask = objective.weight, objective.mask
    mask_u8 = mask[0, 0].byte().cpu().numpy()
    target = endpoints(prior["probe"])
    guards = read_json(NEGATIVE)["decisions"]["4880-B"]
    ordinary = next(line for line in read_json(LINES)["optical"] if line["contrast"] * line["other_contrast"] > 0)
    regions = dict(target=target, parallel_negative=target + guards["false_translation"],
                   same_sign=np.array(ordinary["xy"]), sar_response=np.array(guards["sar_segment"]))
    grid = pixel_grid(optical.shape)
    coverage = []
    for name, segment in regions.items():
        _, coords = sampling_grid(segment)
        start, end = torch.as_tensor(segment, dtype=grid.dtype)
        step, taper, _ = boundary(grid, start, end, prior["sign"])
        region = taper.numpy() > 0
        coverage.append(dict(probe=name, segment=segment.tolist(), support_pixels=int(region.sum()),
                             optical_contrast=source_contrast(optical, coords, False),
                             sar_contrast=source_contrast(sar, coords, True),
                             mask_fraction=float(mask_u8[region].mean()),
                             profile_mask_fraction=float(sample_pixels(mask_u8.astype(np.float32), coords[:, CENTER]).mean())))
        if name == "target":
            shape = step.to(device)[None, None]
            support = (taper > 0).to(device)[None, None]

    fs = {}
    model = V2.build_model().to(device).eval()
    for epoch in EPOCHS:
        ck = V2.run_dir / f"epoch_{epoch:03d}.pth"
        if sha256_file(ck) != prior["checkpoint_hashes"][ck.as_posix()]:
            raise ValueError(f"Changed V2 checkpoint: {ck}")
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        final, coarse, _, _ = model.corrected(o, s, model.encode_pair(o, s))
        fs[epoch] = {"initial": coarse, "final": final}
    cap = torch.stack([torch.minimum(im, 1-im) for frames in fs.values() for im in frames.values()]).amin(0)
    with (PRIOR / "matched_edits.csv").open(encoding="utf-8-sig") as f:
        original = {(int(r["epoch"]), r["method"], float(r["dose_gray"]), r["direction"]): r for r in csv.DictReader(f)}
    outputs = []
    gradient_weight, alpha = prior["gradient_weight"], SPEC["coarse_fusion_weight"]
    for epoch, frames in fs.items():
        for dose in DOSES:
            step = symmetric_step(shape, support, cap, dose)
            gi = {}
            ii = {}
            for name, im in frames.items():
                ii0 = (im - objective.intensity_target).abs()
                gi0 = (objective.sobel(im) - objective.gradient_target).abs()
                for key, direction in (("optical", 1), ("reverse", -1)):
                    edited = im + direction * step
                    di = ((edited - objective.intensity_target).abs() - ii0).double()
                    dg = ((objective.sobel(edited) - objective.gradient_target).abs() - gi0).double() * gradient_weight
                    gi[name, key], ii[name, key] = dg, di
                    before, after = (objective.fusion(o, s, x, s > 0, gradient_weight)[0] for x in (im, edited))
                    diff = (di + dg * weight).mean().item()
                    if abs(diff - (after-before).item()) > 8 * torch.finfo(before.dtype).eps * max(abs(before.item()), abs(after.item())):
                        raise ValueError("V3 training loss and spatial diagnostic disagree")
                    ref = original[epoch, name, dose, key]
                    if abs((di + dg).mean().item() - float(ref["delta_total"])) > 1e-7:
                        raise ValueError("Original V2 edit did not reproduce")
            def combined(values, key):
                return alpha * values["initial", key] + (1-alpha) * values["final", key]
            grad_gap = combined(gi, "optical") - combined(gi, "reverse")
            pressure = grad_gap.clamp_min(0)
            row = dict(epoch=epoch, dose_gray=dose,
                       positive_gradient_pressure_covered=float((pressure * mask).sum() / pressure.sum().clamp_min(1e-16)))
            for key in ("optical", "reverse"):
                di, dg = combined(ii, key).mean(), combined(gi, key)
                row["intensity_" + key] = di.item()
                for version, w in (("v2", 1), ("v3", weight)):
                    row[f"{version}_{key}"] = (di + (w * dg).mean()).item()
            for version in ("v2", "v3"):
                row[version + "_gap"] = row[version + "_optical"] - row[version + "_reverse"]
            outputs.append(row)
    primary = [r for r in outputs if r["dose_gray"] == DOSES[0] and r["epoch"] >= 5]
    stats = dict(epochs=len(primary), optical_preferred=sum(r["v3_gap"] < 0 for r in primary),
                 global_conflict_fraction=float(mask.float().mean()), target_mask_fraction=coverage[0]["mask_fraction"],
                 pressure_coverage_epoch5=primary[0]["positive_gradient_pressure_covered"])
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "coverage.csv", coverage)
    write_csv(OUTPUT / "paired_edits.csv", outputs)
    write_json(OUTPUT / "summary.json", stats)
    Image.fromarray(mask_u8 * 255).save(OUTPUT / "conflict.png")
    write_json(OUTPUT / "run_metadata.json", dict(prior, rule=RULE, conflict_sha256=sha256_file(ROOT / "e2_local_correction_v3.py"),
              screen_sha256=sha256_file(Path(__file__)), source_controls_sha256=[sha256_file(NEGATIVE), sha256_file(LINES)],
              prior_metadata_sha256=sha256_file(PRIOR / "run_metadata.json"), prior_edits_sha256=sha256_file(PRIOR / "matched_edits.csv")))
    group = "QXS-E2-LOCAL-CORRECTION-V3-FROZEN-SCREEN-4880-B"
    ck = V2.run_dir / "epoch_005.pth"
    register_many([dict(id=group, comparable_group=group, task="engineering", status="diagnostic_single_scene",
                        method="Fixed V3 source conflict mask on frozen V2 epoch-end outputs", dataset="QXS-SAROPT",
                        split="val", offset=manifest["splits"]["val"].index(SCENE), sample_count=1,
                        selected_scene_ids=[SCENE], checkpoint=ck.as_posix(), checkpoint_sha256=sha256_file(ck),
                        split_manifest=V2.manifest.as_posix(), split_manifest_sha256=prior["manifest_sha256"],
                        metric_profile=DEFAULT_PROFILE, output_policy="float_loss", result_dir=OUTPUT.as_posix(),
                        metadata_path=(OUTPUT/"run_metadata.json").as_posix(), metrics=stats,
                        metric_source=dict(type="json_paths", path=(OUTPUT/"summary.json").as_posix(), paths={k:[k] for k in stats}),
                        notes="One scene; frozen output-space screening only. SAR-only response is an unverified proxy.")])
    print(stats)


if __name__ == "__main__":
    main()
