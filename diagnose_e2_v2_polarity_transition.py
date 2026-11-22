"""Frozen epoch-end readouts and matched opposite-polarity loss probes for 4880-B."""

import csv
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from diagnose_e2_correction_candidates import gray, longest, make_preview, quantize
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_dose import read_probe
from diagnose_e2_primitive_render import (
    CENTER, LABELS, ROOT, boundary, endpoints, pixel_grid, sampling_grid, source_contrast, write_csv,
)
from e2_local_correction_v2 import SPEC
from maintain_experiment_index import sha256_file
from run_e2_local_correction_v2 import PILOT
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json, write_json
from train_direct_fusion import DirectFusionLosses


SCENE, EPOCHS = "4880", range(4, 9)
OUTPUT = Path("test_result/e2_local_correction_v2_polarity_transition/scene4880_B_v1")
DOSES = (0.25, 2.0)  # gray levels of MAE inside the fixed edit support


def symmetric_step(shape, support, capacity, dose):
    """Use identical perturbations with opposite signs, inside [0,1] at every checkpoint."""
    magnitude = shape.abs()
    def effect(gain):
        return torch.minimum(capacity, gain * magnitude)[support].mean().item() * 255
    lo, hi = 0., 1.
    if effect(hi) < dose:
        raise ValueError("Requested common symmetric budget is unattainable")
    for _ in range(32):
        mid = (lo + hi) / 2
        if effect(mid) < dose:
            lo = mid
        else:
            hi = mid
    return torch.sign(shape) * torch.minimum(capacity, hi * magnitude)


@torch.inference_mode()
def main():
    os.chdir(ROOT)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with (LABELS / "source_labels.csv").open(encoding="utf-8-sig") as f:
        row, = (r for r in csv.DictReader(f) if (r["scene"], r["id"]) == (SCENE, "B"))
    segment = endpoints(row)
    _, coords = sampling_grid(segment)
    manifest = read_json(PILOT.manifest)
    archive = read_json(PILOT.result / "run_metadata.json")
    pair = locate_scene(DATA, manifest, SCENE)
    if (sha256_file(PILOT.manifest) != archive["split_manifest_sha256"]
            or [sha256_file(p) for p in pair] != archive["inputs"][SCENE]):
        raise ValueError("Archived inputs differ")
    optical, sar = read_pair(pair)
    sign = int(np.sign(source_contrast(optical, coords, False)))
    if sign != -1 or source_contrast(sar, coords, True) <= 0:
        raise ValueError("Expected the archived opposite source polarities")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o, s = (to_tensor(im, device) for im in (optical, sar))
    valid = s > 0
    losses = DirectFusionLosses().to(device)
    weight = read_json(PILOT.run_dir / "config.json")["gradient_weight"]
    alpha = SPEC["coarse_fusion_weight"]
    target_i = torch.where(valid, torch.maximum(o, s), o)
    go, gs = losses.sobel(o), losses.sobel(s)
    target_g = torch.where(valid, torch.maximum(go, gs), go)
    model = PILOT.build_model().to(device).eval()
    timeline, frames, hashes = [], {}, {}
    for epoch in EPOCHS:
        checkpoint = PILOT.run_dir / f"epoch_{epoch:03d}.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        stage = "reconstruction" if epoch == 4 else "fusion"
        if (state["epoch"] + 1, state["variant"], state["stage"]) != (epoch, PILOT.variant, stage):
            raise ValueError(f"Checkpoint mismatch: {checkpoint}")
        hashes[checkpoint.as_posix()] = sha256_file(checkpoint)
        model.load_state_dict(state["model"])
        captured = []
        hook = model.decoder.correction.register_forward_hook(lambda mod, args, out: captured.append(out))
        try:
            final = model(o, s)
        finally:
            hook.remove()
        (_, candidates, _), = captured
        frames[epoch] = dict(initial=candidates[:, :1], final=final)
        for name, im in dict(**frames[epoch], optical_candidate=candidates[:, 1:2],
                             sar_candidate=torch.where(valid, candidates[:, 2:], candidates[:, :1])).items():
            for precision, array in (("float", gray(im[0, 0])), ("uint8", quantize(gray(im[0, 0])))):
                probe = read_probe(array, coords, sign)
                active = sign * profile(array, coords)[:, CENTER] >= MIN_SUPPORT
                timeline.append(dict(epoch=epoch, stage=stage, method=name, precision=precision,
                                     hits=int(active.sum()), longest=longest(active), **probe))
        if epoch in (4, 8):
            with (PILOT.run_dir / f"probes_epoch_{epoch:03d}.csv").open(encoding="utf-8-sig") as f:
                saved = {r["method"]: r for r in csv.DictReader(f) if (r["scene"], r["probe"]) == (SCENE, "B")}
            for r in (r for r in timeline if r["epoch"] == epoch and r["precision"] == "uint8"):
                ref = saved[r["method"]]
                if (r["hits"] != int(ref["hits"]) or r["longest"] != int(ref["longest"])
                        or abs(r["center_jump_gray"] - float(ref["center_jump_gray"])) > 1e-5):
                    raise ValueError("Archived endpoint probe differs")

    grid = pixel_grid(optical.shape)
    start, end = torch.tensor(segment, dtype=grid.dtype)
    step, taper, _ = boundary(grid, start, end, sign)
    shape = step.to(device)[None, None]
    support = (taper > 0).to(device)[None, None]
    capacity = torch.stack([torch.minimum(im, 1 - im) for fs in frames.values() for im in fs.values()]).amin(0)
    edits = {dose: symmetric_step(shape, support, capacity, dose) for dose in DOSES}
    deltas = []
    for epoch, fs in frames.items():
        for dose, edit in edits.items():
            changes = {}
            for name, im in fs.items():
                intensity0 = (im - target_i).abs()
                gradient0 = (losses.sobel(im) - target_g).abs()
                changes[name] = {}
                for direction, factor in (("optical", 1), ("reverse", -1)):
                    out = im + factor * edit
                    mae = float((out-im).abs()[support].mean() * 255)
                    if abs(mae-dose) > 1e-4 or out.min().item() < -1e-6 or out.max().item() > 1 + 1e-6:
                        raise ValueError("Edit outside its matched budget")
                    delta_i = ((out-target_i).abs()-intensity0).double().mean().item()
                    delta_g = ((losses.sobel(out)-target_g).abs()-gradient0).double().mean().item()
                    total = delta_i + weight * delta_g
                    # Check the map-wise computation against the actual trainer API.
                    before = losses.fusion(o, s, im, valid.float(), weight)
                    after = losses.fusion(o, s, out, valid.float(), weight)
                    # A difference of two float32 reductions has absolute uncertainty
                    # on the scale of the unperturbed losses, not the small difference.
                    for item, lo, hi in zip((total, delta_i, delta_g), before, after):
                        tolerance = 8 * torch.finfo(lo.dtype).eps * max(abs(lo.item()), abs(hi.item()), 1e-3)
                        if abs(item - (hi-lo).item()) > tolerance:
                            raise ValueError(f"Loss mismatch: epoch={epoch}, {name}, {direction}, "
                                             f"map delta={item}, API delta={(hi-lo).item()}, "
                                             f"base={lo.item()}, tolerance={tolerance}")
                    changes[name][direction] = (delta_i, weight*delta_g)
                    deltas.append(dict(epoch=epoch, method=name, dose_gray=dose, direction=direction,
                                       measured_mae_gray=mae, delta_intensity=delta_i,
                                       delta_weighted_gradient=weight*delta_g, delta_total=total,
                                       edited_jump_gray=read_probe(gray(out[0,0]), coords, sign)["center_jump_gray"]))
            for direction in ("optical", "reverse"):
                di, dg = alpha*np.array(changes["initial"][direction]) + (1-alpha)*np.array(changes["final"][direction])
                deltas.append(dict(epoch=epoch, method="weighted_outputs", dose_gray=dose, direction=direction,
                                   measured_mae_gray=dose, delta_intensity=di, delta_weighted_gradient=dg,
                                   delta_total=di+dg, edited_jump_gray=None))

    def first_negative(method):
        return next((r["epoch"] for r in timeline if r["method"] == method and r["precision"]=="float"
                     and r["center_jump_gray"] < 0), 0)
    flip = first_negative("initial")
    counts = dict(sampled_epochs=len(EPOCHS), coarse_first_negative_epoch=flip,
                  final_first_negative_epoch=first_negative("final"), support_pixels=int(support.sum()))
    metadata = dict(scene=SCENE, probe=row, epochs=list(EPOCHS), doses_gray=list(DOSES),
                    gradient_weight=weight, coarse_fusion_weight=alpha, sign=sign, torch=str(torch.__version__),
                    device=str(device), input_hashes=archive["inputs"][SCENE], checkpoint_hashes=hashes,
                    manifest_sha256=sha256_file(PILOT.manifest), label_sha256=sha256_file(LABELS / "source_labels.csv"),
                    source_sha256=sha256_file(Path(__file__)), endpoint_checks=[4,8],
                    sar_larger_sobel_fraction=float(((gs > go) & valid)[support].float().mean()),
                    sar_larger_intensity_fraction=float(((s > o) & valid)[support].float().mean()))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT / "timeline.csv", timeline)
    write_csv(OUTPUT / "matched_edits.csv", deltas)
    write_json(OUTPUT / "summary.json", counts)
    write_json(OUTPUT / "run_metadata.json", metadata)
    make_preview(dict(Optical=optical, SAR=sar, **{f"E{e} initial": gray(frames[e]["initial"][0,0])
                                                   for e in EPOCHS}), segment, OUTPUT / "timeline.png")
    ck = PILOT.run_dir / f"epoch_{(flip or 8):03d}.pth"
    key = "QXS-E2-LOCAL-CORRECTION-V2-TRANSITION-4880B-V1"
    register_many([dict(id=key, comparable_group=key, task="engineering", status="diagnostic_single_scene",
                        method="Frozen E2 correction V2 epoch-end transition and opposite signed-step losses",
                        dataset="QXS-SAROPT", split="val", offset=archive["scene_ids"].index(SCENE),
                        sample_count=1, selected_scene_ids=[SCENE], checkpoint=ck.as_posix(), checkpoint_sha256=hashes[ck.as_posix()],
                        split_manifest=PILOT.manifest.as_posix(), split_manifest_sha256=metadata["manifest_sha256"],
                        metric_profile=archive["metric_profile"], output_policy=archive["output_policy"],
                        result_dir=OUTPUT.as_posix(), metadata_path=(OUTPUT/"run_metadata.json").as_posix(),
                        metrics=counts, metric_source=dict(type="json_paths", path=(OUTPUT/"summary.json").as_posix(),
                                                          paths={k:[k] for k in counts}),
                        notes="One validation scene; counterfactual output edits use float training loss; endpoint probes match archive.")])


if __name__ == "__main__":
    main()
