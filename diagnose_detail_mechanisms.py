"""Frozen S2 routing controls and E2 Detail transmission diagnostics."""

import argparse
import csv
import os
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_e2_information_path import image_u8, locate_scene, read_pair, safe_pearson, scalar_rms, to_tensor
from e2_base_anchor import OpticalBaseAnchoredE2
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import read_json, sha256_file, write_json
from run_s2_expert_pilot import CHECKPOINT, DATA_ROOT, MANIFEST, validated_run
from s2_expert_models import S2ModelFactory
from utils.dataset_os import validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
STEP = 0.05
QUALITY = ("Qabf", "Nabf", "VIF", "SSIM_O", "SSIM_S", "gradient_corr_O", "gradient_corr_S")


def gradients(image):
    image = image.astype(np.float64)
    return np.concatenate((np.diff(image, axis=0).ravel(), np.diff(image, axis=1).ravel()))


def quality(image, optical, sar):
    if not torch.isfinite(image).all():
        raise FloatingPointError("Non-finite decoder output")
    gray = image_u8(image)
    edge = gradients(gray)
    values = [metric(gray, optical, sar) for metric in (FusionEvaluator.Qabf, FusionEvaluator.Nabf, FusionEvaluator.VIFF)]
    for source in (optical, sar):
        values.append(FusionEvaluator._ssim_pair(gray, source))
    values.extend(safe_pearson(edge, gradients(source)) for source in (optical, sar))
    return {key: float(value) if value is not None and np.isfinite(value) else None
            for key, value in zip(QUALITY, values)}


def fixed_route(weights):
    """Use the most frequent expert pair with its mean weights within that image."""
    active = weights[0] > 0
    if not bool((active.sum(0) == 2).all()):
        raise ValueError("Expected exactly two active experts per block")
    # With three experts, the excluded expert identifies each pair.
    excluded = (~active).long().argmax(dim=0)
    modal = excluded.flatten().bincount(minlength=3).argmax()
    in_pair = excluded == modal
    shared = weights[0, :, in_pair].mean(dim=-1)
    return shared.view(1, 3, 1, 1).expand_as(weights), float(in_pair.float().mean())


def probe_routes(model, encoded, base, o, original, optical, sar):
    _, do, _, ds = encoded
    fusion = model.detail_fusion
    weights = fusion.routing(do, ds, base)
    experts = [expert(do, ds, base) for expert in fusion.experts]
    constant, modal_fraction = fixed_route(weights)
    roll = (weights.shape[-2] // 2, weights.shape[-1] // 2)
    variants = {"regional": weights, "fixed_pair": constant,
                "rolled": weights.roll(roll, dims=(-2, -1))}
    rows = []
    for name, variant in variants.items():
        up = F.interpolate(variant, size=do.shape[-2:], mode="nearest")
        detail = fusion.synthesis(sum(up[:, i:i + 1] * x for i, x in enumerate(experts)))
        image = model.reconstruct(o, base, detail)
        if name == "regional":
            torch.testing.assert_close(image, original, rtol=0, atol=1e-6)
        rows.append(dict(
            condition=name, modal_pair_fraction=modal_fraction,
            weight_map_delta_rms=scalar_rms(variant - weights),
            pair_change_fraction=float(((variant > 0) != (weights > 0)).any(dim=1).float().mean()),
            output_delta_rms=scalar_rms(image - original), **quality(image, optical, sar),
        ))
    return rows, {"roll_blocks": roll}


def fine_part(x, haar):
    bands, size = haar(x)
    return haar.inverse((torch.zeros_like(bands[0]), *bands[1:]), size)


def haar_parts(x, haar):
    first, size = haar(x)
    second, low_size = haar(first[0])
    fine = haar.inverse((torch.zeros_like(first[0]), *first[1:]), size)
    middle_ll = haar.inverse((torch.zeros_like(second[0]), *second[1:]), low_size)
    middle = haar.inverse((middle_ll, *(torch.zeros_like(b) for b in first[1:])), size)
    dc = x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
    return {"fine": fine, "middle": middle, "dc": dc}


def perturb_source_detail(fusion, do, ds, base, source, change):
    if source == "optical":
        return tuple(fusion(do + sign * change, ds, base) for sign in (1, -1))
    return tuple(fusion(do, ds + sign * change, base) for sign in (1, -1))


def probe_transfer(model, encoded, base, o, original, optical, sar):
    bo, do, bs, ds = encoded
    haar = model.lwpd.haar
    fused = model.detail_fusion(do, ds, base)
    source_parts = {"optical": haar_parts(do, haar), "sar": haar_parts(ds, haar)}
    base_parts = {"optical": haar_parts(bo, haar), "sar": haar_parts(bs, haar)}
    fused_parts = haar_parts(fused, haar)
    candidates = []
    for source in ("optical", "sar"):
        candidates.append((source, "fine", "native_detail", source_parts[source]["fine"]))
        # E2 Detail is synthesized from the first-level high bands. A middle-scale
        # perturbation in this input is an injection from that source's Base.
        candidates.append((source, "middle", "source_base_injected", base_parts[source]["middle"]))
    candidates.extend(("decoder", band, "native_fused_detail", direction)
                      for band, direction in fused_parts.items())
    direction_rms = [scalar_rms(direction) for _, _, _, direction in candidates]
    nonzero = [value for value in direction_rms if value > 1e-7]
    if not nonzero:
        raise ValueError("No usable Detail perturbation directions")
    epsilon = STEP * min(nonzero)
    rows = [dict(condition="baseline", **quality(original, optical, sar))]
    for (source, band, origin, direction), size in zip(candidates, direction_rms):
        if size <= 1e-7:
            rows.append(dict(condition=f"{source}_{band}", source=source, band=band,
                             direction_origin=origin, direction_rms=size, status="degenerate"))
            continue
        change = direction * (epsilon / size)
        if source == "decoder":
            plus, minus = fused + change, fused - change
        else:
            plus, minus = perturb_source_detail(model.detail_fusion, do, ds, base, source, change)
        y_plus, y_minus = (model.reconstruct(o, base, detail) for detail in (plus, minus))
        detail_response = (plus - minus) * (0.5 / epsilon)
        image_response = (y_plus - y_minus) * (0.5 / epsilon)
        detail_gain = scalar_rms(detail_response)
        output_gain = scalar_rms(image_response)
        rows.append(dict(
            condition=f"{source}_{band}", source=source, band=band, direction_origin=origin,
            direction_rms=size, status="measured", input_step_rms=epsilon,
            detail_gain=detail_gain,
            detail_same_band_gain=scalar_rms(haar_parts(detail_response, haar)[band]),
            output_gain=output_gain,
            output_per_fused_detail_rms=output_gain / detail_gain if detail_gain > 1e-7 else None,
            output_same_band_gain=scalar_rms(haar_parts(image_response, haar)[band]),
            **{f"{key}_{sign}": value for sign, y in (("plus", y_plus), ("minus", y_minus))
               for key, value in quality(y, optical, sar).items()},
        ))
    return rows, {"native_source_detail_rms": {
        source: {band: scalar_rms(value) for band, value in parts.items()}
        for source, parts in source_parts.items()
    }, "common_step_rms": epsilon}


def probe_directions(model, encoded, base, o, original, optical, sar):
    _, do, _, ds = encoded
    haar = model.lwpd.haar
    fusion = model.detail_fusion
    fused = fusion(do, ds, base)
    source_bands = {"optical": fine_part(do, haar), "sar": fine_part(ds, haar)}
    source_rms = {name: scalar_rms(band) for name, band in source_bands.items()}
    decode_step = STEP * scalar_rms(fine_part(fused, haar))
    source_step = STEP * min(source_rms.values())
    if min(source_step, decode_step) <= 1e-7:
        raise ValueError("A source fine band or the fused fine band is too small for the direction probe")

    rows = [dict(condition="baseline", **quality(original, optical, sar))]
    ratios = {}
    for source, band in source_bands.items():
        plus, minus = perturb_source_detail(fusion, do, ds, base, source, band * (source_step / source_rms[source]))
        full = (plus - minus) / (2 * source_step)
        fine = fine_part(full, haar)
        full_rms, fine_rms = scalar_rms(full), scalar_rms(fine)
        if fine_rms <= 1e-7:
            raise ValueError(f"Fused response has no usable fine component for {source}")
        for scope, direction, size in (("full", full, full_rms), ("fine", fine, fine_rms)):
            unit_direction = direction / size
            change = unit_direction * decode_step
            measured_step = scalar_rms(change)
            if abs(measured_step / decode_step - 1) > 1e-5:
                raise AssertionError("Decoder perturbations must have the same RMS")
            (y_plus, reduced_plus), (y_minus, reduced_minus) = (
                model.decoder(None, base, fused + sign * change) for sign in (1, -1)
            )
            y_half_plus, y_half_minus = (
                model.reconstruct(o, base, fused + sign * 0.5 * change) for sign in (1, -1)
            )
            response = (y_plus - y_minus) / (2 * decode_step)
            half_response = (y_half_plus - y_half_minus) / decode_step
            output_gain = scalar_rms(response)
            reduced_gain = scalar_rms((reduced_plus - reduced_minus) / (2 * decode_step))
            fine_gain = scalar_rms(fine_part(response, haar))
            rows.append(dict(
                condition=f"{source}_{scope}", source=source, scope=scope, source_feature_step_rms=source_step,
                fused_direction_rms=size, fused_fine_energy_fraction=(fine_rms / full_rms) ** 2,
                decoder_input_step_rms=measured_step, reduce_channel_gain=reduced_gain,
                output_gain=output_gain, output_fine_gain=fine_gain,
                half_step_relative_response_difference=scalar_rms(response - half_response) / max(output_gain, 1e-12),
                **{f"{key}_{sign}": value for sign, y in (("plus", y_plus), ("minus", y_minus))
                   for key, value in quality(y, optical, sar).items()},
            ))
    by_condition = {row["condition"]: row for row in rows}
    for scope in ("full", "fine"):
        left, right = (by_condition[f"{source}_{scope}"] for source in ("optical", "sar"))
        ratios[scope] = {key: right[key] / left[key] if left[key] > 1e-7 else None
                         for key in ("reduce_channel_gain", "output_gain", "output_fine_gain")}
    return rows, {"source_fine_rms": source_rms, "sar_over_optical": ratios}


def summarize(kind, scenes):
    by_condition = {}
    for scene in scenes:
        for row in scene["rows"]:
            by_condition.setdefault(row["condition"], {})[scene["scene_id"]] = row
    keys = QUALITY
    if kind != "routes":
        del by_condition["baseline"]
        keys = {
            "transfer": ("detail_gain", "detail_same_band_gain", "output_gain", "output_same_band_gain",
                         "output_per_fused_detail_rms"),
            "directions": ("reduce_channel_gain", "output_gain", "output_fine_gain", "half_step_relative_response_difference",
                           *(f"{key}_{sign}" for key in QUALITY for sign in ("plus", "minus"))),
        }[kind]
    result = {}
    for key in keys:
        common = [scene["scene_id"] for scene in scenes
                  if all(rows[scene["scene_id"]].get(key) is not None for rows in by_condition.values())]
        means = {name: fmean(rows[scene][key] for scene in common) if common else None
                 for name, rows in by_condition.items()}
        result[key] = dict(common_count=len(common), means=means)
        if kind == "routes":
            result[key]["regional_minus_controls"] = {
                name: means["regional"] - value if common else None
                for name, value in means.items() if name != "regional"
            }
    return result


@torch.inference_mode()
def run(args):
    if args.probe == "routes":
        checkpoint, manifest_path, config, _ = validated_run(args, "top2")
        model = S2ModelFactory.build("top2")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("mode") != "top2" or state.get("config") != config:
            raise ValueError("Top-2 checkpoint does not match the recorded pilot")
        probe = probe_routes
    else:
        checkpoint, manifest_path = CHECKPOINT, args.manifest
        model = OpticalBaseAnchoredE2()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("variant") != model.VARIANT:
            raise ValueError("Expected optical Base E2 checkpoint")
        probe = {"transfer": probe_transfer, "directions": probe_directions}[args.probe]
    manifest = read_json(manifest_path)
    validate_manifest_splits(manifest)
    if len(set(args.scene_ids)) != len(args.scene_ids) or not set(args.scene_ids) <= set(manifest["splits"]["val"]):
        raise ValueError("Scene IDs must be unique members of the selected checkpoint's val split")
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    del state
    scenes = []
    for scene in args.scene_ids:
        paths = locate_scene(args.data_root, manifest, scene)
        optical, sar = read_pair(paths)
        o, s = (to_tensor(image, args.device) for image in (optical, sar))
        encoded = model.encode_pair(o, s)
        base, detail = model.fuse_features(*encoded)
        original = model.reconstruct(o, base, detail)
        torch.testing.assert_close(original, model(o, s), rtol=0, atol=1e-6)
        rows, details = probe(model, encoded, base, o, original, optical, sar)
        scenes.append(dict(scene_id=scene, rows=rows, details=details,
                           inputs_sha256={name: sha256_file(path) for name, path in zip(("optical", "sar"), paths)}))
        print(f"{args.probe}: completed scene {scene}", flush=True)
    return checkpoint, manifest_path, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("routes", "transfer", "directions"))
    parser.add_argument("--scene-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output_dir.resolve().relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint, manifest, scenes = run(args)
    summary = summarize(args.probe, scenes)
    metadata = dict(probe=args.probe, split="val", scene_ids=args.scene_ids, device=args.device,
                    precision="fp32_no_tf32", output_policy="clamp_then_round_uint8",
                    metric_profile=DEFAULT_PROFILE, checkpoint=checkpoint.as_posix(),
                    checkpoint_sha256=sha256_file(checkpoint), split_manifest=manifest.as_posix(),
                    split_manifest_sha256=sha256_file(manifest),
                    script_sha256=sha256_file(Path(__file__)),
                    limits=["Frozen diagnostic, no training or test split.",
                            "Perturbation gains measure sensitivity. Injected middle scale is not native source Detail.",
                            "Signed source gradient correlations use all pixels and do not label true structure."])
    if args.probe == "directions":
        metadata["direction_protocol"] = dict(
            source_step_fraction_of_weaker_fine_band=STEP,
            decoder_step_fraction_of_fused_fine_band=STEP, half_step_check=True,
            notes="Fine alignment retains first-level Haar high bands. Orientation and channel composition are free to differ.",
        )
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "diagnosis.json"
    meta_path = output / "run_metadata.json"
    write_json(raw_path, dict(scenes=scenes, summary=summary))
    write_json(meta_path, metadata)
    rows = [dict(scene_id=scene["scene_id"], **row) for scene in scenes for row in scene["rows"]]
    with (output / "per_scene.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    paths = {}
    metrics = {}
    registry_keys = {"routes": ("Qabf", "Nabf", "VIF"), "transfer": ("detail_same_band_gain", "output_same_band_gain"),
                     "directions": ("reduce_channel_gain", "output_gain", "output_fine_gain")}[args.probe]
    for key in registry_keys:
        report = summary[key]
        for condition, value in report["means"].items():
            if value is not None:
                name = f"{condition}_{key}"
                paths[name] = ["summary", key, "means", condition]
                metrics[name] = value
    eid = f"QXS-DETAIL-{args.probe.upper()}-{sha256_file(meta_path)[:12]}"
    register_many([dict(id=eid, comparable_group=eid, method=f"Frozen Detail {args.probe} diagnosis",
                        status="diagnostic_fixed_subset", task="engineering", dataset="QXS-SAROPT",
                        split="val", offset=0, sample_count=len(scenes), scene_ids=args.scene_ids,
                        **{key: metadata[key] for key in ("checkpoint", "checkpoint_sha256", "split_manifest",
                                                         "split_manifest_sha256", "metric_profile", "output_policy")},
                        result_dir=output.as_posix(), metadata_path=meta_path.as_posix(), metrics=metrics,
                        raw_result_path=(output / "per_scene.csv").as_posix(),
                        metric_source=dict(type="json_paths", path=raw_path.as_posix(), paths=paths))])
    print(f"Saved {raw_path}", flush=True)


if __name__ == "__main__":
    main()
