"""Check frozen E2 sampling stability in three directions on disjoint val8."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_cfao_sampling import MARGIN, central, displacement_errors, four_origins, rms, shift, unshift
from diagnose_drf_no_train import gradients
from diagnose_e2_information_path import read_pair, to_tensor
from diagnose_okd_frozen_gradient import line_geometry, line_response
from diagnose_stg_no_train import DEFAULT_LABELS, load_inputs, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "test_result/e2_sampling_origin_holdout_v1"
DATA = ROOT.parent / "datasets/QXSLAB_SAROPT"
DIRECTIONS = {"horizontal": (0, 1), "vertical": (1, 0), "diagonal": (1, 1)}
METHODS = ("zero", "four_origins", "matched_smoothing")


def load_holdout():
    archive, prior, _, _ = load_inputs(DEFAULT_LABELS, DATA)
    sample = json.loads((ANNOTATIONS / "sample.json").read_text(encoding="utf-8"))
    for key in ("archive_metadata", "archive_metadata_sha256", "source_manifest", "source_manifest_sha256"):
        if sample[key] != prior[key]:
            raise ValueError("The held-out subset must use the same source archive and manifest.")
    indices = list(range(0, 64, 8))
    scenes = [archive["scene_ids"][i] for i in indices]
    if (sample["scene_ids"] != scenes or sample["sampling_indices_zero_based"] != indices
            or not set(scenes).isdisjoint(prior["scene_ids"])):
        raise ValueError("Incorrect held-out subset.")
    manifest = json.loads((ROOT / sample["source_manifest"]).read_text(encoding="utf-8"))
    archive_path = ROOT / sample["archive_metadata"]
    paths = {
        scene: (DATA / manifest["optical_dir"] / f"{scene}.png", DATA / manifest["sar_dir"] / f"{scene}.png",
                archive_path.parent / "e2/gray" / f"{scene}.png")
        for scene in scenes
    }
    return archive, sample, paths


def lock_source_probes(paths, annotations):
    accepted, assessment = {}, []
    for scene, files in paths.items():
        optical, sar = read_pair(files[:2])
        source_gradients = dict(optical=gradients(optical, 1.25), sar=gradients(sar, 1.25))
        accepted[scene] = []
        for row in (probe for probe in annotations if probe["scene"] == scene):
            normal, corridor = line_geometry(row)
            profile = {name: line_response(g, normal, corridor).mean(axis=1) for name, g in source_gradients.items()}
            strength = {name: float(abs(values).max()) for name, values in profile.items()}
            source = row["source"]
            other = "sar" if source == "optical" else "optical"
            best = int(np.argmax(np.abs(profile[source])))
            location = corridor[best:best+1]
            signed = float(profile[source][best])
            eligible = (strength[source] >= 2 and strength[source] >= 1.5*strength[other]
                        and location.min() >= MARGIN and location.max() < optical.shape[0]-MARGIN)
            assessment.append(dict(**row, optical_strength=strength["optical"], sar_strength=strength["sar"],
                                   kept=eligible, offset=best-6, reference_signed=signed))
            if eligible:
                accepted[scene].append((row, normal, location, signed))
    return accepted, assessment


def edge_readout(images, probes, scene, condition):
    rows = []
    for method, image in images.items():
        g = gradients(image*255, 1.25)
        for probe, normal, location, source in probes:
            samples = line_response(g, normal, location) * np.sign(source) / abs(source)
            rows.append(dict(scene=scene, id=probe["id"], source=probe["source"], condition=condition, method=method,
                             signed_fraction=float(samples.mean()), half_source_coverage=float((samples >= 0.5).mean())))
    return rows


@torch.inference_mode()
def measure_sample(model, optical, sar, probes, files, scene, output, device):
    o, s = (to_tensor(x, device) for x in (optical, sar))

    def encode(tensor):
        return model.encoder.encoder_level1(model.encoder.patch_embed(tensor))

    def reconstruct(features):
        base, detail = model.fuse_features(*features)
        return model.reconstruct(o, base, detail)

    shared = tuple(map(encode, (o, s)))
    ref = four_origins(model, shared)
    original, four = (reconstruct(features) for features in ref)
    torch.testing.assert_close(original, model(o, s), rtol=0, atol=1e-6)
    baseline, candidate = (x[0, 0].cpu().numpy().copy() for x in (original, four))
    difference = cv2.GaussianBlur(baseline, (0, 0), 1)-baseline
    alpha = rms(central(candidate-baseline))/rms(central(difference))
    if not 0 <= alpha <= 1:
        raise ValueError(f"Cannot match smoothing RMS on {scene} with one Gaussian residual.")

    def variants(zero, origins):
        smooth = zero + alpha*(cv2.GaussianBlur(zero, (0, 0), 1)-zero)
        return dict(zip(METHODS, (zero, origins, smooth)))

    images = variants(baseline, candidate)
    measured_alpha = dict(scene=scene, smoothing_alpha=alpha, four_delta_rms=rms(central(candidate-baseline)),
                          smooth_delta_rms=rms(central(images["matched_smoothing"]-baseline)))
    archived = cv2.imread(str(files[2]), cv2.IMREAD_GRAYSCALE)
    out = output / scene
    out.mkdir()
    image_rows = []
    for method, value in images.items():
        gray = np.rint(np.clip(value, 0, 1)*255).astype(np.uint8)
        if method == "zero" and (archived is None or np.abs(gray.astype(int)-archived).max() > 1):
            raise ValueError(f"Frozen E2 disagrees with the archive for {scene}.")
        if not cv2.imwrite(str(out / f"{method}.png"), gray):
            raise OSError(f"Failed to save {scene}/{method}.")
        metrics = FusionEvaluator.metric_vector(central(gray), central(optical), central(sar))
        image_rows.append(dict(scene=scene, method=method,
                               **dict(zip(FusionEvaluator.METRIC_NAMES, map(float, metrics)))))
    edges = edge_readout(images, probes, scene, "native")
    stability, features = [], []
    for direction, delta in DIRECTIONS.items():
        moved_features = four_origins(model, tuple(shift(x, delta) for x in shared))
        for method, before, after in zip(METHODS[:2], ref, moved_features):
            difference = displacement_errors(before, after, delta)
            scale = [float(central(x).float().square().mean().sqrt()) for x in before]
            features.append(dict(scene=scene, direction=direction, method=method,
                                 optical_detail_relative_rms=difference[1]/max(scale[1], 1e-8),
                                 sar_detail_relative_rms=difference[3]/max(scale[3], 1e-8)))
        shifted_inputs = tuple(map(encode, (shift(o, delta), shift(s, delta))))
        realigned = [unshift(reconstruct(x), delta)[0, 0].cpu().numpy() for x in four_origins(model, shifted_inputs)]
        moved = variants(*realigned)
        for method, before in images.items():
            stability.append(dict(scene=scene, direction=direction, method=method,
                                  shift_output_rms=rms(central(before-moved[method]))))
        edges.extend(edge_readout(moved, probes, scene, direction))
    return image_rows, stability, features, edges, measured_alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--scene-id", help="Optional single scene from the holdout.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    archive, sample, paths = load_holdout()
    if args.scene_id:
        if args.scene_id not in paths:
            parser.error("Scene is outside the held-out sample.")
        paths = {args.scene_id: paths[args.scene_id]}
    annotation_path = ANNOTATIONS / "source_candidates.csv"
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    probes, assessment = lock_source_probes(paths, annotations)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "source_probes.csv", assessment)
    if args.sources_only:
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = archive["checkpoints"]["e2"]
    digest = sha256_file(ROOT / checkpoint)
    if digest != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("E2 checkpoint differs from the archive.")
    state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    image_rows, stability, features, edges, smoothing = [], [], [], [], []
    for scene, files in paths.items():
        scores, errors, shift_details, boundaries, alpha = measure_sample(
            model, *read_pair(files[:2]), probes[scene], files, scene, args.output_dir, args.device,
        )
        image_rows.extend(scores)
        stability.extend(errors)
        features.extend(shift_details)
        edges.extend(boundaries)
        smoothing.append(alpha)
        print(f"{scene}: three shift directions, {len(probes[scene])} source-dominant probes", flush=True)
    for name, rows in (("image_metrics", image_rows), ("output_stability", stability),
                       ("feature_shift", features), ("edge_readout", edges), ("smoothing_rms", smoothing)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    meta = dict(**sample, evaluated_scene_ids=list(paths), directions_dy_dx=DIRECTIONS, crop_margin=MARGIN,
                checkpoint=checkpoint, checkpoint_sha256=digest, metric_profile=DEFAULT_PROFILE,
                output_policy="clamp_round_uint8_inner_crop", device=args.device,
                script_sha256=sha256_file(Path(__file__)), origin_helper_sha256=sha256_file(ROOT / "diagnose_cfao_sampling.py"),
                annotation_sha256=sha256_file(annotation_path),
                inputs_sha256={scene: [sha256_file(p) for p in files] for scene, files in paths.items()})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
