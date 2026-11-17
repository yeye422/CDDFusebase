"""Check linear spectral aggregation and a sampling-origin control on one frozen E2 pair."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_e2_information_path import read_pair, to_tensor
from diagnose_stg_no_train import DEFAULT_LABELS, load_inputs, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
SCENE = "1276"
MARGIN = 16
ORIGINS = ((0, 0), (0, 1), (1, 0), (1, 1))


def central(image):
    return image[..., MARGIN:-MARGIN, MARGIN:-MARGIN]


def rms(array):
    return float(np.sqrt(np.mean(np.abs(array)**2)))


def spectral_probe(detail):
    """Compare the Fourier expression with its explicit position-dependent channel map."""
    rng = np.random.default_rng(42)
    shape = (4, detail.shape[0], 3, 3)
    k = (rng.normal(size=shape) + 1j*rng.normal(size=shape)) / np.sqrt(np.prod(shape[1:]))
    # Frequency conjugate symmetry requires K(-d) = conj(K(d)).
    k = (k + k[..., ::-1, ::-1].conj()) / 2
    spectrum = np.fft.fftshift(np.fft.fft2(detail), axes=(-2, -1))
    height, width = detail.shape[-2:]
    shifts = [(i-1, j-1) for i in range(3) for j in range(3)]
    coordinates = np.arange(height)[:, None] / height, np.arange(width)[None, :] / width
    mapping = np.zeros((shape[0], shape[1], height, width), dtype=np.complex128)

    def apply(spectral_input, cyclic):
        total = np.zeros((shape[0], height, width), dtype=np.complex128)
        for dy, dx in shifts:
            shifted = np.roll(spectral_input, (dy, dx), axis=(-2, -1))
            if not cyclic:
                if dy:
                    shifted[..., 0 if dy > 0 else -1, :] = 0
                if dx:
                    shifted[..., 0 if dx > 0 else -1] = 0
            total += np.einsum("oi,ihw->ohw", k[..., dy+1, dx+1], shifted)
        return np.fft.ifft2(np.fft.ifftshift(total, axes=(-2, -1)))

    for dy, dx in shifts:
        carrier = np.exp(2j*np.pi*(dy*coordinates[0]+dx*coordinates[1]))
        mapping += k[..., dy+1, dx+1, None, None] * carrier
    equivalent = np.einsum("oihw,ihw->ohw", mapping, detail)
    circular, zero_padded = (apply(spectrum, cyclic) for cyclic in (True, False))
    np.testing.assert_allclose(equivalent, circular, rtol=1e-10, atol=1e-12)
    scale = rms(circular)
    metrics = dict(circular_spatial_relative_error=rms(equivalent-circular)/scale,
                   circular_imaginary_relative_rms=rms(circular.imag)/scale,
                   zero_padding_relative_difference=rms(zero_padded.real-circular.real)/scale,
                   zero_padding_imaginary_relative_rms=rms(zero_padded.imag)/rms(zero_padded))
    for dx in (1, 8):
        moved = np.roll(detail, dx, axis=-1)
        shifted_spectrum = np.fft.fftshift(np.fft.fft2(moved), axes=(-2, -1))
        restored = np.roll(apply(shifted_spectrum, cyclic=True), -dx, axis=-1)
        metrics[f"shift{dx}_relative_equivariance_error"] = rms(restored-circular)/scale
    return metrics


def shift(x, delta):
    return torch.roll(x, delta, dims=(-2, -1))


def unshift(x, delta):
    return shift(x, tuple(-value for value in delta))


def four_origins(model, shared):
    values = []
    for delta in ORIGINS:
        encoded = model.lwpd(*(shift(feature, delta) for feature in shared))
        values.append(tuple(unshift(x, delta) for x in encoded))
    return values[0], tuple(sum(parts)/len(ORIGINS) for parts in zip(*values))


def displacement_errors(before, after, delta):
    return [float(central(a-unshift(b, delta)).float().square().mean().sqrt()) for a, b in zip(before, after)]


@torch.inference_mode()
def sampling_probe(model, o, s):
    def encode(x):
        return model.encoder.encoder_level1(model.encoder.patch_embed(x))

    def reconstruct(features):
        base, detail = model.fuse_features(*features)
        return model.reconstruct(o, base, detail)

    shared = tuple(map(encode, (o, s)))
    reference = four_origins(model, shared)
    normal, averaged = (reconstruct(features) for features in reference)
    torch.testing.assert_close(normal, model(o, s), rtol=0, atol=1e-6)
    shift_rows = []
    for distance in (1, 2):
        delta = (0, distance)
        feature_shift = tuple(shift(value, delta) for value in shared)
        shifted = four_origins(model, feature_shift)
        for name, before, after in zip(("single_origin", "four_origins"), reference, shifted):
            errors = displacement_errors(before, after, delta)
            before_rms = [float(central(x).float().square().mean().sqrt()) for x in before]
            relative = [error / max(denominator, 1e-8) for error, denominator in zip(errors, before_rms)]
            shift_rows.append(dict(method=name, shift_pixels=distance,
                                   **dict(zip(("optical_base", "optical_detail", "sar_base", "sar_detail"), relative))))
    names = ("zero", "four_origins", "matched_smoothing")
    baseline, candidate = (x[0, 0].cpu().numpy().copy() for x in (normal, averaged))
    smooth_direction = cv2.GaussianBlur(baseline, (0, 0), 1) - baseline
    budget = rms(central(candidate-baseline))
    alpha = budget / rms(central(smooth_direction))
    smooth = baseline + min(alpha, 1) * smooth_direction
    images = dict(zip(names, (baseline, candidate, smooth)))
    summaries = dict(smoothing_alpha=min(alpha, 1), candidate_delta_rms=budget,
                     smoothing_delta_rms=rms(central(smooth-baseline)))
    delta = (0, 1)
    shifted_input = tuple(map(encode, (shift(o, delta), shift(s, delta))))
    moved_zero, moved_average = (unshift(reconstruct(x), delta)[0, 0].cpu().numpy()
                                 for x in four_origins(model, shifted_input))
    moved_smooth = moved_zero + min(alpha, 1)*(cv2.GaussianBlur(moved_zero, (0, 0), 1)-moved_zero)
    summaries.update({name+"_output_shift1_rms": rms(central(before-after))
                      for (name, before), after in zip(images.items(), (moved_zero, moved_average, moved_smooth))})
    detail_o, detail_s = reference[0][1], reference[0][3]
    detail = torch.cat((detail_o[0, :4], detail_s[0, :4]), dim=0).cpu().numpy().astype(np.float64)
    return detail, shift_rows, images, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-val8", action="store_true", help="Only the sampling-origin comparison on fixed val8.")
    args = parser.parse_args()
    archive, sample, _, paths = load_inputs(DEFAULT_LABELS, ROOT.parent / "datasets/QXSLAB_SAROPT")
    checkpoint = archive["checkpoints"]["e2"]
    digest = sha256_file(ROOT / checkpoint)
    if digest != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Checkpoint differs from the archived E2.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    scenes = list(paths) if args.sampling_val8 else [SCENE]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shifts, scores, summaries = [], [], []
    for scene in scenes:
        optical, sar = read_pair(paths[scene][:2])
        detail, scene_shifts, images, summary = sampling_probe(
            model, *(to_tensor(x, args.device) for x in (optical, sar)),
        )
        gray = {key: np.rint(np.clip(value, 0, 1)*255).astype(np.uint8) for key, value in images.items()}
        archived = cv2.imread(str(paths[scene][2]), cv2.IMREAD_GRAYSCALE)
        if archived is None or np.abs(archived.astype(int)-gray["zero"]).max() > 1:
            raise ValueError(f"Frozen E2 does not match the archived scene {scene}.")
        out = args.output_dir / scene if args.sampling_val8 else args.output_dir
        out.mkdir(exist_ok=True)
        for key, value in gray.items():
            if not cv2.imwrite(str(out / f"{key}.png"), value):
                raise OSError(f"Cannot save {scene} output {key}.")
            values = FusionEvaluator.metric_vector(central(value), central(optical), central(sar))
            scores.append(dict(scene=scene, method=key, **dict(zip(FusionEvaluator.METRIC_NAMES, map(float, values)))))
        shifts.extend(dict(scene=scene, **row) for row in scene_shifts)
        summaries.append(dict(scene=scene, **summary))
        if not args.sampling_val8:
            for filename, values in (("fourier.json", spectral_probe(detail)), ("sampling.json", summary)):
                (out / filename).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        print(f"{scene}: frozen sampling-origin comparison complete", flush=True)
    write_csv(args.output_dir / "feature_shift.csv", shifts)
    write_csv(args.output_dir / "image_metrics.csv", scores)
    write_csv(args.output_dir / "sampling_per_scene.csv", summaries)
    metadata = dict(**sample, evaluated_scene_ids=scenes, checkpoint=checkpoint, checkpoint_sha256=digest,
                    crop_margin=MARGIN, metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8_inner_crop",
                    device=args.device, script_sha256=sha256_file(Path(__file__)),
                    inputs_sha256={scene: [sha256_file(p) for p in paths[scene]] for scene in scenes})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
