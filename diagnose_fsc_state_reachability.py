"""Frozen-E2, untrained-FSC diagnostic of LL-to-SSM state reachability.

No training or learned FSC checkpoint is used.  Reports internal SSM effects,
not fused-image improvements.  Existing outputs are never overwritten.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from e2_fsc_mamba import FSCMambaE2, scan_with_state
from maintain_experiment_index import sha256_file
from run_e2_fsc_mamba import ORIGINAL_LOSS_PILOT, DATA, source_manifest
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("test_result/e2_fsc_state_reachability_v1")


def rms(x):
    return x.float().square().mean().sqrt().item()


def segment_rms(delta):
    return [rms(part) for part in torch.tensor_split(delta, 3, dim=1)]


def parse_values(raw, cast):
    values = [cast(part.strip()) for part in raw.split(",")]
    if not values or len(set(values)) != len(values):
        raise ValueError("Expected a nonempty list without duplicates")
    return values


def selected_sequences(joint, low, axis, line_count):
    """Return matching real-image Detail/LL lines, with line position separate from time."""
    if axis == "row":
        data = joint.permute(0, 2, 3, 1)
        context = low.permute(0, 2, 3, 1)
    elif axis == "column":
        data = joint.permute(0, 3, 2, 1)
        context = low.permute(0, 3, 2, 1)
    else:
        raise ValueError(axis)
    batch, lines, length, channels = data.shape
    if batch != 1:
        raise ValueError("This diagnostic requires a single scene")
    if not 2 <= line_count <= lines:
        raise ValueError("--lines must be between 2 and the available image lines")
    indices = torch.div((2 * torch.arange(line_count, device=data.device) + 1) * lines,
                        2 * line_count, rounding_mode="floor")
    seq = data[:, indices].reshape(line_count, length, channels).float()
    ll_seq = context[:, indices].reshape_as(seq).float()
    return seq, ll_seq, indices.tolist()


def initial_state(ssm, scan, low_sequence, scale):
    summary = ssm.terminal(low_sequence)
    return ssm.initial_state(summary * scale, scan)


def state_effect(scan, sequence, init, reference=None):
    if reference is None:
        reference = scan_with_state(scan, sequence, torch.zeros_like(init))
    changed = scan_with_state(scan, sequence, init)
    return changed - reference


def make_profiles(ssm, joint, low, lengths, scales, line_count):
    profiles, counterfactuals = [], []
    for axis in ("row", "column"):
        scan = ssm.core.row_scan if axis == "row" else ssm.core.column_scan
        x, l, indices = selected_sequences(joint, low, axis, line_count)
        if max(lengths) > x.shape[1]:
            raise ValueError(f"--lengths cannot exceed available {axis} scan length {x.shape[1]}")
        for direction in ("forward", "reverse"):
            seq = x if direction == "forward" else x.flip(1)
            ll = l if direction == "forward" else l.flip(1)
            # Keep the same LL-derived initial state when varying prefix length;
            # otherwise changing the LL window would confound the decay test.
            full_h = initial_state(ssm, scan, ll, 1.0)
            wrong_h = initial_state(ssm, scan, ll.roll(max(1, line_count // 2), 0), 1.0)
            for length in lengths:
                prefix = seq[:, :length]
                zero = scan_with_state(scan, prefix, torch.zeros_like(full_h))
                for scale in scales:
                    delta = state_effect(scan, prefix, full_h * scale, zero)
                    head, middle, tail = segment_rms(delta)
                    profiles.append(dict(axis=axis, direction=direction, length=length,
                                         scale=scale, line_indices=";".join(map(str, indices)),
                                         head_rms=head, middle_rms=middle, tail_rms=tail,
                                         tail_over_head=tail / max(head, 1e-12),
                                         full_delta_rms=rms(delta),
                                         zero_scan_rms=rms(zero),
                                         delta_to_scan=rms(delta) / max(rms(zero), 1e-12)))
            zero = scan_with_state(scan, seq, torch.zeros_like(full_h))
            for scale in scales:
                good = scan_with_state(scan, seq, scale * full_h)
                wrong = scan_with_state(scan, seq, scale * wrong_h)
                good_delta, wrong_delta = good - zero, wrong - zero
                counterfactuals.append(dict(axis=axis, direction=direction,
                                            length=seq.shape[1], scale=scale,
                                            correct_vs_zero_rms=rms(good_delta),
                                            shuffled_vs_zero_rms=rms(wrong_delta),
                                            correct_vs_shuffled_rms=rms(good - wrong),
                                            correct_shuffled_to_correct=rms(good - wrong) / max(rms(good_delta), 1e-12),
                                            initial_correct_rms=rms(full_h * scale),
                                            initial_shuffled_rms=rms(wrong_h * scale),
                                            zero_scan_rms=rms(zero)))
    return profiles, counterfactuals


def ssm_with_low(ssm, joint, low):
    """Exact FSC 2-D SSM path for a supplied projected LL map."""
    d_row, l_row = joint.permute(0, 2, 3, 1), low.permute(0, 2, 3, 1)
    rows = ssm.scan_lines(d_row, l_row, ssm.core.row_scan).permute(0, 3, 1, 2)
    columns = ssm.scan_lines(d_row.transpose(1, 2), l_row.transpose(1, 2),
                             ssm.core.column_scan).permute(0, 3, 2, 1)
    return joint + ssm.core.merge(torch.cat((ssm.core.local(joint), rows, columns), dim=1))


def make_spatial(ssm, joint, low):
    """Check whether correct LL changes the entire bidirectional SSM output."""
    with torch.no_grad():
        correct = ssm_with_low(ssm, joint, low)
        zero = ssm_with_low(ssm, joint, torch.zeros_like(low))
        shifted = low.roll((low.shape[-2] // 2, low.shape[-1] // 2), dims=(-2, -1))
        wrong = ssm_with_low(ssm, joint, shifted)
    return dict(shape=list(joint.shape), correct_vs_zero_rms=rms(correct - zero),
                shuffled_vs_zero_rms=rms(wrong - zero),
                correct_vs_shuffled_rms=rms(correct - wrong),
                correct_shuffled_to_correct=rms(correct - wrong) / max(rms(correct - zero), 1e-12),
                zero_ssm_rms=rms(zero),
                correct_to_zero_relative=rms(correct - zero) / max(rms(zero), 1e-12),
                note="Internal untrained FSC SSM output; not the final fused image")


def load_frozen_e2(model, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = saved["model"]
    mapping = (("encoder", model.encoder), ("decoder", model.decoder),
               ("lwpd", model.lwpd), ("base_fusion", model.base_fusion),
               ("detail_fusion", model.detail_fusion.keep_fusion))
    counts = {}
    for prefix, target in mapping:
        group = {name[len(prefix) + 1:]: value for name, value in weights.items()
                 if name.startswith(prefix + ".")}
        target.load_state_dict(group, strict=True)
        counts[prefix] = len(group)
    return counts


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows produced")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    torch.set_num_threads(2)
    lengths = parse_values(args.lengths, int)
    scales = parse_values(args.scales, float)
    if any(n < 3 for n in lengths) or any(not math.isfinite(s) or s < 0 or s > 2 for s in scales):
        raise ValueError("Lengths must be >=3; scales must be finite and in [0, 2]")
    manifest = source_manifest()
    scene = args.scene or manifest["splits"]["val"][0]
    if scene not in manifest["splits"]["val"]:
        raise ValueError(f"Scene {scene} not in locked validation split")
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / f"scene_{scene}_seed42"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = ORIGINAL_LOSS_PILOT.final
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required archived E2 weights: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(42)
    model = FSCMambaE2("state")
    loaded = load_frozen_e2(model, checkpoint)
    model = model.to(device).eval()
    optical, sar = (to_tensor(image, device) for image in read_pair(locate_scene(DATA, manifest, scene)))
    with torch.inference_mode():
        bo, od, bs, sd = model.encode_pair(optical, sar)
        branch = model.detail_fusion
        joint = branch.signed_joint(od, sd).float()
        with torch.autocast(device_type=device.type, enabled=False):
            low = branch.ssm.ll_projection(torch.cat((branch.bands(bo)[0].float(),
                                                     branch.bands(bs)[0].float()), dim=1)).tanh()
        profiles, conditions = make_profiles(branch.ssm, joint, low, lengths, scales, args.lines)
        spatial = make_spatial(branch.ssm, joint, low)
    summary = dict(status="diagnostic_only_no_optimizer_step", scene=scene, seed=42,
                   input_shape=list(optical.shape), scan_shape=list(joint.shape),
                   device=str(device), lengths=lengths, scales=scales, selected_lines_per_axis=args.lines,
                   checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                   script_sha256=sha256_file(Path(__file__)), frozen_e2_groups=loaded,
                   experimental_ssm="new random-initialized FSC state branch; no trained SSM checkpoint",
                   scale_semantics="initial state scaled; all SSM parameters and x fixed",
                   prefix_semantics="initial state from the full LL line, unchanged across shorter prefixes",
                   zero_semantics="zero state, not zero Base features",
                   shuffle_semantics="different LL lines for scan probes; rolled LL map for 2-D probe",
                   interpretation="Internal SSM sensitivity only; no final image metrics or quality claims",
                   full_spatial=spatial)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fsc-reachability-", dir=output.parent) as temp:
        pending = Path(temp) / "output"
        pending.mkdir()
        write_csv(pending / "profiles.csv", profiles)
        write_csv(pending / "counterfactuals.csv", conditions)
        (pending / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        pending.rename(output)
    print(f"Saved: {output.resolve()}")
    for row in profiles:
        if row["length"] == max(lengths) and row["scale"] == 0.1:
            print(f"{row['axis']} {row['direction']}: L={row['length']}, head={row['head_rms']:.3e}, "
                  f"middle={row['middle_rms']:.3e}, tail={row['tail_rms']:.3e}, "
                  f"tail/head={row['tail_over_head']:.1%}")
    print("2-D SSM correct/shuffled/zero: ",
          {k: round(spatial[k], 8) for k in ("correct_vs_zero_rms", "shuffled_vs_zero_rms", "correct_vs_shuffled_rms")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="Validation scene ID (default: first locked val64 scene)")
    parser.add_argument("--lengths", default="32,64,128", help="Scan prefix lengths")
    parser.add_argument("--scales", default="0,0.03,0.1,0.3,1.0", help="LL initial-state strengths")
    parser.add_argument("--lines", type=int, default=4, help="Representative lines per axis")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
