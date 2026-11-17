"""Frozen-E2 counterfactual test of LL local-state refresh against LL features.

The E2 encoder/decoder/LWPD/fusion keep path are loaded from a real checkpoint;
the SSM branch is held at a shared seed-42 initialization. No optimizer step.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_fsc_state_reachability import (
    load_frozen_e2, rms, segment_rms, selected_sequences, ssm_with_low,
)
from e2_fsc_local_refresh import CARRY, CHUNK_SIZE, LocalRefreshTransfer
from e2_fsc_mamba import CONTEXT_SCALE, FSCMambaE2, scan_with_state
from maintain_experiment_index import sha256_file
from run_e2_fsc_mamba import DATA, ORIGINAL_LOSS_PILOT, source_manifest


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("test_result/e2_fsc_local_refresh_diagnosis_v1")


def score(correct, wrong, zero):
    delta = correct - zero
    head, middle, tail = segment_rms(delta)
    amplitude = rms(zero)
    return dict(correct_vs_zero_rms=rms(delta),
                shuffled_vs_zero_rms=rms(wrong - zero),
                correct_vs_shuffled_rms=rms(correct - wrong),
                zero_rms=amplitude,
                relative_to_zero=rms(delta) / max(amplitude, 1e-12),
                head_rms=head, middle_rms=middle, tail_rms=tail,
                tail_over_head=tail / max(head, 1e-12))


def scan_direction(ssm, scan, x, low, mode):
    if mode == "single_state":
        initial = ssm.initial_state(
            CONTEXT_SCALE * ssm.terminal(low), scan)
        return scan_with_state(scan, x, initial)
    return ssm.scan_one(x, low, scan)


def probe_lines(previous, joint, low, chunk_sizes, line_count):
    rows = []
    for chunk_size in chunk_sizes:
        for mode in ("single_state", "local_feature", "local_refresh"):
            ssm = (previous if mode == "single_state"
                   else LocalRefreshTransfer(previous, mode, chunk_size))
            for axis in ("row", "column"):
                scan = (ssm.core.row_scan if axis == "row"
                        else ssm.core.column_scan)
                x, ll, ids = selected_sequences(
                    joint, low, axis, line_count)
                for direction in ("forward", "reverse"):
                    a, b = ((x, ll) if direction == "forward"
                            else (x.flip(1), ll.flip(1)))
                    correct = scan_direction(ssm, scan, a, b, mode)
                    wrong = scan_direction(
                        ssm, scan, a,
                        b.roll(max(1, line_count // 2), dims=0), mode)
                    zero = scan_direction(
                        ssm, scan, a, torch.zeros_like(b), mode)
                    rows.append(dict(
                        mode=mode, chunk_size=chunk_size, carry=CARRY,
                        axis=axis, direction=direction,
                        lines=";".join(map(str, ids)), length=a.shape[1],
                        **score(correct, wrong, zero)))
    return rows


def probe_spatial(previous, joint, low, chunk_sizes):
    rows = []
    for chunk_size in chunk_sizes:
        for mode in ("single_state", "local_feature", "local_refresh"):
            ssm = (previous if mode == "single_state"
                   else LocalRefreshTransfer(previous, mode, chunk_size))
            correct = ssm_with_low(ssm, joint, low)
            wrong_ll = low.roll(
                (low.shape[-2] // 2, low.shape[-1] // 2),
                dims=(-2, -1))
            wrong = ssm_with_low(ssm, joint, wrong_ll)
            zero = ssm_with_low(ssm, joint, torch.zeros_like(low))
            rows.append(dict(
                mode=mode, chunk_size=chunk_size, carry=CARRY,
                correct_vs_zero_rms=rms(correct - zero),
                shuffled_vs_zero_rms=rms(wrong - zero),
                correct_vs_shuffled_rms=rms(correct - wrong),
                zero_rms=rms(zero),
                correct_to_zero_relative=(
                    rms(correct - zero) / max(rms(zero), 1e-12))))
    return rows


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    torch.set_num_threads(2)
    chunk_sizes = [
        int(value) for value in args.chunk_sizes.split(",")]
    if (not chunk_sizes or len(set(chunk_sizes)) != len(chunk_sizes)
            or any(x < 2 for x in chunk_sizes)):
        raise ValueError(
            "--chunk-sizes must be distinct integers >= 2")
    if args.lines < 2:
        raise ValueError("--lines must be >= 2")
    manifest = source_manifest()
    scene = args.scene or manifest["splits"]["val"][0]
    if scene not in manifest["splits"]["val"]:
        raise ValueError("Selected scene is not in locked val64")
    output = (args.output_dir or OUTPUT_ROOT /
              f"scene_{scene}_seed42")
    if output.exists():
        raise FileExistsError(
            f"Will not overwrite existing results: {output}")
    checkpoint = ORIGINAL_LOSS_PILOT.final
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing archived E2: {checkpoint}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "cpu")
    torch.manual_seed(42)
    model = FSCMambaE2("state")
    loaded = load_frozen_e2(model, checkpoint)
    model = model.to(device).eval()
    optical, sar = (
        to_tensor(img, device)
        for img in read_pair(locate_scene(DATA, manifest, scene)))
    with torch.inference_mode():
        bo, od, bs, sd = model.encode_pair(optical, sar)
        branch = model.detail_fusion
        joint = branch.signed_joint(od, sd).float()
        previous = branch.ssm
        with torch.autocast(
                device_type=device.type, enabled=False):
            low = previous.ll_projection(torch.cat((
                branch.bands(bo)[0].float(),
                branch.bands(bs)[0].float()), dim=1)).tanh()
        profiles = probe_lines(
            previous, joint, low, chunk_sizes, args.lines)
        spatial = probe_spatial(
            previous, joint, low, chunk_sizes)
    summary = dict(
        status="diagnostic_only_no_optimizer_step",
        scene=str(scene), seed=42,
        checkpoint=checkpoint.as_posix(),
        checkpoint_sha256=sha256_file(checkpoint),
        diagnostic_sha256=sha256_file(Path(__file__)),
        device=str(device), image_shape=list(optical.shape),
        ssm_shape=list(joint.shape),
        chunk_sizes=chunk_sizes, carry=CARRY,
        context_scale=CONTEXT_SCALE,
        selected_lines_per_axis=args.lines,
        frozen_e2_groups=loaded,
        ssm_initialization=(
            "shared untrained seed42 FSC module, no parameter updates"),
        conditioning=(
            "correct LL / position-shuffled LL / zero projected LL; Detail unchanged"),
        interpretation=(
            "SSM sensitivity, NOT fusion quality or structural validity"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
            prefix=".fsc-refresh-", dir=output.parent) as temporary:
        pending = Path(temporary) / "result"
        pending.mkdir()
        write_csv(pending / "profiles.csv", profiles)
        write_csv(pending / "spatial.csv", spatial)
        (pending / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        pending.rename(output)
    print(f"Saved: {output.resolve()}", flush=True)
    for row in spatial:
        print(f"{row['mode']:>14s}, chunk={row['chunk_size']:>2}: "
              f"correct/zero={row['correct_to_zero_relative']:.4%}, "
              f"correct/shuffled={row['correct_vs_shuffled_rms']:.3e}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", help="Scene ID in the locked val64")
    parser.add_argument(
        "--chunk-sizes", default="16,32")
    parser.add_argument("--lines", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
