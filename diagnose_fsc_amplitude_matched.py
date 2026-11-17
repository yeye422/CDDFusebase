"""Fixed-gain FSC state-vs-feature SSM response on disjoint QXS validation scenes.

Calibrate RMS response on the first locked validation scene, then freeze that
scalar for distinct holdout scenes. The E2 backbone is loaded from a completed
checkpoint; the FSC SSM remains untrained. No optimizer or image-quality test.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_fsc_state_reachability import load_frozen_e2, ssm_with_low
from e2_fsc_local_refresh import CHUNK_SIZE, CARRY, LocalRefreshTransfer
from e2_fsc_mamba import CONTEXT_SCALE, FSCMambaE2
from fsc_amplitude_matching import calibrate_gain, response_stats, rms
from maintain_experiment_index import sha256_file
from run_e2_fsc_mamba import DATA, ORIGINAL_LOSS_PILOT, source_manifest


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("test_result/e2_fsc_amplitude_matched_v1")


def features(model, scene_id, manifest, device):
    optical, sar = (
        to_tensor(image, device)
        for image in read_pair(locate_scene(DATA, manifest, scene_id)))
    with torch.inference_mode():
        bo, od, bs, sd = model.encode_pair(optical, sar)
        branch = model.detail_fusion
        joint = branch.signed_joint(od, sd).float()
        previous = branch.ssm
        with torch.autocast(device_type=device.type, enabled=False):
            low = previous.ll_projection(torch.cat((
                branch.bands(bo)[0].float(),
                branch.bands(bs)[0].float()), dim=1)).tanh()
    return joint, low


def conditions(ssm, joint, low, gain, *, zero=None):
    """Full bidirectional 2-D SSM under identical Detail and SSM parameters."""
    with torch.inference_mode():
        if zero is None:
            zero = ssm_with_low(ssm, joint, torch.zeros_like(low))
        correct = ssm_with_low(ssm, joint, low * gain)
        shuffled = ssm_with_low(
            ssm, joint, low.roll((low.shape[-2] // 2,
                                 low.shape[-1] // 2), dims=(-2, -1)) * gain)
        stats = response_stats(correct, shuffled, zero)
    return stats, zero


def _effect(ssm, joint, low, gain, zero):
    with torch.inference_mode():
        response = ssm_with_low(ssm, joint, low * gain)
        if not bool(torch.isfinite(response).all()):
            raise ValueError(f"Non-finite state output at gain={gain}")
        return rms(response - zero)


def run(args):
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if args.holdouts < 1 or args.holdouts > 63:
        raise ValueError("--holdouts must be in [1,63]")
    if args.chunk_size < 2:
        raise ValueError("--chunk-size must be >= 2")
    manifest = source_manifest()
    ids = manifest["splits"]["val"]
    if len(ids) < args.holdouts + 1:
        raise ValueError("Insufficient locked val scenes")
    calibration_id = ids[0]
    holdout_ids = ids[1:1 + args.holdouts]
    output = args.output_dir or OUTPUT_ROOT / f"calib_{calibration_id}_seed42_holdout{args.holdouts}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite previous results: {output}")
    checkpoint = ORIGINAL_LOSS_PILOT.final
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing archived E2 checkpoint: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(42)
    model = FSCMambaE2("state")
    loaded = load_frozen_e2(model, checkpoint)
    model = model.to(device).eval()
    previous = model.detail_fusion.ssm
    feature_ssm = LocalRefreshTransfer(previous, "local_feature", args.chunk_size).eval()
    refresh_ssm = LocalRefreshTransfer(previous, "local_refresh", args.chunk_size).eval()

    rows = []
    print(f"Calibration scene={calibration_id}; holdouts={holdout_ids}", flush=True)
    joint, low = features(model, calibration_id, manifest, device)
    with torch.inference_mode():
        feature_zero = ssm_with_low(feature_ssm, joint, torch.zeros_like(low))
        state_zero = ssm_with_low(refresh_ssm, joint, torch.zeros_like(low))
        feature_correct = ssm_with_low(feature_ssm, joint, low)
        target = rms(feature_correct - feature_zero)
        calibration = calibrate_gain(
            lambda gain: _effect(refresh_ssm, joint, low, gain, state_zero),
            target, max_gain=args.max_gain, tolerance=args.tolerance)
    gain = calibration["gain"]
    if not calibration["matched"]:
        print("WARNING: calibration did not achieve requested amplitude;"
              " holdout results must not be called matched.", flush=True)
    print(f"Gain={gain:.5g}; target={target:.6g}; state={calibration['matched_state_rms']:.6g};"
          f" relative error={calibration['relative_error']:.2%}", flush=True)

    def collect(scene, subset, joint, low, reference_target=None):
        with torch.inference_mode():
            for mode, ssm, weight in (
                    ("feature", feature_ssm, 1.0),
                    ("refresh_gain1", refresh_ssm, 1.0),
                    ("refresh_matched", refresh_ssm, gain)):
                measurements, _ = conditions(ssm, joint, low, weight)
                rows.append(dict(scene=str(scene), subset=subset,
                                 mode=mode, chunk_size=args.chunk_size,
                                 gain=weight, **measurements,
                                 reference_target_rms=(reference_target or 0.0),
                                 match_ratio_to_feature=None))
            feat = rows[-3]["correct_vs_zero_rms"]
            for row in rows[-3:]:
                row["reference_target_rms"] = feat
                row["match_ratio_to_feature"] = row["correct_vs_zero_rms"] / max(feat, 1e-12)
        print(f"{scene} ({subset}): feature={feat:.3e}, state_gain1={rows[-2]['correct_vs_zero_rms']:.3e},"
              f" fixed_gain={rows[-1]['correct_vs_zero_rms']:.3e},"
              f" gain/feature={rows[-1]['match_ratio_to_feature']:.2f}", flush=True)

    collect(calibration_id, "calibration", joint, low, target)
    del joint, low
    for scene in holdout_ids:
        joint, low = features(model, scene, manifest, device)
        collect(scene, "holdout", joint, low)
        del joint, low

    summary = dict(
        status="diagnostic_only_no_optimizer_step",
        calibration_scene=calibration_id,
        holdout_scenes=holdout_ids,
        seed=42,
        device=str(device),
        trained_checkpoint=checkpoint.as_posix(),
        checkpoint_sha256=sha256_file(checkpoint),
        sources={name:sha256_file(Path(name)) for name in (
            "diagnose_fsc_amplitude_matched.py",
            "fsc_amplitude_matching.py",
            "e2_fsc_mamba.py",
            "e2_fsc_local_refresh.py",
            "diagnose_fsc_state_reachability.py")},
        loaded_frozen_e2_groups=loaded,
        chunk_size=args.chunk_size,
        carry=CARRY,
        context_scale=CONTEXT_SCALE,
        calibration=calibration,
        matching_scope="One scalar fitted on only first val64 scene; unchanged on all holdouts",
        normalization="Correct LL minus zero LL RMS at the full bidirectional 2-D SSM output",
        shuffled="Same low map rolled half a scan grid in both spatial axes",
        caveat="Untrained SSM; output response amplitude is not structure validity, fusion quality or useful correspondence",
        per_scene_holdout_match_ratios=[row["match_ratio_to_feature"] for row in rows
                                        if row["subset"]=="holdout" and row["mode"]=="refresh_matched"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fsc-matched-", dir=output.parent) as temporary:
        pending = Path(temporary) / "result"
        pending.mkdir()
        with (pending / "comparisons.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (pending / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.rename(output)
    print(f"Saved {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--max-gain", type=float, default=512.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
