"""Audit IPMF-V1 inputs and run the short CUDA precision and gradient checks."""

import argparse
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

import train_lwpd as trainer
from e2_ascd import ASCDE2
from e2_cfdr import CFDRE2, VARIANTS
from e2_ipmf import IPMFE2, require_frozen_cf1, unshuffle
from hcfdr_comparison import EVALUATOR_FILES
from run_cfdr_frozen import PARENT as CF0, checkpoint_path, load_backbone
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, write_json
from train_direct_fusion import deterministic_subset, sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


PARENT = checkpoint_path("CF1", 4)
ASCD_CHECKPOINT = Path("models/e2_ascd_v1_frozen/seed42/epoch_004.pth")
OUTPUT = Path("test_result/ipmf_preflight_v2/report.json")
LR = 1e-4
UPDATE_ATTEMPTS = 32
SCALER_WARMUP = 8
MIN_STABLE_UPDATE_RATE = .99
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SOURCES = ("audit_ipmf_preflight.py", "e2_ipmf.py", "e2_cfdr.py", "e2_ascd.py", "e2_base_anchor.py",
           "direct_fusion.py", "net.py", "lwpd.py", "train_lwpd.py", "train_direct_fusion.py",
           "utils/dataset_os.py", "utils/loss.py")


def load_parent():
    cf0, args = load_backbone()
    recipe = (args.seed, args.batch_size, args.accumulation_steps, args.patch_size, args.gradient_weight,
              args.weight_decay, args.clip_grad, args.invalid_threshold, args.num_workers)
    if recipe != (42, 2, 4, 256, 10., 1e-4, 1., 0, 0):
        raise ValueError(f"Unexpected CF1 recipe: {recipe}")
    manifest = read_json(MANIFEST)
    if manifest["seed"] != 20260924 or [len(manifest["splits"][split]) for split in ("train", "val", "test")] != [
        8000, 1000, 1000
    ]:
        raise ValueError("Expected the locked QXS 8000/1000/1000 split")
    config = PARENT.parent / "config.json"
    saved = torch.load(PARENT, map_location="cpu", weights_only=True)
    settings = read_json(config)
    cf0_hash = sha256_file(CF0)
    protocol = dict(parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=sha256_file(PARENT),
                    cf0_checkpoint=CF0.as_posix(), cf0_checkpoint_sha256=cf0_hash,
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=sha256_file(MANIFEST),
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8",
                    evaluator_sha256={name: sha256_file(Path(name)) for name in EVALUATOR_FILES},
                    source_sha256={name: sha256_file(Path(name)) for name in SOURCES})
    if ((saved["arm"], saved["epoch"], saved["variant"], saved["config_sha256"])
            != ("CF1", 4, VARIANTS["CF1"], sha256_file(config))
            or settings["training_policy"] != "cfdr_only_from_converged_cf0"
            or Path(settings["parent_checkpoint"]).resolve() != CF0.resolve()
            or settings["parent_checkpoint_sha256"] != cf0_hash
            or settings["split_manifest_sha256"] != protocol["split_manifest_sha256"]):
        raise ValueError(f"Unexpected CF1 checkpoint or provenance: {PARENT}")
    inherited = {name for name in saved["model"] if not name.startswith("cfdr.")}
    if inherited != cf0.keys() or any(not torch.equal(saved["model"][name], value) for name, value in cf0.items()):
        raise ValueError("CF1 inherited parameters or buffers differ from the recorded CF0 checkpoint")
    protocol["cf0_tensors_verified"] = len(inherited)
    return saved, args, protocol


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def batch(items, device):
    return tuple(torch.stack(tensors).to(device) for tensors in zip(*(item[:3] for item in items)))


@torch.no_grad()
def input_audit(model, ascd, ipmf, dataset, device):
    rows = []
    for item in dataset:
        optical, sar, _ = batch([item], device)
        encoded = model.encode_pair(optical, sar)
        bo, do, bs, ds = encoded
        bf, _ = model.fuse_features(*encoded)
        features = {}
        for name, detail, ascd_base, ascd_projection, ipmf_base, ipmf_projection in (
            ("optical", do, bf, ascd.ascd.optical_projection, bo, ipmf.optical_projection),
            ("sar", ds, bs, ascd.ascd.sar_projection, bs, ipmf.sar_projection),
        ):
            packed = unshuffle(detail)
            restored = F.pixel_shuffle(packed, 2)
            torch.testing.assert_close(restored, detail, rtol=0, atol=0)
            pooled = F.avg_pool2d(detail, 2)
            zeros = torch.zeros_like(detail)
            prior = F.avg_pool2d(ascd_projection(torch.cat((ascd_base, detail), dim=1)), 2)
            zero = F.avg_pool2d(ascd_projection(torch.cat((ascd_base, zeros), dim=1)), 2)
            projected = ipmf_projection(unshuffle(torch.cat((ipmf_base, detail), dim=1)))
            projected_zero = ipmf_projection(unshuffle(torch.cat((ipmf_base, zeros), dim=1)))
            response = rms(projected - projected_zero)
            floor = 16 * torch.finfo(projected.dtype).eps * max(rms(projected), rms(projected_zero))
            features[name] = dict(detail_rms=rms(detail), pooled_rms=rms(pooled), packed_rms=rms(packed),
                                  detail_elements=detail.numel(), pooled_elements=pooled.numel(),
                                  packed_elements=packed.numel(), inverse_max_error=float((restored - detail).abs().max()),
                                  ascd_input_delta_rms=rms(prior - zero), ascd_input_rms=rms(prior),
                                  ipmf_projection_delta_rms=response, ipmf_projection_noise_floor=floor,
                                  ipmf_projection_pass=math.isfinite(response) and response > floor)
        rows.append(dict(scene=item[3], features=features))
    return rows


@torch.no_grad()
def mask_audit(losses, items, device):
    rows = []
    for item in items:
        optical, sar, valid = batch([item], device)
        if not torch.equal(valid.bool(), sar > 0):
            raise AssertionError("Dataset mask differs from the configured grayscale > 0 rule")
        invalid = ~valid.bool()
        # Sobel has no central tap. A valid center is affected by invalid neighbors in its 3x3 stencil.
        touches = (F.max_pool2d(invalid.float(), 3, stride=1, padding=1) > 0) & ~invalid
        optical_gradient = losses.sobel(optical)
        sar_gradient = losses.sobel(sar)
        target = torch.maximum(optical_gradient, sar_gradient)
        affected = touches & (sar_gradient > optical_gradient)
        # Ablate the SAR gradient contribution only on touched centers for the audit.
        # This measures how much of the current objective comes from those sites.
        target_delta = torch.where(touches, target - optical_gradient, 0.)
        rows.append(dict(scene=item[3], pixels=valid.numel(), invalid=int(invalid.sum()),
                         valid_touching_invalid=int(touches.sum()), touched_where_sar_wins=int(affected.sum()),
                         touched_sar_target_excess_mean=float(target_delta.mean())))
    return rows


def finite_norm(parameters):
    gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.


def update_window(attempts, updates):
    return dict(attempts=attempts, effective_updates=updates, skipped_updates=attempts - updates,
                effective_rate=updates / attempts if attempts else None,
                skipped_rate=(attempts - updates) / attempts if attempts else None)


def precision_audit(parent, args, items, device, precision, *, attempts=UPDATE_ATTEMPTS, warmup=SCALER_WARMUP,
                    additional_identity_items=()):
    items_per_update = args.batch_size * args.accumulation_steps
    if len(items) != attempts * items_per_update or not 0 <= warmup < attempts:
        raise ValueError("Precision audit requires one complete, fixed sample schedule")
    trainer.set_seed(args.seed)
    model = IPMFE2().to(device).train()
    model.load_cf1(parent)
    require_frozen_cf1(model.state_dict(), parent["model"])
    baseline = CFDRE2("CF1").to(device).eval()
    baseline.load_state_dict(parent["model"])
    frozen = [child for name, child in model.named_children() if name != "ipmf"]
    if any(m.training for child in frozen for m in child.modules()):
        raise AssertionError("A frozen CF1 module entered training mode")
    modules = dict(model.ipmf.named_children())
    parameters = list(model.ipmf.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=args.weight_decay)
    scaler = trainer.GradScaler(enabled=precision == "fp16")
    losses = trainer.DirectFusionLosses().to(device)
    identity_errors = []
    identity_items = [*items, *additional_identity_items]
    with torch.no_grad():
        for index in range(0, len(identity_items), args.batch_size):
            current = identity_items[index:index + args.batch_size]
            optical, sar, _ = batch(current, device)
            with torch.autocast("cuda", enabled=precision != "fp32", dtype=DTYPES[precision]):
                expected, actual = baseline(optical, sar), model(optical, sar)
            errors = (actual - expected).abs().flatten(1).amax(dim=1).tolist()
            identity_errors.extend(dict(scene=item[3], max_error=value if math.isfinite(value) else None)
                                   for item, value in zip(current, errors))
    if any(row["max_error"] != 0 for row in identity_errors):
        return dict(eligible=False, identity_checked_samples=len(identity_errors),
                    initialization=identity_errors, failure="The initial IPMF and CF1 outputs were not identical")
    del baseline
    steps = []
    updates = 0
    reached = set()

    def stepped(*_):
        nonlocal updates
        updates += 1

    handle = optimizer.register_step_post_hook(stepped)
    for step in range(attempts):
        optimizer.zero_grad(set_to_none=True)
        before = {name: [p.detach().clone() for p in layer.parameters()] for name, layer in modules.items()}
        total = 0.
        for index in range(step * items_per_update, (step + 1) * items_per_update, args.batch_size):
            optical, sar, valid = batch(items[index:index + args.batch_size], device)
            with torch.autocast("cuda", enabled=precision != "fp32", dtype=DTYPES[precision]):
                fused = model(optical, sar)
                loss = losses.fusion(optical, sar, fused, valid, args.gradient_weight)[0]
            total += float(loss.detach()) / args.accumulation_steps
            scaler.scale(loss / args.accumulation_steps).backward()
        scaler.unscale_(optimizer)
        norms = {name: finite_norm(layer.parameters()) for name, layer in modules.items()}
        missing = [name for name, p in model.ipmf.named_parameters() if p.grad is None]
        pre = finite_norm(parameters)
        finite = math.isfinite(total) and math.isfinite(pre)
        previous = updates
        post = None
        if finite:
            torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
            post = finite_norm(parameters)
            scaler.step(optimizer)
        if scaler.is_enabled():
            scaler.update()
        if updates > previous:
            reached.update(name for name, value in norms.items() if value > 0 and math.isfinite(value))
        delta = {name: math.sqrt(sum(float((p.detach() - old).float().square().sum())
                                        for p, old in zip(layer.parameters(), before[name])))
                 for name, layer in modules.items()}
        steps.append(dict(step=step + 1, loss=total if math.isfinite(total) else None, finite_gradients=math.isfinite(pre),
                          grad_pre_clip=pre if math.isfinite(pre) else None, grad_post_clip=post,
                          module_grad_norms={key: value if math.isfinite(value) else None for key, value in norms.items()},
                          module_update_l2={key: value if math.isfinite(value) else None for key, value in delta.items()},
                          missing_gradients=missing, optimizer_updated=updates > previous, scale=scaler.get_scale()))
    handle.remove()
    require_frozen_cf1(model.state_dict(), parent["model"])
    if any(p.grad is not None or p.requires_grad for name, p in model.named_parameters() if not name.startswith("ipmf.")):
        raise AssertionError("An inherited CF1 parameter is receiving a gradient")
    warmup_stats = update_window(warmup, sum(row["optimizer_updated"] for row in steps[:warmup]))
    stable = update_window(attempts - warmup, sum(row["optimizer_updated"] for row in steps[warmup:]))
    stable_enough = stable["effective_rate"] >= MIN_STABLE_UPDATE_RATE if precision == "fp16" else updates == attempts
    eligible = (stable_enough and reached == set(modules)
                and all(row["loss"] is not None and not row["missing_gradients"] for row in steps)
                and steps[-1]["optimizer_updated"] and all(any(
                    row["module_update_l2"][name] is not None and row["module_update_l2"][name] > 0 for row in steps
                ) for name in modules))
    return dict(eligible=eligible, initial_max_error=0., identity_checked_samples=len(identity_errors),
                initialization=identity_errors, effective_updates=updates, warmup=warmup_stats, stable=stable,
                skipped_updates=len(steps) - updates, modules_with_gradient=sorted(reached),
                frozen_weights_match=True, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    cli = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the IPMF preflight")
    device = torch.device(cli.device)
    parent, args, protocol = load_parent()
    if OUTPUT.exists():
        saved = read_json(OUTPUT)
        if saved["protocol"] != protocol:
            raise FileExistsError(f"Preserve the previous preflight: {OUTPUT}")
        if saved["selected_precision"] is None:
            raise RuntimeError(f"No eligible precision in {OUTPUT}")
        print(f"Existing preflight: {saved['selected_precision']} in {OUTPUT}; SHA-256: {sha256_file(OUTPUT)}", flush=True)
        return
    splits, _ = manifest_pairs(DATA, MANIFEST)
    dataset = OpticalSARDataset(splits["val"][:8], patch_size=args.patch_size,
                               invalid_threshold=args.invalid_threshold, augment=False)
    val_items = [dataset[index] for index in range(len(dataset))]
    train_count = UPDATE_ATTEMPTS * args.batch_size * args.accumulation_steps
    train_pairs = deterministic_subset(splits["train"], train_count, args.seed)
    train_data = OpticalSARDataset(train_pairs, patch_size=args.patch_size, training=True,
                                   invalid_threshold=args.invalid_threshold, min_valid_fraction=args.min_valid_fraction)
    trainer.set_seed(args.seed)
    train_items = [train_data[index] for index in range(len(train_data))]
    if len({item[3] for item in train_items}) != train_count:
        raise ValueError("Precision audit requires distinct training scenes")
    trainer.set_seed(args.seed)
    ipmf = IPMFE2().ipmf.to(device).eval()
    model = CFDRE2("CF1").to(device).eval()
    model.load_state_dict(parent["model"])
    ascd = ASCDE2().to(device).eval()
    ascd_state = torch.load(ASCD_CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    ascd_frontend = ASCDE2.frontend_state(ascd_state)
    cf1_frontend = ASCDE2.frontend_state(parent["model"])
    if ascd_frontend.keys() != cf1_frontend.keys() or any(
        not torch.equal(ascd_frontend[key], value) for key, value in cf1_frontend.items()
    ):
        raise ValueError("The ASCD intervention requires the same frozen Encoder, LWPD and Base Fusion")
    ascd.load_state_dict(ascd_state)
    rows = input_audit(model, ascd, ipmf, val_items, device)
    projected = all(values["ipmf_projection_pass"] for row in rows for values in row["features"].values())
    del model, ascd, ipmf
    mask_rows = mask_audit(trainer.DirectFusionLosses().to(device), [*train_items, *val_items], device)
    precision = {}
    if projected:
        modes = ("fp32", "fp16", "bf16") if torch.cuda.is_bf16_supported() else ("fp32", "fp16")
        for name in modes:
            print(f"Auditing {name}: {UPDATE_ATTEMPTS} optimizer attempts on {train_count} training scenes", flush=True)
            precision[name] = precision_audit(parent, args, train_items, device, name, additional_identity_items=val_items)
    selected = (next((name for name in ("fp16", "bf16", "fp32") if precision.get(name, {}).get("eligible")), None)
                if precision.get("fp32", {}).get("eligible") else None)
    mask_summary = {key: sum(row[key] for row in mask_rows) for key in (
        "pixels", "invalid", "valid_touching_invalid", "touched_where_sar_wins",
    )}
    mask_summary["review_required"] = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    write_json(temporary, dict(protocol=protocol, device=torch.cuda.get_device_name(device), inputs=rows,
                               ipmf_projection_pass=projected, precision_training_scenes=[item[3] for item in train_items],
                               precision_update_attempts=UPDATE_ATTEMPTS, scaler_warmup_attempts=SCALER_WARMUP,
                               minimum_stable_update_rate=MIN_STABLE_UPDATE_RATE,
                               ascd_checkpoint=ASCD_CHECKPOINT.as_posix(),
                               ascd_checkpoint_sha256=sha256_file(ASCD_CHECKPOINT), mask_summary=mask_summary,
                               mask_scenes=mask_rows, precision=precision, selected_precision=selected))
    temporary.replace(OUTPUT)
    print(f"Selected precision: {selected}; preflight: {OUTPUT}; SHA-256: {sha256_file(OUTPUT)}", flush=True)
    if selected is None:
        raise RuntimeError("No training precision passed the IPMF preflight")


if __name__ == "__main__":
    main()
