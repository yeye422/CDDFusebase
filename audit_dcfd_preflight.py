"""Check DCFD initialization and qualify the precision on 32 real update attempts."""

import argparse
import gc
import os
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

import train_lwpd as trainer
from dcfd_losses import DCFDLoss, full_support
from e2_cfdr import CFDRE2
from e2_dcfd import DCFDE2, Reconstruction
from run_dcfd_frozen import (
    DATA, MANIFEST, PREFLIGHT, PREFLIGHT_UPDATES, SEED, atomic_write, device_identity, frozen_state,
    inputs, require_unchanged, setup_optimizer, snapshot, source_pair, train_loader, train_pass,
)
from run_qxs_reference_test1000 import read_json, write_json
from utils.dataset_os import manifest_pairs


def storage_keys(module):
    return {p.untyped_storage().data_ptr() for p in (*module.parameters(), *module.buffers()) if p.numel()}


@torch.inference_mode()
def initialization(parent, device):
    splits, _ = manifest_pairs(DATA, MANIFEST)
    optical, sar = source_pair(splits["val"][0], device)[-1]
    student, teacher = DCFDE2.initialized(parent)
    student.to(device).eval()
    teacher.to(device)
    standard = CFDRE2("CF1").to(device).eval()
    standard.load_state_dict(parent["model"], strict=True)
    storage = [storage_keys(module) for module in (student.decoder.structure, student.decoder.detail, teacher)]
    if any(storage[i] & storage[j] for i in range(3) for j in range(i)) or storage[2] & storage_keys(student):
        raise AssertionError("DCFD paths or teacher share parameter storage")
    if any(isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) for module in student.decoder.dsdi.modules()):
        raise AssertionError("DSDI must preserve source detail resolution")
    shared, encoded, fused_detail = student.prepare(optical, sar)
    detail_input = torch.cat((encoded[1], encoded[3]), dim=1)
    if student.decoder.dsdi(detail_input).shape != shared.shape:
        raise AssertionError("DSDI changed full-resolution geometry")
    actual, structure, detail = student.decoder(shared, encoded, fused_detail, optical, sar)
    reference = teacher(shared, fused_detail, optical, sar)
    standard_base, standard_detail = standard.fuse_features(*standard.encode_pair(optical, sar))
    hidden = standard.decoder.encoder_level2(standard.decoder.reduce_channel(torch.cat((standard_base, standard_detail), dim=1)))
    logits = standard.decoder.output(hidden)
    if standard.uses_image_skip:
        logits = logits + optical
    prediction = standard.decoder.sigmoid(logits)
    raw = standard.cfdr(prediction, hidden, standard_detail, optical, sar)
    source_images = torch.cat((prediction, optical, sar), dim=1)
    base, means = standard.cfdr.smooth(source_images).split((1, 2), dim=1)
    gain = standard.cfdr.gain(standard.cfdr.condition(torch.cat((hidden, standard_detail), dim=1)))
    detail_ref = (gain.softmax(dim=1, dtype=torch.float32)[:, :2] * (source_images[:, 1:] - means)).sum(dim=1, keepdim=True)
    comparisons = dict(base=(actual.base, base), detail=(actual.detail, detail_ref), raw=(actual.raw, raw),
                       fused=(actual.fused, standard(optical, sar).clamp(0, 1)),
                       teacher_base=(reference.base, base), teacher_detail=(reference.detail, detail_ref),
                       teacher_raw=(reference.raw, raw), teacher_fused=(reference.fused, raw.clamp(0, 1)),
                       hidden_structure=(structure, hidden), hidden_detail=(detail, hidden))
    errors = {name: float((actual - expected).abs().max()) for name, (actual, expected) in comparisons.items()}
    if not all(error < 1e-6 for error in errors.values()):
        raise AssertionError(f"DCFD initialization differs from CF1: {errors}")
    return dict(scene=splits["val"][0][0].stem, max_abs_error=errors, parameter_storage_independent=True,
                full_resolution_detail=True)


def check_masked_loss(device):
    loss = DCFDLoss().to(device)
    optical, sar = torch.rand(2, 1, 1, 64, 64, device=device).unbind()
    valid = torch.ones_like(sar, dtype=torch.bool)
    partial = valid.clone()
    partial[..., 32, 32] = False
    empty = torch.zeros_like(valid)
    counts = {}
    for name, mask in (("valid", valid), ("partial", partial), ("empty", empty)):
        smooth = full_support(mask, 5)
        masks = [mask, *(F.avg_pool2d(smooth.float(), factor) == 1 for factor in (2, 4))]
        counts[name] = [int(full_support(window, 11).sum()) for window in masks]
        masked_sar = sar * mask
        raw = torch.full_like(optical, .4, requires_grad=True)
        ref = optical * .9
        source_base = torch.cat((optical, masked_sar), dim=1)
        terms = Reconstruction(raw, raw * .1, raw, raw.clamp(0, 1), source_base, raw)
        target = Reconstruction(ref, ref * .1, ref, ref, source_base, ref)
        total, stats = loss(terms, target, optical, masked_sar, mask)
        total.backward()
        if not all(torch.isfinite(value).all() for value in (total, *stats.values(), raw.grad)):
            raise FloatingPointError(f"Non-finite masked DCFD loss or gradient: {name}")
        if name == "empty":
            expected = (1 - loss.ssim_map(terms.fused, optical)).mean()
            torch.testing.assert_close(stats["loss_full"], expected)
            expected = sum((1 - loss.ssim_map(F.avg_pool2d(raw, k), F.avg_pool2d(optical, k))).mean()
                           for k in (2, 4)) / 2
            torch.testing.assert_close(stats["loss_base"], expected)
    expected = dict(valid=[4096, 1024, 256], partial=[4096-121, 1024-169, 256-144], empty=[0, 0, 0])
    if counts != expected:
        raise AssertionError(f"Unexpected SAR support propagation: {counts}")
    return counts


class UpdateAudit:
    def __init__(self, model):
        decoder = model.decoder
        projections = dict(dsbi=decoder.dsbi[-1], dsdi=decoder.dsdi[-1],
                           structure_to_detail=decoder.bpc.s_to_d[-1], detail_to_structure=decoder.bpc.d_to_s)
        upstream = dict(dsbi_input=decoder.dsbi[0], dsbi_spatial=decoder.dsbi[2],
                        dsdi_input=decoder.dsdi[0], dsdi_spatial=decoder.dsdi[2],
                        structure_to_detail_input=decoder.bpc.s_to_d[0],
                        structure_to_detail_spatial=decoder.bpc.s_to_d[2],
                        detail_hint_spatial=decoder.bpc.detail_hint[0], detail_hint_output=decoder.bpc.detail_hint[1])
        self.parameters = {name: module.weight for name, module in (projections | upstream).items()}
        self.projections = tuple(projections)
        self.upstream = tuple(upstream)
        if any(torch.count_nonzero(p) for module in projections.values() for p in (module.weight, module.bias)):
            raise AssertionError("DCFD residual projections must be exactly zero-initialized")
        self.first_gradient = {}
        self.first_update = {}
        self.before = {}

    def __call__(self, step, *, before):
        if before:
            for name, parameter in self.parameters.items():
                if name in self.first_update or parameter.grad is None:
                    continue
                if torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad):
                    self.first_gradient.setdefault(name, step)
                    self.before[name] = parameter.detach().clone()
        else:
            for name, value in self.before.items():
                if not torch.equal(self.parameters[name].detach(), value):
                    self.first_update[name] = step
            self.before.clear()

    def report(self):
        passed = (all(self.first_gradient.get(name) == self.first_update.get(name) == 1 for name in self.projections)
                  and all(self.first_gradient.get(name, 0) >= 2 and name in self.first_update for name in self.upstream))
        return dict(passed=passed, first_nonzero_gradient=self.first_gradient, first_parameter_update=self.first_update)


def qualify(parent, device, precision):
    trainer.set_seed(SEED)
    model, teacher = DCFDE2.initialized(parent)
    frozen = frozen_state(model)
    teacher_state = snapshot(teacher)
    model.to(device)
    teacher.to(device)
    observer = UpdateAudit(model)
    optimizer, scaler = setup_optimizer(model, precision)
    trainer.set_seed(SEED + 1)
    stats = train_pass(model, teacher, train_loader(1, device), device, precision, optimizer, scaler,
                       PREFLIGHT_UPDATES, observer)
    require_unchanged(frozen, model)
    require_unchanged(teacher_state, teacher)
    gradients = observer.report()
    return dict(**stats, gradient_audit=gradients, frozen_unchanged=True, teacher_unchanged=True,
                effective_update_rate=stats["effective_updates"] / PREFLIGHT_UPDATES,
                passed=gradients["passed"] and stats["effective_updates"] / PREFLIGHT_UPDATES >= .99)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One validation scene and synthetic mask checks")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DCFD preflight")
    device = torch.device(args.device)
    parent, protocol = inputs()
    identity = device_identity(device)
    if not args.smoke and PREFLIGHT.exists():
        saved = read_json(PREFLIGHT)
        if saved["protocol"] == protocol and saved["device"] == identity and saved["passed"]:
            print(f"DCFD preflight already passed: {saved['selected_precision']}, {PREFLIGHT}", flush=True)
            return
        raise FileExistsError(f"Preserve the existing DCFD preflight: {PREFLIGHT}")
    trainer.set_seed(SEED)
    report = dict(protocol=protocol, device=identity, initialization=initialization(parent, device),
                  masks=check_masked_loss(device))
    print(f"FP32 initialization and SAR mask checks passed: {report['initialization']}, {report['masks']}", flush=True)
    if args.smoke:
        return
    report.update(precision={}, selected_precision=None, passed=False, update_attempts=PREFLIGHT_UPDATES)
    candidates = ("fp16", "bf16", "fp32") if torch.cuda.is_bf16_supported() else ("fp16", "fp32")
    for precision in candidates:
        print(f"Checking {precision}, {PREFLIGHT_UPDATES} optimizer attempts", flush=True)
        try:
            result = qualify(parent, device, precision)
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            result = dict(passed=False, error=str(error))
        gc.collect()
        torch.cuda.empty_cache()
        report["precision"][precision] = result
        if result["passed"]:
            report.update(selected_precision=precision, passed=True)
            break
    path = PREFLIGHT if report["passed"] else PREFLIGHT.with_name(f"preflight-failed-{uuid4().hex}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda temporary: write_json(temporary, report))
    if not report["passed"]:
        raise RuntimeError(f"DCFD precision preflight failed: {path}")
    print(f"DCFD locked precision: {report['selected_precision']}, {path}", flush=True)


if __name__ == "__main__":
    main()
