"""Train and evaluate the single frozen-front-end DCFD-V1 configuration."""

import argparse
from contextlib import contextmanager
import csv
from itertools import islice
import math
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np
import torch

import train_lwpd as trainer
from audit_ipmf_preflight import load_parent
from dcfd_losses import DCFDLoss
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cfdr import CFDRE2
from e2_dcfd import DCFDE2
from hcfdr_comparison import EVALUATOR_FILES, OUTPUT as ARCHIVE, reuse_metrics
from run_cfdr_frozen import PARENT as CF0
from run_cfdr_frozen import checkpoint_path, load_backbone
from run_e2_aspr_pilot import METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, MANIFEST, read_image, read_json, write_json
from run_tr0_factorial import write_history
from test_OS import evaluate
from train_direct_fusion import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


PARENT = checkpoint_path("CF1", 4)
ROOT = Path("models/e2_dcfd_v1_frozen/seed42")
OUTPUT = Path("test_result/e2_dcfd_v1_frozen")
PREFLIGHT = ROOT / "preflight.json"
GROUP = "QXS-DCFD-V1-FROZEN-SEED42"
LABEL = "DCFD-V1 双源双路径协同解码"
SEED, EPOCHS, ACCUMULATION, BATCH = 42, 4, 4, 2
PREFLIGHT_UPDATES = 32
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
IMPROVEMENT = {"SSIM": .003, "MS_SSIM": .003, "CC": .020, "SCD": .050, "PSNR": .30}
RETENTION = {"MI": -5., "FMI": -3., "Qabf": -.005, "VIF": -5.}
PERCENT = ("MI", "FMI", "VIF")
VISUAL_POSITIONS = tuple(range(0, 1000, 125))
SOURCES = (*EVALUATOR_FILES, "e2_dcfd.py", "dcfd_losses.py", "run_dcfd_frozen.py", "audit_dcfd_preflight.py",
           "e2_cfdr.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py", "utils/dataset_os.py",
           "train_lwpd.py", "run_e2_aspr_pilot.py")


def inputs():
    parent, _, source = load_parent()
    protocol = dict(
        variant=DCFDE2.VARIANT, parent_checkpoint=PARENT.as_posix(),
        parent_checkpoint_sha256=source["parent_checkpoint_sha256"], cf0_checkpoint_sha256=source["cf0_checkpoint_sha256"],
        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=source["split_manifest_sha256"],
        source_sha256={name: sha256_file(Path(name)) for name in SOURCES},
        seed=SEED, split_seed=20260924, train_count=8000, epochs=EPOCHS, patch_size=256,
        batch_size=BATCH, accumulation=ACCUMULATION, lr_inherited=2.5e-5, lr_new=1e-4,
        weight_decay=1e-4, clip_grad=1.0, invalid_threshold=0,
        loss_weights=dict(base=1., full=.5, detail=2., tether=.15),
    )
    return parent, protocol


def train_loader(epoch, device):
    splits, manifest = manifest_pairs(DATA, MANIFEST)
    if manifest["seed"] != 20260924 or len(splits["train"]) != 8000:
        raise ValueError("Expected the locked QXS train8000 split")
    sampler_rng = torch.Generator().manual_seed(SEED + epoch)
    dataset = OpticalSARDataset(splits["train"], patch_size=256, training=True, invalid_threshold=0)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, generator=sampler_rng,
                                       num_workers=0, pin_memory=device.type == "cuda")


def setup_optimizer(model, precision):
    inherited, introduced = model.parameter_groups()
    optimized = inherited + introduced
    if len(optimized) != len(set(optimized)) or set(optimized) != {p for p in model.parameters() if p.requires_grad}:
        raise ValueError("DCFD optimizer groups must partition the trainable parameters")
    optimizer = torch.optim.AdamW([dict(params=inherited, lr=2.5e-5), dict(params=introduced, lr=1e-4)],
                                  weight_decay=1e-4)
    return optimizer, torch.amp.GradScaler("cuda", enabled=precision == "fp16")


def loss_batch(model, teacher, losses, batch, device, precision):
    optical, sar, valid = [tensor.to(device, non_blocking=True) for tensor in batch[:3]]
    with torch.autocast(device.type, dtype=DTYPES[precision], enabled=precision != "fp32"):
        shared, encoded, fused_detail = model.prepare(optical, sar)
        reference = teacher(shared, fused_detail, optical, sar)
        terms, structure, detail = model.decoder(shared, encoded, fused_detail, optical, sar)
    total, stats = losses(terms, reference, optical, sar, valid)
    with torch.no_grad():
        gain_optical, gain_sar, gain_null = terms.gain.mean(dim=(0, 2, 3)).unbind()
        stats.update(loss_total=total.detach(), path_difference_mae=(structure.float() - detail.float()).abs().mean(),
                     gain_optical=gain_optical, gain_sar=gain_sar, gain_null=gain_null,
                     out_of_range_ratio=((terms.raw < 0) | (terms.raw > 1)).float().mean())
        finite = torch.stack([value.isfinite().all() for value in (*terms, *reference, total)]).all()
        if not finite.item():
            raise FloatingPointError("Non-finite DCFD output, teacher output or loss")
    return total, stats


def train_pass(model, teacher, loader, device, precision, optimizer, scaler, updates, observe=None):
    model.train()
    teacher.eval()
    losses = DCFDLoss().to(device)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    sums = {}
    effective = 0
    grad_sum = 0.
    attempts = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for micro, batch in enumerate(islice(loader, updates * ACCUMULATION), start=1):
        total, stats = loss_batch(model, teacher, losses, batch, device, precision)
        scaler.scale(total / ACCUMULATION).backward()
        for name, value in stats.items():
            sums[name] = sums.get(name, 0) + value.detach()
        if micro % ACCUMULATION:
            continue
        attempts += 1
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if math.isfinite(norm):
            if observe:
                observe(effective + 1, before=True)
            scaler.step(optimizer)
            effective += 1
            grad_sum += norm
            if observe:
                observe(effective, before=False)
        elif not scaler.is_enabled():
            raise FloatingPointError(f"Non-finite {precision} gradient at attempt {attempts}")
        elif all(torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None):
            raise FloatingPointError(f"Gradient norm overflow at attempt {attempts}")
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if attempts != updates or micro != updates * ACCUMULATION:
        raise RuntimeError(f"Expected {updates * ACCUMULATION} microbatches, completed {attempts} update attempts")
    torch.cuda.synchronize(device)
    means = {name: float(value) / micro for name, value in sums.items()}
    if not all(math.isfinite(value) for value in means.values()):
        raise FloatingPointError("Non-finite DCFD training telemetry")
    return dict(**means, attempted_updates=attempts, effective_updates=effective, skipped_updates=attempts - effective,
                grad_norm=grad_sum / max(effective, 1), peak_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20,
                epoch_time=time.perf_counter() - started)


def checkpoint(epoch):
    return ROOT / f"epoch_{epoch:03d}.pth"


def read_checkpoint(epoch, config_digest):
    saved = torch.load(checkpoint(epoch), map_location="cpu", weights_only=True)
    if (saved["variant"], saved["epoch"], saved["config_sha256"]) != (DCFDE2.VARIANT, epoch, config_digest):
        raise ValueError(f"Unexpected DCFD checkpoint provenance: {checkpoint(epoch)}")
    if [row["epoch"] for row in saved["history"]] != list(range(1, epoch + 1)):
        raise ValueError(f"Unexpected DCFD history: {checkpoint(epoch)}")
    return saved


def parameters_and_buffers(module):
    return dict((*module.named_parameters(), *module.named_buffers()))


def snapshot(module, prefixes=("",)):
    return {name: value.detach().cpu().clone() for name, value in parameters_and_buffers(module).items()
            if name.startswith(prefixes)}


def frozen_state(model):
    return snapshot(model, ("encoder.", "lwpd.", "base_fusion.", "detail_fusion.", "decoder.shared."))


def require_unchanged(baseline, module):
    state = parameters_and_buffers(module)
    if any(not torch.equal(value, state[name].detach().cpu()) for name, value in baseline.items()):
        raise ValueError("DCFD frozen parameters or buffers changed")


def atomic_write(path, write, *, replace=False):
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        write(temporary)
        temporary.replace(path) if replace else temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def train_lock():
    import msvcrt

    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".train.lock").open("a+b") as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def train(device):
    parent, protocol = inputs()
    report = read_json(PREFLIGHT)
    precision = report["selected_precision"]
    if (report["protocol"] != protocol or not report["passed"] or precision not in DTYPES
            or report["precision"][precision]["effective_updates"] / PREFLIGHT_UPDATES < .99
            or report["device"] != device_identity(device)):
        raise ValueError(f"Complete a matching DCFD P1 preflight before training: {PREFLIGHT}")
    config = dict(**protocol, precision=precision, preflight_sha256=sha256_file(PREFLIGHT))
    with train_lock():
        train_locked(device, parent, config)


def train_locked(device, parent, config):
    path = ROOT / "config.json"
    existing = sorted(ROOT.glob("epoch_*.pth"))
    if existing != [checkpoint(epoch) for epoch in range(1, len(existing) + 1)] or len(existing) > EPOCHS:
        raise ValueError("Unexpected DCFD checkpoint sequence")
    if not path.exists():
        if existing:
            raise FileNotFoundError(path)
        atomic_write(path, lambda temp: write_json(temp, config))
    if read_json(path) != config:
        raise ValueError(f"Preserve the existing DCFD run with different configuration: {path}")
    digest = sha256_file(path)
    trainer.set_seed(SEED)
    model, teacher = DCFDE2.initialized(parent)
    frozen = frozen_state(model)
    teacher_state = snapshot(teacher)
    model.to(device)
    teacher.to(device)
    optimizer, scaler = setup_optimizer(model, config["precision"])
    history = []
    if existing:
        saved = read_checkpoint(len(existing), digest)
        model.load_state_dict(saved["model"], strict=True)
        require_unchanged(frozen, model)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        del saved
        write_training_views(history)
    for epoch in range(len(existing) + 1, EPOCHS + 1):
        trainer.set_seed(SEED + epoch)
        loader = train_loader(epoch, device)
        stats = train_pass(model, teacher, loader, device, config["precision"], optimizer, scaler, 1000)
        if stats["effective_updates"] / stats["attempted_updates"] < .99:
            failure = ROOT / f"epoch_{epoch:03d}_failed_{uuid4().hex}.json"
            write_json(failure, dict(epoch=epoch, **stats))
            raise FloatingPointError(f"DCFD effective updates below 99%: {failure}")
        require_unchanged(frozen, model)
        require_unchanged(teacher_state, teacher)
        history.append(dict(epoch=epoch, **stats))
        payload = dict(variant=DCFDE2.VARIANT, args={"variant": DCFDE2.VARIANT}, epoch=epoch,
                       config_sha256=digest, history=history, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        atomic_write(checkpoint(epoch), lambda temp: torch.save(payload, temp))
        write_training_views(history)
        print(f"DCFD-V1 epoch {epoch}/{EPOCHS}, loss={stats['loss_total']:.6f}, updates={stats['effective_updates']}/1000",
              flush=True)
    final = checkpoint(EPOCHS)
    entry_id = f"{GROUP}-TRAIN8000-EPOCH04"
    register_many([dict(
        id=entry_id, comparable_group=entry_id, method=LABEL, task="training", status="training_complete",
        dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000, checkpoint=final.as_posix(),
        checkpoint_sha256=sha256_file(final), parent_checkpoint=PARENT.as_posix(),
        parent_checkpoint_sha256=config["parent_checkpoint_sha256"], split_manifest=MANIFEST.as_posix(),
        split_manifest_sha256=config["split_manifest_sha256"], result_dir=ROOT.as_posix(), metadata_path=path.as_posix(),
        metric_profile="training_loss", output_policy="float_loss", metrics={"loss_total": history[-1]["loss_total"]},
        metric_source=dict(type="csv_row", path=(ROOT / "history.csv").as_posix(), match={"epoch": str(EPOCHS)},
                           columns={"loss_total": "loss_total"}),
    )])


def write_training_views(history):
    write_history(ROOT / "history.csv", history)
    latest = checkpoint(history[-1]["epoch"])
    atomic_write(ROOT / "last.pth", lambda path: shutil.copyfile(latest, path), replace=True)


def device_identity(device):
    properties = torch.cuda.get_device_properties(device)
    return dict(name=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                total_memory=properties.total_memory, torch=str(torch.__version__), cuda=torch.version.cuda)


def screen(values):
    improved = [name for name, floor in IMPROVEMENT.items()
                if values.get(name) is not None and values[name] >= floor]
    retained = [name for name, floor in RETENTION.items()
                if values.get(name) is not None and values[name] >= floor]
    nabf = values.get("Nabf")
    return dict(passed=len(improved) >= 3 and bool({"SSIM", "MS_SSIM"}.intersection(improved))
                and len(retained) == len(RETENTION), improved=improved, retained=retained,
                nabf_maintained=nabf <= 0 if nabf is not None else None, deltas=values)


def source_pair(pair, device):
    optical = read_image(pair[0], cv2.IMREAD_COLOR)
    sar = read_image(pair[1], cv2.IMREAD_GRAYSCALE)
    if optical.shape[:2] != sar.shape:
        raise ValueError(f"Source shape mismatch: {pair[0].stem}")
    ycrcb = cv2.cvtColor(optical, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0]
    tensors = [torch.from_numpy(array.astype(np.float32) / 255.)[None, None].to(device)
               for array in (luminance, sar)]
    return optical, sar, ycrcb, tensors


def quantize(fused):
    if not fused.isfinite().all().item():
        raise FloatingPointError("Non-finite DCFD evaluation output")
    return np.rint(fused[0, 0].clamp(0, 1).cpu().numpy() * 255.).astype(np.uint8)


def evaluation_model(arm, saved, device):
    if arm == "CF0":
        state, _ = load_backbone()
        model = OpticalBaseAnchoredE2()
    elif arm == "CF1":
        state = torch.load(PARENT, map_location="cpu", weights_only=True)["model"]
        model = CFDRE2("CF1")
    else:
        state = saved["model"]
        model = DCFDE2()
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def infer(model, pairs, device):
    rows = []
    for i, pair in enumerate(pairs, start=1):
        _, sar, ycrcb, tensors = source_pair(pair, device)
        fused = quantize(model(*tensors))
        rows.append((evaluate(fused, ycrcb[:, :, 0], sar, DEFAULT_PROFILE), (), ()))
        if i == 1 or i % 50 == 0:
            print(f"Evaluated {i}/{len(pairs)} scenes", flush=True)
    return rows


@torch.inference_mode()
def benchmark(model, pairs, folder, device):
    for kind in ("gray", "rgb", "preview"):
        (folder / kind).mkdir()
    _, _, _, warmup = source_pair(pairs[0], device)
    for _ in range(3):
        model(*warmup)
    torch.cuda.synchronize(device)
    del warmup
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for pair in pairs:
        optical, sar, ycrcb, tensors = source_pair(pair, device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fused = model(*tensors)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.)
        gray = quantize(fused)
        ycrcb[:, :, 0] = gray
        color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        preview = np.concatenate((optical, cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR), color), axis=1)
        for kind, image in (("gray", gray), ("rgb", color), ("preview", preview)):
            if not cv2.imwrite(str(folder / kind / f"{pair[0].stem}.png"), image):
                raise OSError(f"Could not save {kind} image for {pair[0].stem}")
        del tensors, fused
    return dict(parameters=sum(p.numel() for p in model.parameters()), peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                mean_inference_ms=float(np.mean(times)), scene_count=len(times), precision="fp32", warmup=3)


def coverage_and_screening(work, ids, rows, contrasts):
    ms_col = METRICS.index("MS_SSIM")
    ms = {arm: np.asarray([row[0][ms_col] for row in items]) for arm, items in rows.items()}
    baseline_mask = np.isfinite(ms["CF1"])
    coverage = dict(baseline_valid_count=int(baseline_mask.sum()),
                    baseline_valid_scene_ids=[scene for scene, ok in zip(ids, baseline_mask) if ok], arms={})
    for arm, values in ms.items():
        finite = np.isfinite(values)
        lost = baseline_mask & ~finite
        gained = ~baseline_mask & finite
        fixed = values[baseline_mask]
        difference = fixed - ms["CF1"][baseline_mask]
        complete = bool(fixed.size and not lost.any())
        interval = None
        if complete and difference.size > 1:
            rng = np.random.default_rng(42)
            resamples = rng.integers(difference.size, size=(2000, difference.size))
            interval = np.quantile(difference[resamples].mean(axis=1), [.025, .975]).tolist()
        coverage["arms"][arm] = dict(valid_count=int(finite.sum()), overlap_count=int((baseline_mask & finite).sum()),
                                      lost_scene_ids=[scene for scene, ok in zip(ids, lost) if ok],
                                      gained_scene_ids=[scene for scene, ok in zip(ids, gained) if ok],
                                      fixed_baseline_mean=float(fixed.mean()) if complete else None,
                                      fixed_baseline_delta=float(difference.mean()) if complete else None,
                                      fixed_baseline_ci95=interval)
    write_json(work / "coverage.json", coverage)
    with (work / "paired_comparison.csv").open(newline="", encoding="utf-8-sig") as handle:
        paired = {(row["control"], row["candidate"], row["metric"]): row for row in csv.DictReader(handle)}
    screens = {}
    for control, arm in contrasts:
        if control != "CF1":
            continue
        changes = {}
        for name in (*IMPROVEMENT, *RETENTION, "Nabf"):
            row = paired[control, arm, name]
            if name == "MS_SSIM":
                changes[name] = coverage["arms"][arm]["fixed_baseline_delta"]
            elif not row["candidate_minus_control"]:
                changes[name] = None
            else:
                delta = float(row["candidate_minus_control"])
                if name in PERCENT:
                    baseline = float(row["control_mean"])
                    delta = 100 * delta / baseline if baseline > 0 else None
                changes[name] = delta
        screens[arm] = screen(changes)
    write_json(work / "screening.json", dict(improvement=IMPROVEMENT, retention=RETENTION, percent_metrics=PERCENT,
                                             candidates=screens))
    return paired


def metric_table(path, labels, summaries, paired):
    lines = ["# DCFD-V1 val1000 融合指标", "", "| 指标 | 共同有效景数 | " + " | ".join(labels) + " |",
             "| --- | ---: | " + " | ".join(["---:"] * len(labels)) + " |"]
    for name in METRICS:
        count = paired["CF1", "DCFD", name]["common_count"]
        cells = [f"{summaries[arm][name]:.6f}" if name in summaries[arm] else "—" for arm in labels]
        lines.append(f"| {name} | {count} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validation(device, *, trend):
    pairs, manifest = manifest_pairs(DATA, MANIFEST)
    all_ids = manifest["splits"]["val"]
    if manifest["seed"] != 20260924 or len(all_ids) != 1000:
        raise ValueError("Expected the locked QXS val1000 split")
    offset, limit = (200, 200) if trend else (0, 1000)
    ids = all_ids[offset:offset + limit]
    selected = pairs["val"][offset:offset + limit]
    arms = {f"DCFD_E{epoch:02d}": epoch for epoch in (1, 2, 4)} if trend else {"DCFD": 4}
    paths = {"CF1": PARENT, **{arm: checkpoint(epoch) for arm, epoch in arms.items()}}
    if not trend:
        paths = {"CF0": CF0, **paths}
    labels = {arm: f"{LABEL} 第 {arms[arm]} 轮" if arm in arms else arm for arm in paths}
    contrasts = [("CF1", arm) for arm in arms] if trend else [("CF0", "CF1"), ("CF0", "DCFD"), ("CF1", "DCFD")]
    config_path = ROOT / "config.json"
    config_digest = sha256_file(config_path)
    config = read_json(config_path)
    manifest_digest = sha256_file(MANIFEST)
    hashes = {arm: sha256_file(path) for arm, path in paths.items()}
    if (config["split_manifest_sha256"] != manifest_digest or config["parent_checkpoint_sha256"] != hashes["CF1"]
            or config["variant"] != DCFDE2.VARIANT
            or any(sha256_file(Path(name)) != digest for name, digest in config["source_sha256"].items())):
        raise ValueError(f"DCFD training configuration differs from the active sources or input data: {config_path}")
    cached = {arm: digest for arm, digest in hashes.items() if arm not in arms}
    archive_meta = reuse_metrics(ARCHIVE, all_ids, manifest_digest, cached, 4)
    raw = {arm: ARCHIVE / arm / ("evaluator_metrics.csv" if arm == "CF1" else "metrics.csv") for arm in cached}
    protocol = dict(split="val", offset=offset, scene_ids=ids, device=device_identity(device),
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", inference_precision="fp32",
                    training_precision=config["precision"],
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_digest,
                    checkpoints={arm: path.as_posix() for arm, path in paths.items()}, checkpoint_hashes=hashes,
                    training_config=config_path.as_posix(), training_config_sha256=config_digest,
                    source_sha256=config["source_sha256"],
                    archive=dict(metadata=archive_meta.as_posix(), metadata_sha256=sha256_file(archive_meta),
                                 metrics={arm: dict(path=path.as_posix(), sha256=sha256_file(path)) for arm, path in raw.items()}),
                    visual_scene_ids=[] if trend else [ids[i] for i in VISUAL_POSITIONS])
    output = OUTPUT / ("val200_trend/offset200" if trend else "val1000/epoch_004")
    if output.exists():
        if read_json(output / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous DCFD validation: {output}")
    else:
        work = output.with_name(f".{output.name}-pending-{uuid4().hex}")
        work.mkdir(parents=True)
        rows = {arm: read_metric_rows(path, all_ids)[offset:offset + limit] for arm, path in raw.items()}
        efficiency = {}
        for arm in labels:
            (work / arm).mkdir()
            saved = read_checkpoint(arms[arm], config_digest) if arm in arms else None
            if trend and saved is None:
                continue
            model = evaluation_model(arm, saved, device)
            del saved
            if arm in arms:
                print(f"Evaluating {arm}", flush=True)
                rows[arm] = infer(model, selected, device)
            if not trend:
                efficiency[arm] = benchmark(model, [selected[i] for i in VISUAL_POSITIONS], work / arm, device)
            del model
        summaries = summarize(work, ids, rows, contrasts, diagnostic_fields=(), source_metrics=())
        paired = coverage_and_screening(work, ids, rows, contrasts)
        if not trend:
            write_json(work / "efficiency.json", efficiency)
            metric_table(work / "comparison_14_metrics.md", labels, summaries, paired)
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        work.rename(output)
    register_validation(output, labels, protocol, trend)
    print(f"DCFD evaluation: {output / 'screening.json'}", flush=True)


def register_validation(output, labels, protocol, trend):
    summary_path = output / "registry_metrics.json"
    summaries = read_json(summary_path)
    group = f"{GROUP}-{'VAL200-OFFSET200-TREND-E01-E02-E04' if trend else 'VAL1000-E04'}-REF-FMI"
    register_many([dict(
        id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split="val", offset=protocol["offset"],
        sample_count=len(protocol["scene_ids"]), task="fusion", method=label, status="development_validation",
        result_dir=(output / arm).as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
        checkpoint=protocol["checkpoints"][arm], checkpoint_sha256=protocol["checkpoint_hashes"][arm],
        split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
        metric_profile=protocol["metric_profile"], output_policy=protocol["output_policy"], metrics=summaries[arm],
        metric_source=dict(type="json_paths", path=summary_path.as_posix(), paths={key: [arm, key] for key in summaries[arm]}),
        notes="汇总使用组内共同有限样本，MS-SSIM 筛选使用 CF1 固定有效集合。",
    ) for arm, label in labels.items()])


def interactive_screen():
    targets = {name: ("≥", limit) for name, limit in (IMPROVEMENT | RETENTION).items()}
    targets["Nabf"] = ("≤", 0.)
    print("输入相对同批 CF1 的变化，输入 r 清空，q 退出。")
    while True:
        values = {}
        while len(values) < len(targets):
            name = tuple(targets)[len(values)]
            operator, limit = targets[name]
            unit = "%" if name in PERCENT else " dB" if name == "PSNR" else ""
            try:
                raw = input(f"{name}（{operator} {limit:+g}{unit}）> ").strip().lower()
            except EOFError:
                return
            if raw == "q":
                return
            if raw == "r":
                values.clear()
                continue
            try:
                value = float(raw)
                if math.isfinite(value):
                    values[name] = value
                    continue
            except ValueError:
                pass
            print("请输入有限数值。")
        result = screen(values)
        reached = set(result["improved"] + result["retained"])
        if result["nabf_maintained"]:
            reached.add("Nabf")
        print("，".join(f"{name} {'达标' if name in reached else '未达标'}" for name in targets))
        retained = len(result["retained"]) + int(result["nabf_maintained"])
        print(f"弱项达标 {len(result['improved'])}/5，优势保持 {retained}/5（含 NAB/F 优先项）。")
        print("数值筛选通过" if result["passed"] else "数值筛选未通过")
        try:
            if input("输入 r 清空重填，回车退出 > ").strip().lower() != "r":
                return
        except EOFError:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    training, trend, formal = (stages.add_parser(name) for name in ("train", "trend", "validate"))
    for stage in (training, trend, formal):
        stage.add_argument("--device", choices=("cuda",), default="cuda")
    training.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    training.add_argument("--epochs", type=int, choices=(EPOCHS,), default=EPOCHS)
    trend.add_argument("--epochs", type=int, nargs=3, default=[1, 2, 4])
    trend.add_argument("--offset", type=int, choices=(200,), default=200)
    trend.add_argument("--limit", type=int, choices=(200,), default=200)
    formal.add_argument("--epoch", type=int, choices=(4,), default=4)
    formal.add_argument("--limit", type=int, choices=(1000,), default=1000)
    stages.add_parser("screen", help="Interactively screen ten changes relative to CF1")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if args.stage == "screen":
        interactive_screen()
        return
    if args.stage == "trend" and args.epochs != [1, 2, 4]:
        parser.error("trend requires --epochs 1 2 4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DCFD")
    device = torch.device(args.device)
    if args.stage == "train":
        train(device)
    else:
        validation(device, trend=args.stage == "trend")


if __name__ == "__main__":
    main()
