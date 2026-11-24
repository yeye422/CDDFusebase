"""Run the TR0 factorial experiment, its budget control and the Early decoder control."""

import argparse
from contextlib import nullcontext
import csv
from functools import partial
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.checkpoint import checkpoint

from e2_tr0 import E2ComplementaryLearning, E2ProgressiveJointDecoding
from run_e2_pac_mamba import write_json
from train_direct_fusion import DirectFusionLosses, sha256_file
from train_lwpd import build_loaders, encoder_groups, set_seed
from tr0_complementary import ComplementaryReconstructionLoss, ComplementaryViewGenerator
from utils.dataset_os import digest_splits, evenly_spaced, validate_manifest_splits


MAIN_GROUPS = ("C00", "C01", "C10", "C11")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model(architecture):
    if architecture == "e2":
        return E2ComplementaryLearning()
    if architecture in ("early", "progressive"):
        return E2ProgressiveJointDecoding(early=architecture == "early")
    raise ValueError(f"Unknown architecture: {architecture}")


def lock_manifests(config, root):
    source_path, pilot_path = (Path(config[key]) for key in ("source_manifest", "pilot_manifest"))
    source, pilot = read_json(source_path), read_json(pilot_path)
    validate_manifest_splits(source)
    validate_manifest_splits(pilot)
    train, screen = (pilot["splits"][key] for key in ("train", "val"))
    source_val = source["splits"]["val"]
    if (pilot["source_manifest_sha256"] != sha256_file(source_path)
            or pilot["seed"] != config["seed"]
            or any(pilot[key] != source[key] for key in ("optical_dir", "sar_dir"))
            or len(train) != config["train_count"] or len(screen) != config["screen_count"]
            or not set(train).issubset(source["splits"]["train"])
            or not set(screen).issubset(source_val) or pilot["splits"]["test"]):
        raise ValueError("The historical train1024/val64 manifest does not match the TR0 protocol")
    screen_ids = set(screen)
    remaining = [scene for scene in source_val if scene not in screen_ids]
    confirm = evenly_spaced(remaining, config["confirm_count"])
    if len(confirm) != config["confirm_count"]:
        raise ValueError("The source validation split is too small for confirmation")
    confirm_splits = dict(train=train, val=confirm, test=[])
    confirmation = dict(pilot, splits=confirm_splits, splits_sha256=digest_splits(confirm_splits))
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, manifest in (("screen", pilot), ("confirm", confirmation)):
        path = root / f"{name}_manifest.json"
        if path.exists():
            if (read_json(path) != manifest
                    or name == "screen" and sha256_file(path) != sha256_file(pilot_path)):
                raise ValueError(f"Existing manifest differs: {path}")
        elif name == "screen":
            path.write_bytes(pilot_path.read_bytes())
        else:
            write_json(path, manifest)
        manifests[name] = path
    return manifests


def make_optimizer(model, config):
    early, tail = encoder_groups(model)
    groups = dict(encoder_early=early, encoder_tail=tail, lwpd=model.lwpd.parameters(),
                  base_fusion=model.base_fusion.parameters())
    if isinstance(model, E2ProgressiveJointDecoding):
        decoder = model.decoder
        groups["synthesis"] = (
            parameter for module in (decoder.base_projection, decoder.optical_projection,
                                     decoder.sar_projection, decoder.synthesis_stages)
            for parameter in module.parameters()
        )
        groups["decoder"] = (
            parameter for module in (decoder.decoder_stages, decoder.output)
            for parameter in module.parameters()
        )
    else:
        groups.update(decoder=model.decoder.parameters(), synthesis=model.detail_fusion.parameters())
    return torch.optim.AdamW(
        [{"params": list(params), "name": name} for name, params in groups.items()],
        weight_decay=config["weight_decay"],
    )


def configure_stage(model, optimizer, stage, config):
    for group in optimizer.param_groups:
        name = group["name"]
        active = stage != "I" or (name != "base_fusion" and (
            name != "synthesis" or isinstance(model, E2ProgressiveJointDecoding)))
        if stage == "III":
            lr = config["fusion_synthesis_lr" if name in ("base_fusion", "synthesis") else "fusion_backbone_lr"]
        else:
            lr = config["reconstruction_lr" if stage == "I" else "proxy_lr"]
        group["lr"] = lr if active else 0.0
        for parameter in group["params"]:
            parameter.requires_grad_(active)


def forward_losses(model, losses, proxy_loss, config, stage, optical, sar, valid,
                   *, proxy_pair=None, checkpointing=False):
    call = partial(checkpoint, use_reentrant=False, preserve_rng_state=False) if checkpointing else (
        lambda function, *inputs: function(*inputs)
    )
    encoded = call(model.encode_pair, optical, sar)
    optical_hat, sar_hat = call(model.reconstruct_sources, optical, sar, encoded)
    optical_rec = losses.reconstruction(optical, optical_hat)
    sar_rec = losses.reconstruction(sar, sar_hat, valid)
    reconstruction = 0.5 * (optical_rec + sar_rec)
    terms = dict(optical_reconstruction=optical_rec, sar_reconstruction=sar_rec,
                 reconstruction=reconstruction)
    if stage == "I":
        return reconstruction, terms
    if stage == "II":
        surrogate = call(model.encode_pair, *proxy_pair)
        prediction = call(model.fuse_and_decode, *surrogate)
        task_loss, pixels, edges = proxy_loss(prediction, optical)
        terms.update(proxy=task_loss, proxy_pixels=pixels, proxy_gradient=edges)
    else:
        prediction = call(model.fuse_and_decode, *encoded)
        task_loss, intensity, gradient = losses.fusion(
            optical, sar, prediction, valid, config["gradient_weight"])
        terms.update(fusion=task_loss, intensity=intensity, gradient=gradient)
    return task_loss + config["reconstruction_weight"] * reconstruction, terms


def gradient_norms(optimizer):
    norms = {}
    for group in optimizer.param_groups:
        parameters = [p.grad.detach() for p in group["params"] if p.grad is not None]
        if parameters:
            norms[f"grad_{group['name']}"] = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(gradient) for gradient in parameters
            ]))
    return norms


def precision_context(config, device):
    if config["precision"] == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if config["precision"] == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported precision: {config['precision']}")


def run_epoch(model, losses, proxy_loss, views, loader, config, stage, proxy, epoch,
              device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if training:
        set_seed(config["seed"] + epoch)
        loader.generator.manual_seed(config["seed"] + epoch)
        optimizer.zero_grad(set_to_none=True)
    proxy_seed = config["proxy_seed"] + epoch if training else config["proxy_validation_seed"]
    generator = torch.Generator().manual_seed(proxy_seed)
    accumulation = config["accumulation_steps"]
    totals, grad_totals = {}, {}
    count = steps = 0
    for index, (optical, sar, valid, _) in enumerate(loader):
        optical, sar, valid = (tensor.to(device, non_blocking=True) for tensor in (optical, sar, valid))
        view_stats, pair = {}, None
        if stage == "II":
            first, second, repeated, view_stats = views(optical, generator)
            pair = (first, second) if proxy == "complementary" else (repeated, repeated)
        with torch.set_grad_enabled(training), precision_context(config, device):
            total, terms = forward_losses(
                model, losses, proxy_loss, config, stage, optical, sar, valid, proxy_pair=pair,
                checkpointing=training and config["activation_checkpointing"],
            )
        if training:
            window_size = min(accumulation, len(loader) - index // accumulation * accumulation)
            (total / window_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                for key, value in gradient_norms(optimizer).items():
                    grad_totals[key] = grad_totals.get(key, 0.0) + float(value)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        for key, value in dict(terms, **view_stats, loss=total).items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(optical)
        count += len(optical)
    averages = {key: value / count for key, value in totals.items()}
    if not all(math.isfinite(value) for value in averages.values()):
        raise FloatingPointError(f"Non-finite loss or diagnostics at epoch {epoch}, stage {stage}")
    averages.update({key: value / steps for key, value in grad_totals.items()})
    averages.update(seconds=time.perf_counter() - started, samples=count, optimizer_steps=steps)
    if device.type == "cuda":
        averages["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return averages


def write_history(path, history):
    fields = list(dict.fromkeys(key for row in history for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def checkpoint_path(config, root, arm, *, proxy_end=False):
    stage_two = config["reconstruction_epochs"] + config["proxy_epochs"]
    epoch = stage_two if proxy_end else stage_two + config["fusion_epochs"]
    return root / arm / f"epoch_{epoch:03d}.pth"


def train(config, root, arm, manifest, device, resume):
    plan = config["experiments"][arm]
    folder = root / arm
    saved_config = dict(config=config, arm=arm, device=str(device),
                        manifest_sha256=sha256_file(manifest),
                        source_sha256={name: sha256_file(Path(name)) for name in (
                            "run_tr0_factorial.py", "e2_tr0.py", "tr0_complementary.py", "tr0_joint_decoder.py",
                            "train_lwpd.py", "train_direct_fusion.py", "direct_fusion.py", "utils/dataset_os.py",
                            "e2_base_anchor.py", "lwpd.py", "net.py",
                        )})
    final = checkpoint_path(config, root, arm)
    config_path = folder / "config.json"
    history_path = folder / "history.csv"
    if folder.exists() and read_json(config_path) != saved_config:
        raise ValueError(f"Existing training settings differ: {folder}")
    if final.is_file():
        finished = torch.load(final, map_location="cpu", weights_only=True)
        write_history(history_path, finished["history"])
        return final
    if folder.exists() and not resume:
        raise FileExistsError(f"Use --resume for existing partial training: {folder}")
    needs_setup = not folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    if needs_setup:
        write_json(config_path, saved_config)
    set_seed(config["seed"])
    args = argparse.Namespace(
        **{key: config[key] for key in ("batch_size", "patch_size", "min_valid_fraction", "invalid_threshold", "seed")},
        data_root=Path(config["data_root"]), split_manifest=manifest, optical_dir="GF2", sar_dir="GF3",
        train_limit=0, val_limit=0, num_workers=0,
    )
    training, validation, _, _ = build_loaders(args, device, folder)
    if (len(training.dataset), len(validation.dataset)) != (config["train_count"], config["screen_count"]):
        raise ValueError("Resolved dataset sizes differ from the TR0 protocol")
    model = build_model(plan["architecture"]).to(device)
    losses = DirectFusionLosses().to(device)
    proxy_loss = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
    views = ComplementaryViewGenerator(**config["view_generator"])
    optimizer = make_optimizer(model, config)
    stages = (["I"] * config["reconstruction_epochs"]
              + ["II" if plan["proxy"] else "I"] * config["proxy_epochs"]
              + ["III"] * config["fusion_epochs"])
    history = []
    checkpoints = sorted(folder.glob("epoch_*.pth"))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        history = state["history"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        del state
    torch.backends.cudnn.deterministic = True
    for epoch in range(len(history) + 1, len(stages) + 1):
        stage = stages[epoch - 1]
        configure_stage(model, optimizer, stage, config)
        row = dict(epoch=epoch, stage=stage)
        for label, loader, update in (("train", training, optimizer), ("val", validation, None)):
            stats = run_epoch(model, losses, proxy_loss, views, loader, config, stage, plan["proxy"],
                              epoch, device, update)
            row.update({f"{label}_{key}": value for key, value in stats.items()})
        history.append(row)
        path = folder / f"epoch_{epoch:03d}.pth"
        temporary = path.with_suffix(".tmp")
        torch.save(dict(epoch=epoch, arm=arm, stage=stage, variant=model.variant, args={"variant": model.variant},
                        model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                        manifest_sha256=saved_config["manifest_sha256"],
                        config_sha256=sha256_file(config_path)), temporary)
        temporary.replace(path)
        write_history(history_path, history)
        print(f"{arm} epoch {epoch}/{len(stages)} stage {stage}: "
              f"train={row['train_loss']:.6f} val={row['val_loss']:.6f}", flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "train", "screen", "confirm", "proxy-check", "diagnose"))
    parser.add_argument("--config", type=Path, default=Path("configs/tr0_v1.json"))
    parser.add_argument("--run-id", default="seed42")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must be a single alphanumeric name with optional '-' or '_'")
    config = read_json(args.config)
    default_arms = ("B0", *MAIN_GROUPS) if args.action in ("screen", "confirm") else MAIN_GROUPS
    arms = tuple(args.arms or default_arms)
    if len(set(arms)) != len(arms) or any(arm not in config["experiments"] and (
            arm != "B0" or args.action not in ("screen", "confirm")) for arm in arms):
        parser.error("Invalid or duplicate experiment arms for this action")
    root = Path("models/tr0_v1") / args.run_id
    output = Path("test_result/tr0_v1") / args.run_id
    if args.action == "lock":
        for label, path in lock_manifests(config, root).items():
            print(f"{label}: {path} SHA-256 {sha256_file(path)}")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.action in ("train", "diagnose") and device.type == "cuda" and (
            config["precision"] == "bf16" and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("The configured BF16 precision requires a BF16-capable GPU")
    from tr0_evaluation import diagnose, evaluate_arms, proxy_check, register_trainings

    if args.action == "diagnose":
        diagnose(config, device)
        return
    manifests = lock_manifests(config, root)
    if args.action == "train":
        completed = {}
        for arm in arms:
            completed[arm] = train(config, root, arm, manifests["screen"], device, args.resume)
        register_trainings(config, args.run_id, completed, manifests["screen"])
    elif args.action == "proxy-check":
        proxy_check(config, root, output, arms, manifests["screen"], device, args.run_id)
    else:
        evaluate_arms(config, root, output, arms, manifests[args.action], args.action, device, args.run_id)


if __name__ == "__main__":
    main()
