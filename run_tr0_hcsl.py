"""Continue C01 Stage I into the C03-H appearance-anchored proxy experiment."""

import argparse
import os
from pathlib import Path

import cv2
import torch

from run_tr0_c02 import OUTPUT, ROOT
from run_tr0_factorial import (
    build_loaders, build_model, checkpoint_path, configure_stage, forward_losses,
    lock_manifests, make_optimizer, precision_context, read_json, run_epoch, set_seed,
    sha256_file, write_history, write_json,
)
from train_direct_fusion import DirectFusionLosses
from tr0_complementary import ComplementaryReconstructionLoss
from tr0_evaluation import HCSL_ARM, HCSL_ARMS, evaluate_arms, register_trainings
from tr0_hcsl import AppearanceAnchoredViewGenerator
from utils.dataset_os import OpticalSARDataset, manifest_pairs


CONTRAST_MIN = 0.7


def load_parent(config, root, manifest):
    epoch = config["reconstruction_epochs"]
    path = root / "C01" / f"epoch_{epoch:03d}.pth"
    path_config = path.parent / "config.json"
    recorded = read_json(path_config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    manifest_hash = sha256_file(manifest)
    if (recorded["config"] != config or recorded["arm"] != "C01" or recorded["device"] != "cuda"
            or recorded["manifest_sha256"] != manifest_hash or state["manifest_sha256"] != manifest_hash
            or state["config_sha256"] != sha256_file(path_config)
            or (state["arm"], state["epoch"], state["stage"]) != ("C01", epoch, "I")):
        raise ValueError(f"Unexpected C01 Stage I training state: {path}")
    sources = recorded["source_sha256"]
    for name, digest in sources.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Training source differs from C01: {name}")
    parent = dict(checkpoint=path.as_posix(), checkpoint_sha256=sha256_file(path),
                  epoch=epoch, arm="C01", optimizer_restored=True)
    return state, parent, sources


def train_hcsl(config, root, manifest, device):
    state, parent, sources = load_parent(config, root, manifest)
    stages = ["I"] * config["reconstruction_epochs"] + ["II"] * config["proxy_epochs"] + ["III"] * config["fusion_epochs"]
    folder = root / HCSL_ARM
    metadata = folder / "config.json"
    run = dict(config=config, arm=HCSL_ARM, device=str(device), parent=parent, stages=stages,
               proxy=dict(mix_fraction=0.5, contrast_min=CONTRAST_MIN),
               manifest_sha256=sha256_file(manifest), source_sha256={**sources, **{
                   name: sha256_file(Path(name)) for name in ("run_tr0_hcsl.py", "tr0_hcsl.py")
               }})
    folder.mkdir(parents=True, exist_ok=True)
    if metadata.is_file():
        if read_json(metadata) != run:
            raise ValueError(f"Existing C03-H training settings differ: {folder}")
    else:
        write_json(metadata, run)
    config_hash = sha256_file(metadata)
    checkpoints = sorted(folder.glob("epoch_*.pth"))
    if checkpoints:
        inherited = state["history"]
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if (state["arm"] != HCSL_ARM or state["parent"] != parent
                or state["manifest_sha256"] != run["manifest_sha256"] or state["config_sha256"] != config_hash
                or state["history"][:parent["epoch"]] != inherited
                or state["epoch"] != len(state["history"]) or state["stage"] != state["history"][-1]["stage"]
                or checkpoints[-1].name != f"epoch_{state['epoch']:03d}.pth"):
            raise ValueError(f"Existing C03-H checkpoint provenance differs: {checkpoints[-1]}")
    history = state["history"]
    steps = history[0]["train_optimizer_steps"]
    if len(history) > len(stages) or any(
        (row["epoch"], row["stage"], row["train_optimizer_steps"]) != (epoch, stage, steps)
        for epoch, (row, stage) in enumerate(zip(history, stages), start=1)
    ):
        raise ValueError("C03-H history differs from the fixed stage or optimizer-step budget")
    if len(history) < len(stages):
        set_seed(config["seed"])
        data_args = argparse.Namespace(
            **{key: config[key] for key in ("batch_size", "patch_size", "min_valid_fraction", "invalid_threshold", "seed")},
            data_root=Path(config["data_root"]), split_manifest=manifest, optical_dir="GF2", sar_dir="GF3",
            train_limit=0, val_limit=0, num_workers=0,
        )
        train_loader, val_loader, _, _ = build_loaders(data_args, device, folder)
        if (len(train_loader.dataset), len(val_loader.dataset)) != (config["train_count"], config["screen_count"]):
            raise ValueError("Resolved dataset sizes differ from the TR0 protocol")
        model = build_model("e2").to(device)
        optimizer = make_optimizer(model, config)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        del state
        losses = DirectFusionLosses().to(device)
        proxy = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
        views = AppearanceAnchoredViewGenerator(**config["view_generator"], contrast_min=CONTRAST_MIN)
        torch.backends.cudnn.deterministic = True
        for epoch in range(len(history) + 1, len(stages) + 1):
            stage = stages[epoch - 1]
            configure_stage(model, optimizer, stage, config)
            row = dict(epoch=epoch, stage=stage)
            for label, loader, update in (("train", train_loader, optimizer), ("val", val_loader, None)):
                stats = run_epoch(model, losses, proxy, views, loader, config, stage, "complementary", epoch, device, update)
                row.update({f"{label}_{name}": value for name, value in stats.items()})
            if row["train_optimizer_steps"] != steps:
                raise ValueError("C03-H update count differs from the inherited schedule")
            if stage == "II" and (row["train_hcsl_fraction"], row["val_hcsl_fraction"]) != (0.5, 0.5):
                raise ValueError("C03-H requires a 50/50 proxy mixture on the locked data")
            history.append(row)
            path = folder / f"epoch_{epoch:03d}.pth"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(epoch=epoch, stage=stage, arm=HCSL_ARM, variant=model.variant,
                            args={"variant": model.variant}, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            history=history, parent=parent, manifest_sha256=run["manifest_sha256"],
                            config_sha256=config_hash), temporary)
            temporary.replace(path)
            write_history(folder / "history.csv", history)
            print(f"{HCSL_ARM} epoch {epoch}/{len(stages)} stage {stage}: "
                  f"train={row['train_loss']:.6f} val={row['val_loss']:.6f}", flush=True)
    write_history(folder / "history.csv", history)
    return checkpoint_path(config, root, HCSL_ARM)


def smoke(config, manifest, device):
    state, _, _ = load_parent(config, ROOT, manifest)
    splits, _ = manifest_pairs(Path(config["data_root"]), manifest)
    optical, sar, valid, scene = OpticalSARDataset(splits["val"][:1], patch_size=config["patch_size"], augment=False)[0]
    optical, sar, valid = (tensor.unsqueeze(0).to(device) for tensor in (optical, sar, valid))
    # Duplicate one patch solely to exercise both routes in the mixed batch.
    source = optical.expand(2, -1, -1, -1)
    seed = config["proxy_validation_seed"]
    views = AppearanceAnchoredViewGenerator(**config["view_generator"], contrast_min=CONTRAST_MIN)
    first, second, stats, selected = views.mixed_views(source, torch.Generator().manual_seed(seed))
    index = int(selected[0])
    v1_first, v1_second, _, _ = views.v1(source, torch.Generator().manual_seed(seed))
    torch.testing.assert_close(first[1 - index], v1_first[1 - index], rtol=0, atol=0)
    torch.testing.assert_close(second[1 - index], v1_second[1 - index], rtol=0, atol=0)
    if not all(torch.isfinite(x).all() and x.min() >= -1e-6 and x.max() <= 1 + 1e-6 for x in (first, second)):
        raise AssertionError("Generated views must stay in [0, 1]")
    regions = [float(stats[f"hcsl_{key}"]) for key in ("anchor_only", "supplement_only", "both_clear", "both_degraded")]
    if not all(abs(value - target) < 0.1 for value, target in zip(regions, (0.3, 0.3, 0.3, 0.1))):
        raise AssertionError(f"Unexpected proxy coverage: {regions}")
    model = build_model("e2").to(device)
    optimizer = make_optimizer(model, config)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if not optimizer.state:
        raise AssertionError("C01 optimizer state is empty")
    configure_stage(model, optimizer, "II", config)
    losses = DirectFusionLosses().to(device)
    proxy = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
    pair = (first[index:index + 1], second[index:index + 1])
    with precision_context(config, device):
        loss, _ = forward_losses(model, losses, proxy, config, "II", optical, sar, valid,
                                 proxy_pair=pair, checkpointing=config["activation_checkpointing"])
    loss.backward()
    if not torch.isfinite(loss) or not any(
        p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
        for p in model.detail_fusion.parameters()
    ):
        raise AssertionError("The Stage II proxy loss must backpropagate into Detail Fusion")
    panels = (("Target", optical[0]), ("V1 optical", first[1 - index]), ("V1 supplement", second[1 - index]),
              ("V2 anchor", first[index]), ("V2 transformed", second[index]))
    images = []
    for title, tensor in panels:
        gray = tensor.squeeze().clamp(0, 1).mul(255).round().byte().cpu().numpy()
        image = cv2.copyMakeBorder(gray, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=20)
        cv2.putText(image, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 240, 1, cv2.LINE_AA)
        images.append(image)
    path = Path("tmp/tr0_hcsl_smoke") / f"{scene}_proxy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.hconcat(images)):
        raise OSError(f"Cannot write proxy preview: {path}")
    print(dict(scene=scene, loss=float(loss), preview=str(path), **{k: float(v) for k, v in stats.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "screen", "confirm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    config = read_json("configs/tr0_v1.json")
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("C03-H requires the CUDA BF16 setup used for C01")
    manifests = lock_manifests(config, ROOT)
    if args.action == "smoke":
        smoke(config, manifests["screen"], device)
    elif args.action == "train":
        final = train_hcsl(config, ROOT, manifests["screen"], device)
        register_trainings(config, "seed42", {HCSL_ARM: final}, manifests["screen"])
    else:
        evaluate_arms(config, ROOT, OUTPUT, HCSL_ARMS, manifests[args.action], args.action, device, "seed42")


if __name__ == "__main__":
    main()
