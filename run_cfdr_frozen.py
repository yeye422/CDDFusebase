"""Screen CFDR by training only its new reconstruction module."""

import argparse
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cfdr import ARMS, CFDRE2, VARIANTS
from run_cfdr import BASELINE, load_parent
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import STAGE_LEARNING_RATES, config_values
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file
import train_lwpd as trainer
from utils.dataset_os import OpticalSARDataset, manifest_pairs


PARENT = BASELINE / "epoch_016.pth"
ROOT = Path("models/e2_cfdr_v21_frozen/seed42")
EPOCHS = 2
EXTENDED_EPOCHS = 4
INITIAL_RUNNER_SHA256 = "fd53b31bef20a592f0228508f725b6459a552b726454cb4a9b91e53dde53124e"


class FrozenCFDRE2(CFDRE2):
    def __init__(self, arm, backbone):
        super().__init__(arm)
        missing, unexpected = self.load_state_dict(backbone, strict=False)
        if set(missing) != {name for name in self.state_dict() if name.startswith("cfdr.")} or unexpected:
            raise ValueError("The converged CF0 backbone does not match CFDR")
        self.requires_grad_(False)
        self.cfdr.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.cfdr.train(mode)
        return self


def load_backbone():
    _, args = load_parent()
    state = torch.load(PARENT, map_location="cpu", weights_only=False)
    if (state["epoch"], state["stage"], state["variant"]) != (15, "fusion", OpticalBaseAnchoredE2.VARIANT):
        raise ValueError("Frozen screening requires the completed CF0 fusion checkpoint")
    if (Path(state["args"]["split_manifest"]).resolve() != MANIFEST.resolve()
            or Path(args.data_root).resolve() != DATA.resolve()):
        raise ValueError("CF0 uses a different source dataset or split")
    return state["model"], args


def frozen_losses(model, losses, optical, sar, valid, args):
    with torch.no_grad():
        encoded = model.encode_pair(optical, sar)
    # Frozen parameters make the backbone inference-only. The original source
    # reconstruction loss is constant with respect to the CFDR head.
    fused, stats = model.fuse_and_reconstruct(optical, sar, encoded, return_stats=True)
    loss, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    return loss, dict(fusion=loss, intensity=intensity, gradient=gradient, **stats)


def checkpoint_path(arm, epochs=EPOCHS):
    root = ROOT if epochs == EPOCHS else ROOT / f"to_epoch_{epochs:03d}"
    return root / arm / f"epoch_{epochs:03d}.pth"


def train_arm(arm, device, epochs=EPOCHS):
    backbone, args = load_backbone()
    final = checkpoint_path(arm, epochs)
    folder = final.parent
    metadata = folder / "config.json"
    protocol = dict(arm=arm, parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=sha256_file(PARENT),
                    split_manifest_sha256=sha256_file(MANIFEST), epochs=EPOCHS, args=config_values(args),
                    training_policy="cfdr_only_from_converged_cf0", lr=STAGE_LEARNING_RATES["fusion_layers"],
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "run_cfdr_frozen.py", "run_cfdr.py", "e2_cfdr.py", "train_lwpd.py", "train_direct_fusion.py",
                        "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py", "utils/dataset_os.py",
                    )})
    state = None
    inherited = []
    first_saved_epoch = 1
    if epochs == EXTENDED_EPOCHS:
        parent = checkpoint_path(arm)
        parent_config = parent.parent / "config.json"
        expected = dict(protocol, source_sha256={**protocol["source_sha256"],
                                                "run_cfdr_frozen.py": INITIAL_RUNNER_SHA256})
        if read_json(parent_config) not in (protocol, expected):
            raise ValueError(f"The first two epochs use a different frozen recipe: {parent_config}")
        state = read_checkpoint(parent, arm, sha256_file(parent_config), EPOCHS)
        inherited = state["history"]
        if any(not torch.equal(value, state["model"][key]) for key, value in backbone.items()):
            raise ValueError(f"The first two epochs changed the CF0 backbone: {parent}")
        first_saved_epoch = EPOCHS + 1
        protocol.update(epochs=epochs, continuation_checkpoint=parent.as_posix(),
                        continuation_checkpoint_sha256=sha256_file(parent))
    folder.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        recorded = read_json(metadata)
        if epochs == EPOCHS and final.is_file() and recorded == dict(
            protocol, source_sha256={**protocol["source_sha256"], "run_cfdr_frozen.py": INITIAL_RUNNER_SHA256}
        ):
            protocol = recorded
        if recorded != protocol:
            raise ValueError(f"Existing frozen CFDR protocol differs: {folder}")
    if not metadata.exists():
        write_json(metadata, protocol)
    config_hash = sha256_file(metadata)
    history = list(inherited)
    checkpoints = sorted(folder.glob("epoch_???.pth"))
    if checkpoints:
        current_epoch = first_saved_epoch + len(checkpoints) - 1
        state = read_checkpoint(checkpoints[-1], arm, config_hash, current_epoch)
        history = state["history"]
        if (current_epoch > epochs or history[:len(inherited)] != inherited
                or [item.name for item in checkpoints] != [f"epoch_{i:03d}.pth" for i in range(first_saved_epoch, current_epoch + 1)]):
            raise ValueError(f"Unexpected frozen CFDR checkpoint: {checkpoints[-1]}")
    if not final.is_file():
        trainer.set_seed(args.seed)
        model = FrozenCFDRE2(arm, backbone).to(device)
        optimizer = torch.optim.AdamW([dict(params=list(model.cfdr.parameters()), group_name="cfdr")],
                                      lr=protocol["lr"], weight_decay=args.weight_decay)
        scaler = trainer.GradScaler(enabled=True)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
        del state, backbone
        training, _, _, _ = trainer.build_loaders(args, device, folder)
        if len(training.dataset) != 8000:
            raise ValueError("Frozen CFDR requires the original train8000")
        losses = trainer.DirectFusionLosses().to(device)
        with patch.object(trainer, "forward_losses", frozen_losses):
            for epoch in range(len(history) + 1, epochs + 1):
                trainer.set_seed(args.seed + epoch)
                training.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, training, device, args, optimizer, scaler)
                if not all(math.isfinite(value) for key, value in stats.items() if not key.startswith("grad_")):
                    raise FloatingPointError(f"Nonfinite frozen {arm} statistics at epoch {epoch}")
                row = {f"train_{key}": value if math.isfinite(value) else None for key, value in stats.items()}
                row.update(epoch=epoch, train_seconds=time.perf_counter() - started,
                           train_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
                history.append(row)
                path = folder / f"epoch_{epoch:03d}.pth"
                temporary = path.with_suffix(".tmp")
                torch.save(dict(arm=arm, epoch=epoch, variant=model.variant, args={"variant": model.variant},
                                model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                config_sha256=config_hash, history=history), temporary)
                temporary.replace(path)
                write_history(folder / "history.csv", history)
                print(f"Frozen {arm} epoch {epoch}/{epochs}: train={row['train_loss']:.6f}", flush=True)
    write_history(folder / "history.csv", history)
    entry_id = f"QXS-CFDR-V21-FROZEN-{arm}-SEED42-TRAIN8000"
    if epochs != EPOCHS:
        entry_id += f"-EPOCH{epochs:02d}"
    return dict(id=entry_id, comparable_group=entry_id, method=f"冻结 CF0 + {arm}", task="training",
                status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
                checkpoint=final.as_posix(), checkpoint_sha256=sha256_file(final),
                parent_checkpoint=protocol.get("continuation_checkpoint", PARENT.as_posix()),
                parent_checkpoint_sha256=protocol.get("continuation_checkpoint_sha256", protocol["parent_checkpoint_sha256"]),
                split_manifest=MANIFEST.as_posix(), split_manifest_sha256=protocol["split_manifest_sha256"],
                result_dir=folder.as_posix(), metadata_path=metadata.as_posix(), metric_profile="training_loss",
                output_policy="float_loss", metrics={"train_loss": history[-1]["train_loss"]},
                metric_source=dict(type="csv_row", path=(folder / "history.csv").as_posix(),
                                   match={"epoch": str(epochs)}, columns={"train_loss": "train_loss"}))


def read_checkpoint(path, arm, config_hash, epoch):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if ((state["arm"], state["variant"], state["epoch"], state["config_sha256"])
            != (arm, VARIANTS[arm], epoch, config_hash)
            or [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1))):
        raise ValueError(f"Unexpected frozen checkpoint provenance: {path}")
    return state


def smoke(device):
    backbone, args = load_backbone()
    splits, _ = manifest_pairs(DATA, MANIFEST)
    optical, sar, valid, scene = OpticalSARDataset(splits["val"][:1], patch_size=args.patch_size)[0]
    optical, sar, valid = (item[None].to(device) for item in (optical, sar, valid))
    losses = trainer.DirectFusionLosses().to(device)
    for arm in ARMS:
        trainer.set_seed(args.seed)
        model = FrozenCFDRE2(arm, backbone).to(device).train()
        if any(module.training for name, module in model.named_children() if name != "cfdr") or not model.cfdr.training:
            raise AssertionError("The backbone must stay in eval mode during CFDR training")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _ = frozen_losses(model, losses, optical, sar, valid, args)
        loss.backward()
        gradients = [p.grad for p in model.cfdr.parameters() if p.grad is not None]
        if (not torch.isfinite(loss) or not gradients or not any(torch.count_nonzero(g) for g in gradients)
                or not all(torch.isfinite(g).all() for g in gradients)
                or any(p.grad is not None for name, p in model.named_parameters() if not name.startswith("cfdr."))):
            raise AssertionError(f"Frozen gradient check failed: {arm}")
        print(dict(arm=arm, scene=scene, loss=float(loss),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val"))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS, help="Training arms")
    parser.add_argument("--to-epoch", type=int, choices=(EPOCHS, EXTENDED_EPOCHS), default=EPOCHS)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CFDR protocol")
    device = torch.device(args.device)
    if args.action == "smoke":
        smoke(device)
    elif args.action == "train":
        register_many([train_arm(arm, device, args.to_epoch) for arm in dict.fromkeys(args.arms)])
    else:
        from cfdr_evaluation import evaluate_frozen

        previous = {arm: checkpoint_path(arm) for arm in ARMS} if args.to_epoch == EXTENDED_EPOCHS else None
        evaluate_frozen(PARENT, {arm: checkpoint_path(arm, args.to_epoch) for arm in ARMS}, args.to_epoch,
                        device, previous=previous)


if __name__ == "__main__":
    main()
