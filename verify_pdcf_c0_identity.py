"""P0 single-sample verification for the zero-initialized PDCF wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from pdcf_model import PDCFModel


DEFAULT_CHECKPOINT = Path(
    "models/reliability_v5/frequency_minimax_seed42/best.pth"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that zero-initialized PDCF exactly reproduces C0"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height < 4 or args.width < 4:
        raise ValueError("height and width must both be at least 4")
    checkpoint_path = args.checkpoint.resolve()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = PDCFModel.from_c0_checkpoint(
        checkpoint_path, map_location=device
    ).to(device).eval()
    optical = torch.rand(1, 1, args.height, args.width, device=device)
    sar = torch.rand(1, 1, args.height, args.width, device=device)
    with torch.inference_mode():
        c0_output, c0_decoder_feature = model.forward_c0(optical, sar)
        pdcf_output, pdcf_decoder_feature, diagnostics = model(
            optical, sar, return_diagnostics=True
        )

    output_max_abs = float((pdcf_output - c0_output).abs().max())
    decoder_max_abs = float(
        (pdcf_decoder_feature - c0_decoder_feature).abs().max()
    )
    delta_base_max_abs = float(diagnostics["delta_base"].abs().max())
    delta_detail_max_abs = float(diagnostics["delta_detail"].abs().max())
    exact_identity = bool(
        torch.equal(pdcf_output, c0_output)
        and torch.equal(pdcf_decoder_feature, c0_decoder_feature)
        and delta_base_max_abs == 0.0
        and delta_detail_max_abs == 0.0
    )
    frozen_names = ("encoder", "decoder", "base_fuse", "detail_fuse")
    frozen_c0 = all(
        not parameter.requires_grad
        for name in frozen_names
        for parameter in getattr(model, name).parameters()
    )
    result = {
        "protocol": "PDCF_P0_C0_identity_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "input_shape": [1, 1, args.height, args.width],
        "seed": args.seed,
        "exact_identity": exact_identity,
        "frozen_c0": frozen_c0,
        "output_max_abs": output_max_abs,
        "decoder_feature_max_abs": decoder_max_abs,
        "delta_base_max_abs": delta_base_max_abs,
        "delta_detail_max_abs": delta_detail_max_abs,
        "c0_parameters": sum(
            parameter_count(getattr(model, name)) for name in frozen_names
        ),
        "pdcf_parameters": parameter_count(model.core),
        "pdcf_trainable_parameters": parameter_count(
            model.core, trainable_only=True
        ),
    }
    if not exact_identity:
        raise AssertionError(json.dumps(result, ensure_ascii=False, indent=2))
    if not frozen_c0:
        raise AssertionError("C0 parameters are not fully frozen")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

