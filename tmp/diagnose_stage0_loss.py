from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_OS import (
    OpticalSARLosses,
    build_models,
    configure_trainability,
    load_model_weights,
    normalized_pcgrad_backward,
)
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def grad_summary(loss, named_parameters, retain_graph):
    parameters = [parameter for _, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    by_module = {}
    flat_parts = []
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        module = name.split(".", 1)[0]
        squared = float(torch.sum(gradient.detach().float() ** 2).cpu())
        by_module[module] = by_module.get(module, 0.0) + squared
        flat_parts.append(gradient.detach().float().reshape(-1).cpu())
    flat = torch.cat(flat_parts)
    return flat, {name: value ** 0.5 for name, value in by_module.items()}


def cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-12))


def inspect(checkpoint_path, optical, sar, valid, device):
    models = build_models(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_weights(models, checkpoint)
    for model in models.values():
        model.eval()

    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    generated, _ = models["decoder"](optical, fused_base, fused_detail)

    losses = OpticalSARLosses().to(device)
    total, optical_loss, optical_l1, optical_ssim, structure_loss = (
        losses.fusion_common(
            optical,
            sar,
            generated,
            valid,
            structure_weight=1.0,
            structure_sigma=1.0,
            structure_percentile=99.0,
        )
    )
    output_grad_opt = torch.autograd.grad(
        optical_loss, generated, retain_graph=True
    )[0]
    output_grad_str = torch.autograd.grad(
        structure_loss, generated, retain_graph=True
    )[0]

    named_parameters = [
        (f"{module_name}.{parameter_name}", parameter)
        for module_name, model in models.items()
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_grad_opt, module_opt = grad_summary(
        optical_loss, named_parameters, retain_graph=True
    )
    parameter_grad_str, module_str = grad_summary(
        structure_loss, named_parameters, retain_graph=False
    )

    optical_gradient = losses.normalized_sobel(optical)
    sar_structure = losses.sar_structure(sar, valid, 1.0, 99.0)
    structure_target = torch.maximum(optical_gradient, sar_structure)
    generated_gradient = losses.normalized_sobel(generated)

    print(f"checkpoint={checkpoint_path}")
    print(
        f"loss total={float(total):.6f} optical={float(optical_loss):.6f} "
        f"l1={float(optical_l1):.6f} dssim={float(optical_ssim):.6f} "
        f"structure={float(structure_loss):.6f}"
    )
    print(
        "image stats: "
        f"output_optical_mae={float(torch.mean(torch.abs(generated-optical))):.6f}, "
        f"grad_optical={float(optical_gradient.mean()):.6f}, "
        f"grad_sar_structure={float(sar_structure.mean()):.6f}, "
        f"grad_target={float(structure_target.mean()):.6f}, "
        f"grad_output={float(generated_gradient.mean()):.6f}, "
        f"sar_target_fraction={float((sar_structure > optical_gradient).float().mean()):.4%}"
    )
    print(
        "output gradients: "
        f"opt_l2={float(torch.linalg.vector_norm(output_grad_opt)):.6f}, "
        f"str_l2={float(torch.linalg.vector_norm(output_grad_str)):.6f}, "
        f"cos={cosine(output_grad_opt.flatten(), output_grad_str.flatten()):.6f}, "
        f"str_zero_fraction={float((output_grad_str == 0).float().mean()):.4%}"
    )
    print(
        "parameter gradients: "
        f"opt_l2={float(torch.linalg.vector_norm(parameter_grad_opt)):.6f}, "
        f"str_l2={float(torch.linalg.vector_norm(parameter_grad_str)):.6f}, "
        f"cos={cosine(parameter_grad_opt, parameter_grad_str):.6f}"
    )
    for module in models:
        print(
            f"  {module}: opt={module_opt.get(module, 0.0):.6f}, "
            f"str={module_str.get(module, 0.0):.6f}"
        )
    print()


def inspect_v2(checkpoint_path, optical, sar, valid, device):
    models = build_models(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_weights(models, checkpoint)
    configure_trainability(models, "fusion_only")
    for model in models.values():
        model.eval()

    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    generated, _ = models["decoder"](optical, fused_base, fused_detail)
    losses = OpticalSARLosses().to(device)
    result = losses.fusion_common_v2(
        optical,
        sar,
        generated,
        valid,
        optical_weight=1.0,
        structure_weight=1.0,
        structure_sigma=1.0,
        structure_percentile=99.0,
        optical_lowpass_sigma=2.0,
        sar_edge_percentile=85.0,
        sar_edge_margin=0.02,
    )
    (
        total,
        optical_loss,
        optical_l1,
        optical_ssim,
        structure_loss,
        edge_fraction,
        output_optical_mae,
    ) = result
    output_grad_opt = torch.autograd.grad(
        optical_loss, generated, retain_graph=True
    )[0]
    output_grad_str = torch.autograd.grad(
        structure_loss, generated, retain_graph=True
    )[0]
    named_parameters = [
        (f"{module_name}.{parameter_name}", parameter)
        for module_name, model in models.items()
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_grad_opt, module_opt = grad_summary(
        optical_loss, named_parameters, retain_graph=True
    )
    parameter_grad_str, module_str = grad_summary(
        structure_loss, named_parameters, retain_graph=False
    )
    print(f"checkpoint={checkpoint_path} (B1-v2 diagnostic)")
    print(
        f"loss total={float(total):.6f} optical_low={float(optical_loss):.6f} "
        f"l1={float(optical_l1):.6f} dssim={float(optical_ssim):.6f} "
        f"structure={float(structure_loss):.6f}, "
        f"edge_fraction={float(edge_fraction):.4%}, "
        f"output_optical_mae={float(output_optical_mae):.6f}"
    )
    print(
        "output gradients: "
        f"opt_l2={float(torch.linalg.vector_norm(output_grad_opt)):.6f}, "
        f"str_l2={float(torch.linalg.vector_norm(output_grad_str)):.6f}, "
        f"cos={cosine(output_grad_opt.flatten(), output_grad_str.flatten()):.6f}"
    )
    print(
        "trainable parameter gradients: "
        f"opt_l2={float(torch.linalg.vector_norm(parameter_grad_opt)):.6f}, "
        f"str_l2={float(torch.linalg.vector_norm(parameter_grad_str)):.6f}, "
        f"cos={cosine(parameter_grad_opt, parameter_grad_str):.6f}"
    )
    for module in ("base_fuse", "detail_fuse"):
        print(
            f"  {module}: opt={module_opt.get(module, 0.0):.6f}, "
            f"str={module_str.get(module, 0.0):.6f}"
        )
    print()


def inspect_v3(checkpoint_path, optical, sar, valid, device):
    models = build_models(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_weights(models, checkpoint)
    configure_trainability(models, "fusion_only")
    for model in models.values():
        model.eval()
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    generated, _ = models["decoder"](optical, fused_base, fused_detail)
    losses = OpticalSARLosses().to(device)
    result = losses.fusion_common_v3(
        optical,
        sar,
        generated,
        valid,
        optical_weight=0.2,
        optical_gradient_weight=0.25,
        structure_weight=1.0,
        structure_sigma=1.0,
        structure_percentile=99.0,
        optical_lowpass_sigma=2.0,
        optical_l1_tolerance=0.02,
        optical_dssim_tolerance=0.03,
        sar_edge_percentile=85.0,
        sar_edge_margin=0.02,
    )
    (
        total,
        _,
        optical_l1,
        optical_ssim,
        optical_constraint,
        optical_gradient,
        sar_gradient,
        structure,
        edge_fraction,
        retention,
        alignment,
        output_optical_mae,
    ) = result
    named_parameters = [
        (f"{module_name}.{parameter_name}", parameter)
        for module_name, model in models.items()
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    grad_constraint, _ = grad_summary(
        optical_constraint, named_parameters, retain_graph=True
    )
    grad_optical, _ = grad_summary(
        optical_gradient, named_parameters, retain_graph=True
    )
    grad_sar, _ = grad_summary(sar_gradient, named_parameters, retain_graph=True)
    grad_total, _ = grad_summary(total, named_parameters, retain_graph=False)
    print(f"checkpoint={checkpoint_path} (B1-v3 diagnostic)")
    print(
        f"loss total={float(total):.6f}, constraint={float(optical_constraint):.6f}, "
        f"raw_l1={float(optical_l1):.6f}, raw_dssim={float(optical_ssim):.6f}, "
        f"opt_grad={float(optical_gradient):.6f}, sar_grad={float(sar_gradient):.6f}, "
        f"structure={float(structure):.6f}"
    )
    print(
        f"edge_fraction={float(edge_fraction):.4%}, retention={float(retention):.6f}, "
        f"alignment={float(alignment):.6f}, output_optical_mae={float(output_optical_mae):.6f}"
    )
    print(
        "weighted trainable gradient norms: "
        f"constraint={0.2 * float(torch.linalg.vector_norm(grad_constraint)):.6f}, "
        f"opt_grad={0.25 * float(torch.linalg.vector_norm(grad_optical)):.6f}, "
        f"sar_grad={float(torch.linalg.vector_norm(grad_sar)):.6f}, "
        f"total={float(torch.linalg.vector_norm(grad_total)):.6f}"
    )
    print()


def inspect_v5(checkpoint_path, optical, sar, valid, device):
    models = build_models(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_weights(models, checkpoint)
    configure_trainability(models, "fusion_only")
    for model in models.values():
        model.eval()
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    generated, _ = models["decoder"](optical, fused_base, fused_detail)
    losses = OpticalSARLosses().to(device)
    result = losses.fusion_common_v4(
        optical,
        sar,
        generated,
        valid,
        structure_weight=1.0,
        structure_sigma=1.0,
        structure_percentile=99.0,
        optical_lowpass_sigma=2.0,
        charbonnier_epsilon=1e-3,
    )
    total, optical_loss, _, _, structure_loss, output_optical_mae = result
    parameters = [
        parameter
        for name in ("base_fuse", "detail_fuse")
        for parameter in models[name].parameters()
        if parameter.requires_grad
    ]
    stats = normalized_pcgrad_backward(
        optical_loss, structure_loss, parameters
    )
    combined_norm = torch.sqrt(
        sum(torch.sum(parameter.grad.float() ** 2) for parameter in parameters)
    )
    print(f"checkpoint={checkpoint_path} (B1-v5 diagnostic)")
    print(
        f"optical={float(optical_loss):.6f}, structure={float(structure_loss):.6f}, "
        f"output_optical_mae={float(output_optical_mae):.6f}"
    )
    print(
        f"grad_app={float(stats['gradient_app_norm']):.6f}, "
        f"grad_str={float(stats['gradient_str_norm']):.6f}, "
        f"cosine={float(stats['gradient_cosine']):.6f}, "
        f"conflict={int(stats['gradient_conflict'].item())}, "
        f"combined_normalized_norm={float(combined_norm):.6f}"
    )
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, _ = manifest_pairs(
        Path("../datasets/QXSLAB_SAROPT"),
        Path(
            "../datasets/QXSLAB_SAROPT/splits/"
            "cddfuse_8000_1000_1000_seed20260924.json"
        ),
    )
    dataset = OpticalSARDataset(
        splits["val"], patch_size=128, training=False, augment=False, limit=4
    )
    optical, sar, valid, _ = next(iter(DataLoader(dataset, batch_size=4)))
    optical = optical.to(device)
    sar = sar.to(device)
    valid = valid.to(device)

    for checkpoint in (
        Path(
            "models/qxslab_saropt_from_ivf/"
            "CDDFuse_OS_20260924_012025/best.pth"
        ),
        Path("models/reliability_v4/B1_seed42/best.pth"),
    ):
        inspect(checkpoint, optical, sar, valid, device)
    inspect_v2(
        Path(
            "models/qxslab_saropt_from_ivf/"
            "CDDFuse_OS_20260924_012025/best.pth"
        ),
        optical,
        sar,
        valid,
        device,
    )
    inspect_v2(
        Path("models/reliability_v4/B1_v2_pilot/best.pth"),
        optical,
        sar,
        valid,
        device,
    )
    inspect_v3(
        Path(
            "models/qxslab_saropt_from_ivf/"
            "CDDFuse_OS_20260924_012025/best.pth"
        ),
        optical,
        sar,
        valid,
        device,
    )
    inspect_v5(
        Path(
            "models/qxslab_saropt_from_ivf/"
            "CDDFuse_OS_20260924_012025/best.pth"
        ),
        optical,
        sar,
        valid,
        device,
    )


if __name__ == "__main__":
    main()
