import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_OS import OpticalSARLosses, build_models, load_model_weights
from utils.dataset_os import OpticalSARDataset, manifest_pairs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits, _ = manifest_pairs(
    Path("../datasets/QXSLAB_SAROPT"),
    Path("../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json"),
)
dataset = OpticalSARDataset(
    splits["val"], patch_size=128, training=False, augment=False, limit=2
)
optical, sar, valid, names = next(iter(DataLoader(dataset, batch_size=2)))
optical = optical.to(device)
sar = sar.to(device)
valid = valid.to(device)

models = build_models(device)
checkpoint_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "models/reliability_v4/B1_seed42/best.pth"
)
checkpoint = torch.load(
    checkpoint_path, map_location=device
)
load_model_weights(models, checkpoint)
for model in models.values():
    model.eval()

optical_base, optical_detail, _ = models["encoder"](optical)
sar_base, sar_detail, _ = models["encoder"](sar)
fused_base = models["base_fuse"](optical_base + sar_base)
fused_detail = models["detail_fuse"](optical_detail + sar_detail)
generated, _ = models["decoder"](optical, fused_base, fused_detail)

losses = OpticalSARLosses().to(device)
total, optical_loss, optical_l1, optical_ssim, structure_loss = losses.fusion_common(
    optical, sar, generated, valid, 1.0, 1.0, 99.0
)

all_parameters = []
parameter_groups = {}
for name, model in models.items():
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_groups[name] = parameters
    all_parameters.extend(parameters)


def gradients(loss):
    values = torch.autograd.grad(loss, all_parameters, retain_graph=True, allow_unused=True)
    return [
        torch.zeros_like(parameter) if value is None else value
        for parameter, value in zip(all_parameters, values)
    ]


grad_optical = gradients(optical_loss)
grad_structure = gradients(structure_loss)

print(f"device={device}, checkpoint={checkpoint_path}, samples={list(names)}")
print(
    f"losses total={total.item():.6f}, optical={optical_loss.item():.6f}, "
    f"l1={optical_l1.item():.6f}, dssim={optical_ssim.item():.6f}, "
    f"structure={structure_loss.item():.6f}"
)
print(f"output_optical_mae={(generated - optical).abs().mean().item():.6f}")
with torch.no_grad():
    optical_gradient = losses.normalized_sobel(optical)
    sar_structure = losses.sar_structure(sar, valid, 1.0, 99.0)
    target = torch.maximum(optical_gradient, sar_structure)
    valid_pixels = valid > 0.5
    print(
        "gradient_stats: "
        f"optical_mean={optical_gradient[valid_pixels].mean().item():.6f}, "
        f"sar_mean={sar_structure[valid_pixels].mean().item():.6f}, "
        f"target_mean={target[valid_pixels].mean().item():.6f}, "
        f"sar_wins={(sar_structure[valid_pixels] > optical_gradient[valid_pixels]).float().mean().item():.4%}"
    )

offset = 0
for name, parameters in parameter_groups.items():
    count = len(parameters)
    gradients_optical = grad_optical[offset : offset + count]
    gradients_structure = grad_structure[offset : offset + count]
    offset += count
    norm_optical_sq = sum(float((value.float() ** 2).sum()) for value in gradients_optical)
    norm_structure_sq = sum(
        float((value.float() ** 2).sum()) for value in gradients_structure
    )
    dot = sum(
        float((left.float() * right.float()).sum())
        for left, right in zip(gradients_optical, gradients_structure)
    )
    norm_optical = math.sqrt(norm_optical_sq)
    norm_structure = math.sqrt(norm_structure_sq)
    cosine = dot / max(norm_optical * norm_structure, 1e-12)
    print(
        f"{name}: grad_optical={norm_optical:.6e}, "
        f"grad_structure={norm_structure:.6e}, cosine={cosine:.6f}, "
        f"clip=1e-2"
    )
