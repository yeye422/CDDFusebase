from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_OS import load_models, normalize_fused_tensor
from train_OS import build_models, load_model_weights
from utils.dataset_os import manifest_pairs


def fuse_baseline(models, optical_tensor, sar_tensor):
    encoder, decoder, base_fuse, detail_fuse = models
    optical_base, optical_detail, _ = encoder(optical_tensor)
    sar_base, sar_detail, _ = encoder(sar_tensor)
    fused_base = base_fuse(optical_base + sar_base)
    fused_detail = detail_fuse(optical_detail + sar_detail)
    return decoder(optical_tensor, fused_base, fused_detail)[0]


def fuse_a1(models, optical_tensor, sar_tensor):
    optical_base, optical_detail, _ = models["encoder"](optical_tensor)
    sar_base, sar_detail, _ = models["encoder"](sar_tensor)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail, diagnostics = models["detail_injector"](
        optical_detail, sar_detail
    )
    fused = models["decoder"](optical_tensor, fused_base, fused_detail)[0]
    return fused, diagnostics


def to_u8(tensor):
    tensor = normalize_fused_tensor(tensor, "clamp")
    return np.rint(tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)


def colorize(y, optical_ycrcb):
    output = optical_ycrcb.copy()
    output[:, :, 0] = y
    return cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, _ = manifest_pairs(
        Path("../datasets/QXSLAB_SAROPT"),
        Path(
            "../datasets/QXSLAB_SAROPT/splits/"
            "cddfuse_8000_1000_1000_seed20260924.json"
        ),
    )
    b0 = load_models(
        Path(
            "models/qxslab_saropt_from_ivf/"
            "CDDFuse_OS_20260924_012025/best.pth"
        ),
        device,
    )
    b1 = load_models(Path("models/reliability_v4/B1_v5_pilot/best.pth"), device)
    a1 = build_models(torch.device(device), variant="demand_only")
    load_model_weights(
        a1,
        torch.load(
            Path("models/reliability_v4/A1_pilot/best.pth"), map_location=device
        ),
    )
    for model in a1.values():
        model.eval()

    output_dir = Path("tmp/A1_pilot_val_preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for optical_path, sar_path in splits["val"][:4]:
            optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            optical_ycrcb = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
            optical_y = optical_ycrcb[:, :, 0].astype(np.float32) / 255.0
            sar = sar_u8.astype(np.float32) / 255.0
            optical_tensor = torch.from_numpy(optical_y)[None, None].to(device)
            sar_tensor = torch.from_numpy(sar)[None, None].to(device)
            b0_y = to_u8(fuse_baseline(b0, optical_tensor, sar_tensor))
            b1_y = to_u8(fuse_baseline(b1, optical_tensor, sar_tensor))
            a1_tensor, diagnostics = fuse_a1(a1, optical_tensor, sar_tensor)
            a1_y = to_u8(a1_tensor)
            preview = np.concatenate(
                (
                    optical_bgr,
                    cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR),
                    colorize(b0_y, optical_ycrcb),
                    colorize(b1_y, optical_ycrcb),
                    colorize(a1_y, optical_ycrcb),
                ),
                axis=1,
            )
            output_path = output_dir / f"{optical_path.stem}.png"
            cv2.imwrite(str(output_path), preview)
            print(
                output_path,
                f"demand_mean={float(diagnostics['demand_mean']):.4f}",
                f"demand_std={float(diagnostics['demand_std']):.4f}",
                f"injection_norm={float(diagnostics['injection_norm']):.4f}",
            )


if __name__ == "__main__":
    main()
