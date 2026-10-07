from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_OS import load_models, normalize_fused_tensor
from utils.dataset_os import manifest_pairs


def fuse(models, optical_y, sar, device):
    encoder, decoder, base_fuse, detail_fuse = models
    optical_tensor = torch.from_numpy(optical_y)[None, None].to(device)
    sar_tensor = torch.from_numpy(sar)[None, None].to(device)
    with torch.no_grad():
        optical_base, optical_detail, _ = encoder(optical_tensor)
        sar_base, sar_detail, _ = encoder(sar_tensor)
        fused_base = base_fuse(optical_base + sar_base)
        fused_detail = detail_fuse(optical_detail + sar_detail)
        fused, _ = decoder(optical_tensor, fused_base, fused_detail)
        fused = normalize_fused_tensor(fused, "clamp")
    return np.rint(fused.squeeze().cpu().numpy() * 255.0).astype(np.uint8)


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
    v5 = load_models(Path("models/reliability_v4/B1_v5_seed42/best.pth"), device)
    output_dir = Path("tmp/B1_v5_seed42_val_preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    for optical_path, sar_path in splits["val"][:4]:
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        optical_ycrcb = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
        optical_y = optical_ycrcb[:, :, 0].astype(np.float32) / 255.0
        sar = sar_u8.astype(np.float32) / 255.0
        b0_y = fuse(b0, optical_y, sar, device)
        v5_y = fuse(v5, optical_y, sar, device)
        b0_ycrcb = optical_ycrcb.copy()
        b0_ycrcb[:, :, 0] = b0_y
        v5_ycrcb = optical_ycrcb.copy()
        v5_ycrcb[:, :, 0] = v5_y
        panels = [
            optical_bgr,
            cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(b0_ycrcb, cv2.COLOR_YCrCb2BGR),
            cv2.cvtColor(v5_ycrcb, cv2.COLOR_YCrCb2BGR),
        ]
        preview = np.concatenate(panels, axis=1)
        cv2.imwrite(str(output_dir / f"{optical_path.stem}.png"), preview)
        print(output_dir / f"{optical_path.stem}.png")


if __name__ == "__main__":
    main()
