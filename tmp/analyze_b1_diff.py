from pathlib import Path

import cv2
import numpy as np


root = Path("test_result/reliability_v4")
b0_dir = root / "B0_clamp" / "gray"
b1_dir = root / "B1_v5_seed42" / "gray"
optical_dir = Path("../datasets/QXSLAB_SAROPT/opt_256_oc_0.2")
common = sorted(
    set(path.name for path in b0_dir.glob("*.png"))
    & set(path.name for path in b1_dir.glob("*.png")),
    key=lambda name: int(Path(name).stem),
)

rows = []
source_rows = []
for name in common:
    b0 = cv2.imread(str(b0_dir / name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b1 = cv2.imread(str(b1_dir / name), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    difference = np.abs(b0 - b1)
    rows.append(
        (
            name,
            float(difference.mean()),
            float(np.sqrt(np.mean((b0 - b1) ** 2))),
            float((difference >= 5).mean()),
            float((difference >= 10).mean()),
            float(difference.max()),
        )
    )
    optical_bgr = cv2.imread(str(optical_dir / name), cv2.IMREAD_COLOR)
    optical_y = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    source_rows.append(
        (
            float(np.abs(b0 - optical_y).mean()),
            float(np.abs(b1 - optical_y).mean()),
            float((np.abs(b1 - optical_y) <= 2).mean()),
        )
    )

matrix = np.asarray([row[1:] for row in rows], dtype=np.float64)
print(f"common_images={len(rows)}")
print(
    "mean_over_images: "
    f"mae={matrix[:, 0].mean():.4f}, "
    f"rmse={matrix[:, 1].mean():.4f}, "
    f"pixels_ge_5={matrix[:, 2].mean():.4%}, "
    f"pixels_ge_10={matrix[:, 3].mean():.4%}, "
    f"max_abs={matrix[:, 4].max():.1f}"
)
source_matrix = np.asarray(source_rows, dtype=np.float64)
print(
    "distance_to_optical_y: "
    f"b0_mae={source_matrix[:, 0].mean():.4f}, "
    f"b1_mae={source_matrix[:, 1].mean():.4f}, "
    f"b1_pixels_within_2={source_matrix[:, 2].mean():.4%}"
)
print("smallest_mae:")
for row in sorted(rows, key=lambda item: item[1])[:5]:
    print(row)
print("largest_mae:")
for row in sorted(rows, key=lambda item: item[1], reverse=True)[:5]:
    print(row)
