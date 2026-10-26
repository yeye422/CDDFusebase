"""Create labeled contact sheets for aligned SPD method outputs."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IDS = ("43", "50", "59", "73", "77", "105", "109", "130")
METHODS = (
    ("Optical Y", ROOT.parent / "datasets/QXSLAB_SAROPT/opt_256_oc_0.2"),
    ("SAR", ROOT.parent / "datasets/QXSLAB_SAROPT/sar_256_oc_0.2"),
    ("B0", ROOT / "test_result/reliability_v4/B0_clamp/gray"),
    (
        "B0 + Full SPD",
        ROOT / "test_result/v21_softmask/spd_nomask_on_b0_test1000/images/e2_feature",
    ),
    (
        "C0-minimax",
        ROOT / "test_result/reliability_v5/frequency_minimax_seed42/gray",
    ),
    (
        "C0 + SPD",
        ROOT / "test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/images/e2_feature",
    ),
    (
        "Detail-only",
        ROOT / "test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/images/e2_feature",
    ),
)


def read_luminance(label, directory, stem):
    path = next(iter(directory.glob(f"{stem}.*")), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label} image {stem} in {directory}")
    if label == "Optical Y":
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {path}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def labeled_header(width, height, text):
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = 0.55 if len(text) < 13 else 0.43
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    x = max((width - text_width) // 2, 3)
    y = (height + text_height) // 2 - 2
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )
    return image


def make_sheet(stems, output):
    tile = 128
    header_height = 30
    id_width = 44
    sheet = np.full(
        (header_height + len(stems) * tile, id_width + len(METHODS) * tile, 3),
        255,
        dtype=np.uint8,
    )
    for column, (label, _) in enumerate(METHODS):
        left = id_width + column * tile
        sheet[:header_height, left : left + tile] = labeled_header(
            tile, header_height, label
        )
    for row, stem in enumerate(stems):
        top = header_height + row * tile
        id_panel = np.full((tile, id_width, 3), 245, dtype=np.uint8)
        cv2.putText(
            id_panel,
            stem,
            (4, tile // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
        sheet[top : top + tile, :id_width] = id_panel
        for column, (label, directory) in enumerate(METHODS):
            image = read_luminance(label, directory, stem)
            if image.shape[:2] != (tile, tile):
                image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            left = id_width + column * tile
            sheet[top : top + tile, left : left + tile] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not save {output}")


def make_difference_sheet(stems, output):
    tile = 192
    header_height = 30
    id_width = 44
    columns = (
        "B0",
        "Full SPD",
        "|Full-B0| x6",
        "C0-minimax",
        "|C0-B0| x6",
        "Detail-only",
        "|Detail-B0| x6",
    )
    directories = {label: directory for label, directory in METHODS}
    sheet = np.full(
        (header_height + len(stems) * tile, id_width + len(columns) * tile, 3),
        255,
        dtype=np.uint8,
    )
    for column, label in enumerate(columns):
        left = id_width + column * tile
        sheet[:header_height, left : left + tile] = labeled_header(
            tile, header_height, label
        )
    for row, stem in enumerate(stems):
        top = header_height + row * tile
        cv2.putText(
            sheet,
            stem,
            (4, top + tile // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
        b0 = read_luminance("B0", directories["B0"], stem)
        full = read_luminance(
            "B0 + Full SPD", directories["B0 + Full SPD"], stem
        )
        c0 = read_luminance("C0-minimax", directories["C0-minimax"], stem)
        detail = read_luminance(
            "Detail-only", directories["Detail-only"], stem
        )
        images = (
            b0,
            full,
            np.clip(cv2.absdiff(full, b0).astype(np.float32) * 6.0, 0, 255).astype(np.uint8),
            c0,
            np.clip(cv2.absdiff(c0, b0).astype(np.float32) * 6.0, 0, 255).astype(np.uint8),
            detail,
            np.clip(cv2.absdiff(detail, b0).astype(np.float32) * 6.0, 0, 255).astype(np.uint8),
        )
        for column, image in enumerate(images):
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            left = id_width + column * tile
            sheet[top : top + tile, left : left + tile] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not save {output}")


def main():
    output_dir = ROOT / "test_result/v21_softmask/visual_comparison"
    make_sheet(IDS[:4], output_dir / "comparison_1.jpg")
    make_sheet(IDS[4:], output_dir / "comparison_2.jpg")
    make_difference_sheet(("43", "59", "105", "130"), output_dir / "difference_x6.jpg")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
