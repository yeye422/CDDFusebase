"""Build full-image and objective same-ROI visual comparisons for PAC-Mamba."""

import argparse
from pathlib import Path

import cv2
import numpy as np


METHODS = (
    "Optical",
    "SAR",
    "Wavelet",
    "U2Fusion",
    "Laplacian",
    "B0+SPD",
    "LWPD",
    "PAC-Mamba",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="+",
        default=("17667", "15605", "17848", "11039", "13550"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../comparation/results/qxslab_paper_1000_manual/visual_pac"),
    )
    parser.add_argument("--roi-size", type=int, default=64)
    return parser.parse_args()


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def scene_images(stem):
    root = Path("..")
    optical_bgr = cv2.imread(
        str(root / "datasets/QXSLAB_SAROPT/opt_256_oc_0.2" / f"{stem}.png"),
        cv2.IMREAD_COLOR,
    )
    if optical_bgr is None:
        raise FileNotFoundError(f"Optical image {stem}")
    optical = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    paths = (
        root / "datasets/QXSLAB_SAROPT/sar_256_oc_0.2" / f"{stem}.png",
        root / "comparation/results/qxslab_paper_1000_manual/wavelet/gray" / f"{stem}.png",
        root / "comparation/results/qxslab_paper_1000_manual/u2fusion/gray" / f"{stem}.png",
        root / "comparation/results/qxslab_paper_1000_manual/laplacian/gray" / f"{stem}.png",
        Path("test_result/v21_softmask/spd_nomask_on_b0_test1000_fullviz_v1/images/e2_feature") / f"{stem}.png",
        Path("test_result/lwpd_qxs_full_v1/e4_seed42_test1000/gray") / f"{stem}.png",
        Path("test_result/pac_mamba_qxs_full_v1/seed42_test1000/gray") / f"{stem}.png",
    )
    return [optical, *(read_gray(path) for path in paths)]


def strongest_change_roi(lwpd, pac, size):
    difference = cv2.absdiff(lwpd, pac).astype(np.float32)
    response = cv2.boxFilter(
        difference,
        ddepth=-1,
        ksize=(size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    _, _, _, maximum = cv2.minMaxLoc(response)
    center_x, center_y = maximum
    left = int(np.clip(center_x - size // 2, 0, lwpd.shape[1] - size))
    top = int(np.clip(center_y - size // 2, 0, lwpd.shape[0] - size))
    return left, top, size, size


def labelled_tile(gray, label, roi=None, zoom=False):
    if zoom:
        left, top, width, height = roi
        gray = gray[top : top + height, left : left + width]
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if roi is not None and not zoom:
        left, top, width, height = roi
        cv2.rectangle(
            color,
            (left, top),
            (left + width - 1, top + height - 1),
            (0, 0, 255),
            2,
        )
    header = np.full((30, color.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((header, color), axis=0)


def make_scene_board(stem, images, roi_size):
    roi = strongest_change_roi(images[-2], images[-1], roi_size)
    full = np.concatenate(
        [
            labelled_tile(image, method, roi=roi)
            for method, image in zip(METHODS, images)
        ],
        axis=1,
    )
    zoom = np.concatenate(
        [
            labelled_tile(image, f"{method} ROI", roi=roi, zoom=True)
            for method, image in zip(METHODS, images)
        ],
        axis=1,
    )
    title = np.full((38, full.shape[1], 3), 255, dtype=np.uint8)
    difference = cv2.absdiff(images[-2], images[-1])
    cv2.putText(
        title,
        (
            f"Scene {stem} | objective ROI from max local |PAC-LWPD| "
            f"| image MAE={difference.mean():.3f}/255"
        ),
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((title, full, zoom), axis=0)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boards = []
    for stem in args.ids:
        images = scene_images(stem)
        board = make_scene_board(stem, images, args.roi_size)
        output = args.output_dir / f"{stem}_comparison.png"
        cv2.imwrite(str(output), board)
        thumbnail = cv2.resize(
            board,
            (board.shape[1] // 2, board.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        boards.append(thumbnail)
        print(output)
    contact = np.concatenate(boards, axis=0)
    contact_path = args.output_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    print(contact_path)


if __name__ == "__main__":
    main()

