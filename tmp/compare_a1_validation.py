import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("test_result/reliability_v4/validation")
FILES = {
    "B1": ROOT / "B1_v5_pilot_200/metrics.csv",
    "A1_learned": ROOT / "A1_pilot_learned_200/metrics.csv",
    "A1_fixed": ROOT / "A1_pilot_fixed05_200/metrics.csv",
    "A1prime": ROOT / "A1prime_pilot_200/metrics.csv",
}
LOWER_IS_BETTER = {"Nabf"}


def load(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    names = rows[0][1:]
    values = {
        row[0]: np.asarray(row[1:], dtype=np.float64)
        for row in rows[1:]
        if row[0] not in {"mean", "std"}
    }
    return names, values


names, baseline = load(FILES["B1"])
_, learned = load(FILES["A1_learned"])
_, fixed = load(FILES["A1_fixed"])
_, residual = load(FILES["A1prime"])
ids = sorted(set(baseline) & set(learned) & set(fixed), key=int)


def report(reference_name, reference, candidate_name, candidate):
    print(f"{candidate_name} vs {reference_name} ({len(ids)} paired images)")
    reference_matrix = np.stack([reference[name] for name in ids])
    candidate_matrix = np.stack([candidate[name] for name in ids])
    for index, metric in enumerate(names):
        ref = reference_matrix[:, index]
        cand = candidate_matrix[:, index]
        relative = (cand.mean() / ref.mean() - 1.0) * 100.0
        wins = cand < ref if metric in LOWER_IS_BETTER else cand > ref
        print(
            f"{metric:8s} mean_change={relative:+8.3f}% "
            f"paired_win_rate={wins.mean():7.2%}"
        )
    print()


report("B1", baseline, "A1_learned", learned)
report("A1_fixed", fixed, "A1_learned", learned)
report("B1", baseline, "A1prime", residual)
report("A1_fixed", fixed, "A1prime", residual)

print("Metric correlations on B1 validation outputs")
baseline_matrix = np.stack([baseline[name] for name in ids])
name_to_index = {name: index for index, name in enumerate(names)}
for left, right in (
    ("Nabf", "Qabf"),
    ("Nabf", "AG"),
    ("Nabf", "SF"),
    ("Nabf", "FMI"),
    ("Nabf", "VIF"),
    ("Qabf", "AG"),
    ("Qabf", "SF"),
    ("Qabf", "FMI"),
    ("Qabf", "VIF"),
):
    correlation = np.corrcoef(
        baseline_matrix[:, name_to_index[left]],
        baseline_matrix[:, name_to_index[right]],
    )[0, 1]
    print(f"corr({left},{right})={correlation:+.4f}")

print("\nCorrelations of paired A1prime-B1 changes")
residual_matrix = np.stack([residual[name] for name in ids])
changes = residual_matrix - baseline_matrix
for left, right in (
    ("Nabf", "Qabf"),
    ("Nabf", "SSIM"),
    ("Nabf", "VIF"),
    ("Nabf", "FMI"),
    ("Qabf", "VIF"),
    ("Qabf", "FMI"),
):
    correlation = np.corrcoef(
        changes[:, name_to_index[left]],
        changes[:, name_to_index[right]],
    )[0, 1]
    print(f"corr(delta {left},delta {right})={correlation:+.4f}")

b1_gray = ROOT / "B1_v5_pilot_200/gray"
residual_gray = ROOT / "A1prime_pilot_200/gray"
pixel_rows = []
for name in ids:
    b1_image = cv2.imread(str(b1_gray / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    residual_image = cv2.imread(
        str(residual_gray / f"{name}.png"), cv2.IMREAD_GRAYSCALE
    )
    difference = np.abs(b1_image.astype(np.float32) - residual_image.astype(np.float32))
    pixel_rows.append(
        (
            float(difference.mean()),
            float(np.sqrt(np.mean(difference ** 2))),
            float((difference >= 5).mean()),
            float((difference >= 10).mean()),
        )
    )
pixel_matrix = np.asarray(pixel_rows)
print("\nA1prime vs B1 output-pixel difference")
print(
    f"MAE={pixel_matrix[:,0].mean():.4f}/255, "
    f"RMSE={pixel_matrix[:,1].mean():.4f}/255, "
    f"pixels>=5={pixel_matrix[:,2].mean():.2%}, "
    f"pixels>=10={pixel_matrix[:,3].mean():.2%}"
)
