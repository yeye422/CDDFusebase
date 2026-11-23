"""Source-only review manifest and conservative repeated-crop screen."""

from itertools import combinations

import cv2
import numpy as np

from diagnose_dcr_v12 import csv_rows
from maintain_experiment_index import sha256_file

from .config import read_json, write_json


def correlation(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.


def verify_translation(left, right, dx, dy, config):
    height, width = left[0].shape
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    area = max(0, x1 - x0) * max(0, y1 - y0) / (height * width)
    if area < config["repeat_min_overlap"]:
        return None
    optical, sar = (correlation(a[y0:y1, x0:x1], b[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
                    for a, b in zip(left, right))
    if optical >= config["repeat_min_optical_correlation"] and sar >= config["repeat_min_sar_correlation"]:
        return dict(dx=dx, dy=dy, overlap_fraction=area, optical_correlation=optical, sar_correlation=sar)
    return None


def estimate_translation(left, right, matcher, config):
    a, desc_a = left
    b, desc_b = right
    if desc_a is None or desc_b is None or len(desc_b) < 2:
        return None
    matches = [row[0] for row in matcher.knnMatch(desc_a, desc_b, k=2)
               if len(row) == 2 and row[0].distance < 0.75 * row[1].distance]
    if len(matches) < config["repeat_min_keypoint_matches"]:
        return None
    deltas = np.array([np.subtract(b[item.trainIdx].pt, a[item.queryIdx].pt) for item in matches])
    bins, counts = np.unique(np.rint(deltas / 4).astype(int), axis=0, return_counts=True)
    center = 4 * bins[counts.argmax()]
    members = deltas[np.abs(deltas - center).max(axis=1) <= 3.]
    if len(members) < config["repeat_min_inliers"]:
        return None
    return np.rint(np.median(members, axis=0)).astype(int).tolist(), len(members)


def review_packet(packet, output, config, manifest_only=False):
    intake = read_json(packet / "intake.json")
    if not intake["source_only"]:
        raise ValueError("Expected a D1-0 source-only packet")
    files = {}
    for scene in intake["scenes"]:
        found = [sorted(packet.glob(f"{scene}_{source}.*")) for source in ("optical", "sar")]
        if any(len(items) != 1 for items in found):
            raise ValueError(f"Expected a unique optical/SAR pair for {scene}")
        pair = [items[0] for items in found]
        expected = intake["inputs"][scene]
        if [sha256_file(path) for path in pair] != list(expected.values()):
            raise ValueError(f"Review image changed for {scene}")
        files[scene] = pair
    parents = {scene: scene for scene in files}

    def parent(scene):
        while parents[scene] != scene:
            scene = parents[scene]
        return scene

    links = []
    if not manifest_only:
        cv2.setNumThreads(1)
        cv2.setRNGSeed(217)
        orb = cv2.ORB_create(nfeatures=900)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        images = {scene: [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in pair]
                  for scene, pair in files.items()}
        if any(image is None or image.shape != (256, 256) for pair in images.values() for image in pair):
            raise ValueError("Expected aligned 256×256 source images")
        features = {scene: orb.detectAndCompute(pair[0], None) for scene, pair in images.items()}
        for left, right in combinations(files, 2):
            if all(np.array_equal(a, b) for a, b in zip(images[left], images[right])):
                audit = dict(dx=0, dy=0, overlap_fraction=1., optical_correlation=1., sar_correlation=1.)
                inliers = None
            else:
                transform = estimate_translation(features[left], features[right], matcher, config)
                if transform is None:
                    continue
                (dx, dy), inliers = transform
                audit = verify_translation(images[left], images[right], dx, dy, config)
            if audit:
                parents[parent(right)] = parent(left)
                links.append(dict(left=left, right=right, inliers=inliers, **audit))
    grouped = {scene: [] for scene in parents if parent(scene) == scene}
    for scene in files:
        grouped[parent(scene)].append(scene)
    repeat_groups = {f"repeat-{index:03d}": members for index, members in enumerate(
        (members for members in grouped.values() if len(members) > 1), 1)}
    membership = {scene: key for key, members in repeat_groups.items() for scene in members}
    output.mkdir(parents=True, exist_ok=False)
    fields = ["scene", "optical", "sar", "suspected_repeat_group", "review_status", "parent_scene_id",
              "provenance_file", "reviewer", "notes"]
    csv_rows(output / "review_queue.csv", [dict(scene=scene, optical=str(pair[0]), sar=str(pair[1]),
             suspected_repeat_group=membership.get(scene), review_status="unreviewed")
             for scene, pair in files.items()], fields)
    categories = ["common", "opposite_polarity", "optical_only", "sar_only", "weak_or_textured",
                  "misaligned", "parallel"]
    csv_rows(output / "review_probes_template.csv", [],
             ["scene", "category", "source", "x1", "y1", "x2", "y2", "verification_evidence", "reviewer"])
    csv_rows(output / "suspected_repeat_links.csv", links,
             ["left", "right", "inliers", "dx", "dy", "overlap_fraction", "optical_correlation", "sar_correlation"])
    write_json(output / "source_review.json", dict(source_only=True, packet_sha256=sha256_file(packet / "intake.json"),
               manifest_only=manifest_only, source_pairs=len(files), groups=repeat_groups, links=links,
               review_categories=categories, verified_geographic_blocks=0, verified_sar_only_probes=0,
               settings={key: value for key, value in config.items() if key.startswith("repeat_")}))
    print(f"Source review: {len(files)} pairs, {len(repeat_groups)} suspected repeat groups, 0 verified geographic blocks")
