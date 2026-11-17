"""在固定小样本上比较 SAR 点证据、真实邻接和约束重连。"""

import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

from maintain_experiment_index import sha256_file


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "datasets/QXSLAB_SAROPT"
DEFAULT_LABELS = ROOT / "test_result/stg_no_train_v1/val8"
MAX_NODES = 192
SNAP_RADIUS = 12
NUM_REWIRES = 12
GRAPH_WEIGHT = 0.35
FALSE_LINK_BUDGETS = (0, 1, 2, 4)


def load_inputs(labels_dir, dataset):
    sample_path = labels_dir / "sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / sample["source_manifest"]
    archive_path = ROOT / sample["archive_metadata"]
    if (sha256_file(manifest_path) != sample["source_manifest_sha256"]
            or sha256_file(archive_path) != sample["archive_metadata_sha256"]):
        raise ValueError("The archive metadata or manifest has changed.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    scenes = sample["scene_ids"]
    if len(scenes) != 8 or len(set(scenes)) != 8:
        raise ValueError("This audit uses eight distinct fixed validation scenes.")
    if scenes != [archive["scene_ids"][i] for i in sample["sampling_indices_zero_based"]]:
        raise ValueError("The samples do not match the archived evaluation order.")
    if not set(scenes).issubset(manifest["splits"]["val"]):
        raise ValueError("A scene is outside the validation split.")
    labels_path = labels_dir / "source_labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    keys = [(row["scene"], row["id"]) for row in labels]
    if set(scene for scene, _ in keys) != set(scenes) or len(keys) != len(set(keys)):
        raise ValueError("Invalid or duplicate source annotations.")
    paths = {
        scene: (
            dataset / manifest["optical_dir"] / f"{scene}.png",
            dataset / manifest["sar_dir"] / f"{scene}.png",
            archive_path.parent / "e2/gray" / f"{scene}.png",
        )
        for scene in scenes
    }
    return archive, sample, labels, paths


def endpoints(row):
    return np.array([[int(row["x1"]), int(row["y1"])],
                     [int(row["x2"]), int(row["y2"])]])


def draw_review(labels, paths, output):
    colors = {"connected": "#49e2ff", "disconnected": "#ff9b50", "uncertain": "#e5ca65"}
    for start in range(0, len(paths), 4):
        sheet = Image.new("RGB", (3 * 276, 4 * 278), "#1c2026")
        draw = ImageDraw.Draw(sheet)
        for row_index, scene in enumerate(list(paths)[start:start + 4]):
            for column, (title, path) in enumerate(zip(("Optical", "SAR", "Archived E2"), paths[scene])):
                x, y = 10 + 276 * column, 20 + 278 * row_index
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB"), (x, y))
                draw.text((x, y - 17), f"{scene}  {title}", fill="white")
                for row in labels:
                    if row["scene"] != scene:
                        continue
                    a, b = endpoints(row) + (x, y)
                    color = colors[row["label"]]
                    # End marks indicate the segment without obscuring the interior.
                    for point in (a, b):
                        px, py = point
                        draw.ellipse((px-3, py-3, px+3, py+3), outline=color, width=1)
                    draw.text(tuple(a + (4, -13)), row["id"], fill=color, stroke_width=1, stroke_fill="black")
        sheet.save(output / f"review_{start // 4 + 1}.png")


def sar_peaks(sar):
    sar = sar.astype(np.float32) / 255
    fine = cv2.GaussianBlur(sar, (0, 0), 0.85)
    local = cv2.GaussianBlur(fine, (0, 0), 3.5)
    variance = cv2.GaussianBlur(fine**2, (0, 0), 3.5) - local**2
    prominence = (fine - local) / np.sqrt(np.maximum(variance, 0) + 0.025**2)
    valid = cv2.erode((sar > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    local_max = cv2.dilate(prominence, np.ones((7, 7), np.uint8))
    y, x = np.where(valid & (prominence == local_max) & (prominence >= 0.8))
    order = np.argsort(-prominence[y, x], kind="stable")[:MAX_NODES]
    xy = np.column_stack((x[order], y[order]))
    # Rank preserves peak order while avoiding cross-image grey-level comparisons.
    strength = 1 - np.arange(len(xy)) / max(1, len(xy))
    return xy, strength


def real_edges(distances):
    edges = set()
    for i, row in enumerate(distances):
        neighbors = np.flatnonzero((row > 0) & (row <= 36))
        order = np.argsort(row[neighbors], kind="stable")[:3]
        edges.update(tuple(sorted((i, int(j)))) for j in neighbors[order])
    return edges


def rewire(edges, distances, rng, mode="length_bins"):
    """Double-edge swaps preserve each degree and limit spatial displacement."""
    current = set(edges)
    slots = sorted(edges)
    if not slots:
        return current, 0
    swaps = 0
    length_bin = np.ceil(distances / 8).astype(int)
    initial_length = sum(distances[edge] for edge in edges)
    current_length = initial_length
    attempts = 40 if mode == "length_bins" else 120
    proposals = rng.integers((0, 0, 0), (len(slots), len(slots), 2), size=(attempts * len(slots), 3))
    for first, second, flip in proposals:
        a, b = slots[first]
        c, d = slots[second]
        if len({a, b, c, d}) < 4:
            continue
        old = ((a, b), (c, d))
        if flip:
            c, d = d, c
        new = (tuple(sorted((a, c))), tuple(sorted((b, d))))
        if any(edge in current or distances[edge] > 36 for edge in new):
            continue
        new_length = current_length + sum(distances[e] for e in new) - sum(distances[e] for e in old)
        if mode == "length_bins" and sorted(length_bin[e] for e in old) != sorted(length_bin[e] for e in new):
            continue
        if mode == "local" and not 0.85 * initial_length <= new_length <= 1.15 * initial_length:
            continue
        current.difference_update(old)
        current.update(new)
        current_length = new_length
        slots[first], slots[second] = new
        swaps += 1
        if swaps >= 3 * len(slots):
            break
    return current, swaps


def adjacency(edges, size):
    graph = [set() for _ in range(size)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def link_support(a, b, graph, distances, max_hops):
    if b in graph[a]:
        return 1.0
    length = distances[a, b]
    max_length = 1.25 * length
    shortest = np.inf
    frontier = {int(a): 0.0}
    for _ in range(max_hops):
        next_frontier = {}
        for node, cost in frontier.items():
            for neighbor in graph[node]:
                total = cost + distances[node, neighbor]
                if total + distances[neighbor, b] > max_length:
                    continue
                if neighbor == b:
                    shortest = min(shortest, total)
                else:
                    next_frontier[neighbor] = min(next_frontier.get(neighbor, np.inf), total)
        frontier = next_frontier
        if not frontier:
            break
    return 0.8 * length / shortest


def load_visibility(path, labels):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        values = list(csv.DictReader(handle))
    result = {(row["scene"], row["id"]): row["e2_visibility"] for row in values}
    expected = {(row["scene"], row["id"]) for row in labels if row["label"] == "connected"}
    if set(result) != expected or len(result) != len(values):
        raise ValueError("Every connected source segment needs one E2 readout.")
    if set(result.values()) - {"present", "missing", "uncertain"}:
        raise ValueError("Unsupported E2 visibility label.")
    return result


def score_scene(sar, labels, visibility, seed, max_hops, rewire_mode):
    points, strength = sar_peaks(sar)
    distances = cdist(points, points)
    edges = real_edges(distances)
    real = adjacency(edges, len(points))
    rng = np.random.default_rng(seed)
    nulls = [rewire(edges, distances, rng, rewire_mode) for _ in range(NUM_REWIRES)]
    graphs = [adjacency(edge_set, len(points)) for edge_set, _ in nulls]
    valid = []
    for row in labels:
        key = (row["scene"], row["id"])
        scored = dict(row, e2_visibility=visibility.get(key, ""), candidate=False)
        if len(points):
            gaps = cdist(endpoints(row), points)
            a, b = np.argmin(gaps, axis=1)
            length = distances[a, b]
            scored.update(a_x=points[a, 0], a_y=points[a, 1], b_x=points[b, 0], b_y=points[b, 1],
                          snap_a=float(gaps[0, a]), snap_b=float(gaps[1, b]))
            if gaps[0, a] <= SNAP_RADIUS and gaps[1, b] <= SNAP_RADIUS and 14 <= length <= 80:
                point = float(min(strength[a], strength[b]) * np.exp(-(length / 48)**2))
                support = float(link_support(a, b, real, distances, max_hops))
                scored.update(candidate=True, point=point, graph=point + GRAPH_WEIGHT * support,
                              graph_support=support)
                for i, graph in enumerate(graphs):
                    scored[f"rewire_{i:02d}"] = point + GRAPH_WEIGHT * link_support(a, b, graph, distances, max_hops)
        valid.append(scored)
    audit = {"nodes": len(points), "edges": len(edges),
             "mean_edge_length": sum(distances[e] for e in edges) / len(edges) if edges else None,
             "rewire_mean_edge_length": [sum(distances[e] for e in null) / len(null) if null else None
                                         for null, _ in nulls],
             "retained_edge_fraction": [len(edges & null) / len(edges) if edges else None for null, _ in nulls],
             "swap_counts": [swaps for _, swaps in nulls]}
    return valid, audit


def hits_at_budget(rows, method, false_budget):
    available = sorted((row for row in rows if row["candidate"]), key=lambda row: -row[method])
    tp, fp = 0, 0
    for _, block in groupby(available, key=lambda row: row[method]):
        group = list(block)
        false_links = sum(row["label"] == "disconnected" for row in group)
        if fp + false_links > false_budget:
            break
        fp += false_links
        tp += len(group) - false_links
    return {"hits": tp, "false_links": fp}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data-root", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--max-hops", type=int, choices=(2, 6), default=2)
    parser.add_argument("--rewire-mode", choices=("length_bins", "local"), default="length_bins")
    args = parser.parse_args()
    archive, sample, labels, paths = load_inputs(args.labels_dir, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.review:
        draw_review(labels, paths, args.output_dir)
        return
    checkpoint = archive["checkpoints"]["e2"]
    checkpoint_hash = archive["checkpoint_hashes"]["e2"]
    if sha256_file(ROOT / checkpoint) != checkpoint_hash:
        raise ValueError("The archived E2 checkpoint hash does not match.")
    visibility_path = args.labels_dir / "e2_visibility.csv"
    visibility = load_visibility(visibility_path, labels)
    rows, audit = [], {}
    for scene, (_, sar_path, _) in paths.items():
        with Image.open(sar_path) as image:
            sar = np.asarray(image.convert("L"))
        scene_labels = [row for row in labels if row["scene"] == scene]
        scored, audit[scene] = score_scene(sar, scene_labels, visibility, int(scene), args.max_hops, args.rewire_mode)
        rows.extend(scored)
    positives = [row for row in rows if row["e2_visibility"] == "missing"]
    negatives = [row for row in rows if row["label"] == "disconnected"]
    primary = positives + negatives
    methods = ("point", "graph", *(f"rewire_{i:02d}" for i in range(NUM_REWIRES)))
    operating_points = [dict(budget=budget, method=method, **hits_at_budget(primary, method, budget))
                        for budget in FALSE_LINK_BUDGETS for method in methods]
    summary = dict(
        scenes=len(paths), source_connected=len(visibility),
        connected_with_candidates=sum(row["candidate"] for row in rows if row["label"] == "connected"),
        e2_missing=len(positives), e2_missing_scenes=len({row["scene"] for row in positives}),
        missing_with_candidates=sum(row["candidate"] for row in positives),
        negatives=len(negatives), negatives_with_candidates=sum(row["candidate"] for row in negatives),
        graph_supported_missing=sum(row.get("graph_support", 0) > 0 for row in positives),
        graph_supported_negatives=sum(row.get("graph_support", 0) > 0 for row in negatives),
        graph_by_scene=audit, operating_points=operating_points,
    )
    write_csv(args.output_dir / "segments.csv", rows)
    write_csv(args.output_dir / "operating_points.csv", operating_points)
    metadata = dict(
        **sample, checkpoint=checkpoint, checkpoint_sha256=checkpoint_hash,
        script_sha256=sha256_file(Path(__file__)), labels_sha256=sha256_file(args.labels_dir / "source_labels.csv"),
        visibility_sha256=sha256_file(visibility_path), max_nodes=MAX_NODES, snap_radius=SNAP_RADIUS,
        graph_weight=GRAPH_WEIGHT, num_rewires=NUM_REWIRES, max_hops=args.max_hops, rewire_mode=args.rewire_mode,
        image_sha256={scene: {name: sha256_file(path) for name, path in zip(("optical", "sar", "e2"), files)}
                      for scene, files in paths.items()},
    )
    for name, payload in (("summary.json", summary), ("run_metadata.json", metadata)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: val for key, val in summary.items() if not isinstance(val, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
