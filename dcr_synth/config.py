"""Freeze the experiment inputs before evaluating registered scenes."""

import json
from pathlib import Path

from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def load_config(path):
    stage_config = read_json(path)
    protocol = read_json(ROOT / stage_config["protocol"])
    settings = protocol["settings"]
    if settings["eta"] <= 0 or settings["u"] <= 0 or settings["atom_floor"] <= 0:
        raise ValueError("eta, u and atom_floor must be positive")
    if len(set(settings["fd_steps"])) < 2 or min(settings["fd_steps"]) <= 0:
        raise ValueError("Register at least two positive FD steps")
    if sorted(set(settings["doses_gl"])) != settings["doses_gl"] or not (
            {0.245, 0.5} <= set(settings["doses_gl"])) or min(settings["doses_gl"]) <= 0:
        raise ValueError("Include both registered dose budgets in a strictly increasing sequence")
    ids = set()
    for case in protocol["cases"]:
        if case["id"] in ids:
            raise ValueError("Duplicate case identity")
        ids.add(case["id"])
        if case["role"] not in {"development", "same_image", "difficult", "preserve", "holdout", "negative"}:
            raise ValueError(f"Invalid case role for {case['id']}")
        windows = case["windows"]
        if not 2 <= len(windows) <= 4 or not case["protections"]:
            raise ValueError(f"Register 2–4 windows and protection readouts for {case['id']}")
        radius = settings["window_size"] // 2
        for x, y, _ in windows:
            if not (radius <= x < 256 - radius and radius <= y < 256 - radius):
                raise ValueError(f"Out-of-frame action window for {case['id']}")
        lines = [case["target"]] + [item["line"] for item in case["protections"]]
        for line in lines:
            if any(not 0 <= line[key] < 256 for key in ("x1", "y1", "x2", "y2")) or (
                    (line["x1"], line["y1"]) == (line["x2"], line["y2"])):
                raise ValueError(f"Invalid probe in {case['id']}")
        if any(item["tolerance_gl"] <= 0 for item in case["protections"]):
            raise ValueError(f"Nonpositive protection tolerance for {case['id']}")
    if settings["window_size"] != 15:
        raise ValueError("V1.2 uses registered 15×15 windows")
    return stage_config, protocol


def freeze(protocol, manifest, inputs):
    paths = {key: protocol[key] for key in ("checkpoint", "split_manifest", "archive")}
    digests = {key: sha256_file(ROOT / path) for key, path in paths.items()}
    for key in ("checkpoint", "split_manifest"):
        if digests[key] != protocol[f"{key}_sha256"]:
            raise ValueError(f"{key} differs from the registered SHA-256")
    if any(case["scene"] not in manifest["splits"]["val"] for case in protocol["cases"]):
        raise ValueError("All cases must belong to the frozen validation split")
    holdouts = [case for case in protocol["cases"] if case["role"] == "holdout"]
    non_holdout_scenes = {case["scene"] for case in protocol["cases"] if case["role"] != "holdout"}
    if any(case["scene"] in non_holdout_scenes for case in holdouts):
        raise ValueError("Holdout case shares a crop with development or archive cases")
    blocks = {case.get("geo_block") for case in protocol["cases"] if case["role"] != "holdout"}
    confirmed = {case["geo_block"] for case in holdouts if case.get("geo_block")
                 and case["geo_block"] not in blocks and case.get("verified_natural_missing")}
    return dict(protocol=protocol, sha256=digests, sources=inputs, metric_profile=DEFAULT_PROFILE,
                confirmed_holdout_blocks=len(confirmed),
                sar_only_probes=sum(item["role"] == "sar_only" and item.get("verified", False)
                                    for case in protocol["cases"] for item in case["protections"]))
