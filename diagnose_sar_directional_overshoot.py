"""Decompose NAB/F on 12 rank-spaced archived E2 validation images."""

import os
from pathlib import Path
from statistics import mean
import tempfile

from diagnose_source_interval_tolerance import BANDS, analyze_scene, read_scores, write_csv
from e2_source_interval_loss import INTERVAL
from run_e2_matched_no_pac import register
from run_e2_sar_directional import EXPERIMENT
from run_e2_scratch_twostage import BASE_ANCHOR
from run_qxs_reference_test1000 import DATA, read_json, write_json
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


OUTPUT = Path("test_result/e2_sar_directional_overshoot_val12_v1")
GROUP = "QXS-E2-SAR-DIRECTIONAL-OVERSHOOT-VAL12-v1"
METRICS = ("e2_nabf", "directional_nabf", "delta_nabf", *(f"delta_{band}" for band in BANDS))


def main():
    os.chdir(Path(__file__).resolve().parent)
    manifest = EXPERIMENT.manifest
    manifest_hash = sha256_file(manifest)
    result_meta = EXPERIMENT.result / "run_metadata.json"
    result = read_json(result_meta)
    checkpoints = {"e2": BASE_ANCHOR.final, "directional": EXPERIMENT.final}
    hashes = {key: sha256_file(path) for key, path in checkpoints.items()}
    splits, _ = manifest_pairs(DATA, manifest)
    pairs = {pair[0].stem: pair for pair in splits["val"]}
    expected = dict(scene_ids=list(pairs), split="val", offset=0, split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8")
    if (len(pairs) != 1000 or any(result.get(key) != value for key, value in expected.items())
            or any(result["checkpoint_hashes"].get(key) != value for key, value in hashes.items())):
        raise ValueError("Archived results do not match the directional val1000 protocol")
    metric_paths = {key: EXPERIMENT.result / key / "metrics.csv" for key in checkpoints}
    scores = {key: read_scores(path) for key, path in metric_paths.items()}
    if any(list(values) != list(pairs) for values in scores.values()):
        raise ValueError("Unexpected scene order in the saved metric files")
    ranked = sorted(pairs, key=lambda scene: (scores["e2"][scene], int(scene)))
    halves = {"low_half": ranked[:500], "high_half": ranked[500:]}
    selection = {group: [half[(2*i+1)*len(half)//12] for i in range(6)] for group, half in halves.items()}
    spec = {key: INTERVAL[key] for key in ("upper_ratio", "upper_slack")}
    metadata = dict(expected, scene_ids=[scene for half in selection.values() for scene in half],
                    selection=selection, selection_rule="Six equal-rank bin midpoints per baseline NAB/F half",
                    split_manifest=manifest.as_posix(), checkpoint_hashes=hashes, tolerance=spec,
                    source_hashes={path.as_posix(): sha256_file(path)
                                   for path in (result_meta, *metric_paths.values())})
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != metadata:
            raise FileExistsError(f"Existing audit uses a different protocol: {OUTPUT}")
        for path, digest in read_json(OUTPUT / "input_hashes.json").items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"Archived image changed: {path}")
        summaries = read_json(OUTPUT / "summary.json")
    else:
        paths = dict(e2=BASE_ANCHOR.result / "gray", directional=EXPERIMENT.result / "directional" / "gray")
        rows, input_hashes = [], {}
        for name, ids in selection.items():
            for scene in ids:
                row, inputs = analyze_scene(pairs[scene], paths, scores, 0, spec, candidate_key="directional")
                del row["baseline_nabf_quartile"]
                rows.append(dict(group=name, **row))
                input_hashes.update(inputs)
        summaries = {}
        for name, selected in (("all12", rows), *((name, [row for row in rows if row["group"] == name])
                                                   for name in selection)):
            summaries[name] = dict(sample_count=len(selected), **{
                key: mean(row[key] for row in selected) for key in METRICS
            })
        with tempfile.TemporaryDirectory(prefix=".directional-audit-", dir=OUTPUT.parent) as temporary:
            work = Path(temporary) / "results"
            work.mkdir()
            write_csv(work / "per_image.csv", rows)
            write_csv(work / "summary.csv", [dict(group=group, **values) for group, values in summaries.items()])
            write_json(work / "summary.json", summaries)
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "input_hashes.json", input_hashes)
            work.rename(OUTPUT)
    metrics = {name: summaries["all12"][name] for name in METRICS}
    register(dict(id=GROUP, method="方向收缩版存档图轻微过冲分解", dataset="QXS-SAROPT", task="fusion",
                  status="diagnostic_fixed_subset", comparable_group=GROUP, split="val", offset=0, sample_count=12,
                  split_manifest=manifest.as_posix(), split_manifest_sha256=manifest_hash,
                  checkpoint=checkpoints["directional"].as_posix(), checkpoint_sha256=hashes["directional"],
                  parent_checkpoint=checkpoints["e2"].as_posix(), parent_checkpoint_sha256=hashes["e2"],
                  metric_profile=DEFAULT_PROFILE, output_policy="saved_uint8", result_dir=OUTPUT.as_posix(),
                  metadata_path=(OUTPUT / "run_metadata.json").as_posix(), metrics=metrics,
                  metric_source=dict(type="csv_row", path=(OUTPUT / "summary.csv").as_posix(),
                                     match={"group": "all12"}, columns={key: key for key in METRICS})))
    print((OUTPUT / "summary.csv").resolve())


if __name__ == "__main__":
    main()
