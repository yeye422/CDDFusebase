"""Partition the archived residual model's NAB/F difference without model inference."""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from diagnose_source_interval_tolerance import contribution_map, read_scores, write_csv
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence_full import RESIDUAL
from run_e2_matched_no_pac import register
from run_e2_scratch_twostage import BASE_ANCHOR, ORIGINAL_LOSS_PILOT
from run_qxs_reference_test1000 import DATA, read_image, read_json, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


def source_regions(optical, sar, ge_o, ge_s):
    # Structure tensor coherence distinguishes locally consistent orientation
    # from multi-directional texture; it is not a physical speckle label.
    z = sar.astype(float)
    gx, gy = (cv2.Sobel(z, cv2.CV_64F, dx, dy, ksize=3) for dx, dy in ((1, 0), (0, 1)))
    xx, yy, xy = (cv2.blur(v, (7, 7)) for v in (gx * gx, gy * gy, gx * gy))
    coh = np.hypot(xx - yy, 2 * xy) / (xx + yy + 1e-12)
    valid = cv2.erode((sar > 0).astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    strong = np.maximum(ge_o, ge_s) >= 2  # cutoff used by the actual Kumar metric
    radar = valid & strong & (ge_s > ge_o)
    return dict(outside=~valid, weak=valid & ~strong, optical_dominant=valid & strong & (ge_o >= ge_s),
                sar_directional=radar & (coh >= .5), sar_mixed=radar & (coh < .5))


def analyze(pair, paths, scores):
    scene = pair[0].stem
    optical = cv2.cvtColor(read_image(pair[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sar = read_image(pair[1], cv2.IMREAD_GRAYSCALE)
    edges = [FusionEvaluator._kumar_edges(im) for im in (optical, sar)]
    regions = source_regions(optical, sar, edges[0][0], edges[1][0])
    inputs = list(pair)
    maps, over = [], []
    for key, folder in paths.items():
        path = folder / f"{scene}.png"
        image = read_image(path, cv2.IMREAD_UNCHANGED)
        if image.dtype != np.uint8 or image.shape != sar.shape:
            raise ValueError(f"Invalid image: {path}")
        contrib, _, _, mask = contribution_map(image, *edges)
        np.testing.assert_allclose(contrib.sum(), scores[key][scene], atol=1e-9, rtol=0)
        maps.append(contrib)
        over.append(mask)
        inputs.append(path)
    diff = maps[1] - maps[0]
    partitions = dict(source=regions, overshoot=dict(new=over[1] & ~over[0], shared=over[1] & over[0],
                                                    removed=over[0] & ~over[1], neither=~(over[0] | over[1])))
    rows = []
    for kind, masks in partitions.items():
        if not np.all(np.sum(list(masks.values()), axis=0) == 1):
            raise ValueError("Invalid partition")
        for region, mask in masks.items():
            d = diff[mask]
            rows.append(dict(scene=scene, partition=kind, region=region, area=float(mask.mean()),
                             e2=float(maps[0][mask].sum()), residual=float(maps[1][mask].sum()),
                             delta=float(d.sum()), increases=float(d.clip(min=0).sum()),
                             decreases=float(d.clip(max=0).sum())))
    return rows, {p.as_posix(): sha256_file(p) for p in inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("first", "val64"), default="first", nargs="?")
    mode = parser.parse_args().mode
    os.chdir(Path(__file__).resolve().parent)
    meta = read_json(RESIDUAL.result / "run_metadata.json")
    pairs, _ = manifest_pairs(DATA, RESIDUAL.manifest)
    lookup = {p[0].stem:p for p in pairs["val"]}
    if (list(lookup) != meta["scene_ids"] or meta["metric_profile"] != DEFAULT_PROFILE
            or meta["output_policy"] != "clamp_then_uint8" or meta["split_manifest_sha256"] != sha256_file(RESIDUAL.manifest)
            or meta["checkpoint_hashes"] != {k:sha256_file(exp.final) for k,exp in (("e2",BASE_ANCHOR),("candidate",RESIDUAL))}):
        raise ValueError("Unexpected validation archive")
    ids = meta["scene_ids"][:1] if mode=="first" else read_json(ORIGINAL_LOSS_PILOT.manifest)["splits"]["val"]
    paths = dict(e2=BASE_ANCHOR.result/"gray", candidate=RESIDUAL.result/"candidate/gray")
    scores = {k:read_scores(RESIDUAL.result/k/"metrics.csv") for k in paths}
    out = Path("test_result/e2_residual_exchange_artifact_regions_v1") / mode
    if out.exists():
        raise FileExistsError(out)
    rows, hashes = [], {}
    for scene in ids:
        items, inputs = analyze(lookup[scene], paths, scores)
        rows.extend(items)
        hashes.update(inputs)
    summary = []
    for kind, region in dict.fromkeys((r["partition"],r["region"]) for r in rows):
        part = [r for r in rows if (r["partition"],r["region"]) == (kind,region)]
        summary.append(dict(partition=kind, region=region, n=len(part), **{
            key:float(np.mean([r[key] for r in part])) for key in ("area","e2","residual","delta","increases","decreases")}))
    values = {k:float(np.mean([scores[k][i] for i in ids])) for k in paths}
    values["delta"] = values["candidate"] - values["e2"]
    for kind in ("source","overshoot"):
        np.testing.assert_allclose(sum(r["delta"] for r in summary if r["partition"]==kind),values["delta"],atol=1e-9)
    out.mkdir(parents=True)
    write_csv(out/"per_scene.csv",rows)
    write_csv(out/"regions.csv",summary)
    write_json(out/"summary.json",values)
    write_json(out/"run_metadata.json",dict(scene_ids=ids,inputs=hashes,checkpoint_hashes=meta["checkpoint_hashes"],
               split_manifest_sha256=meta["split_manifest_sha256"],metric_profile=DEFAULT_PROFILE,
               script_sha256=sha256_file(Path(__file__)),archive_sha256=sha256_file(RESIDUAL.result/"run_metadata.json"),
               gradient_cutoff=2,coherence_window=7,coherence_cutoff=.5))
    group="QXS-RESIDUAL-EXCHANGE-NABF-REGIONS-V1-"+mode.upper()
    register(dict(id=group, comparable_group=group, task="engineering",status="diagnostic_single_scene" if len(ids)==1 else "diagnostic_fixed_subset",
             method="Regional decomposition of archived residual-exchange NAB/F",dataset="QXS-SAROPT",split="val",offset=0,
             sample_count=len(ids),selected_scene_ids=ids,checkpoint=RESIDUAL.final.as_posix(),checkpoint_sha256=meta["checkpoint_hashes"]["candidate"],
             parent_checkpoint=BASE_ANCHOR.final.as_posix(),parent_checkpoint_sha256=meta["checkpoint_hashes"]["e2"],
             split_manifest=RESIDUAL.manifest.as_posix(),split_manifest_sha256=meta["split_manifest_sha256"],metric_profile=DEFAULT_PROFILE,
             output_policy="clamp_then_uint8", result_dir=out.as_posix(),metadata_path=(out/"run_metadata.json").as_posix(),metrics=values,
             metric_source=dict(type="json_paths",path=(out/"summary.json").as_posix(),paths={k:[k] for k in values})))
    print(values, out/"regions.csv")


if __name__ == "__main__":
    main()
