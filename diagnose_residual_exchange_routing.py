"""Frozen residual exchange: separate the immediate write and neighbor weighting effects."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import e2_local_correspondence as matching
from diagnose_e2_information_path import read_pair, to_tensor
from diagnose_source_interval_tolerance import contribution_map, read_scores, write_csv
from inspect_residual_exchange_artifacts import source_regions
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence_full import RESIDUAL
from run_e2_matched_no_pac import register
from run_e2_scratch_twostage import BASE_ANCHOR, ORIGINAL_LOSS_PILOT
from run_qxs_reference_test1000 import DATA, read_image, read_json, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


def equalize(w):
    support = w > 0
    return support * (w.sum(1, keepdim=True) / support.sum(1, keepdim=True).clamp_min(1))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("first", "val64"), default="first", nargs="?")
    mode = parser.parse_args().mode
    os.chdir(Path(__file__).resolve().parent)
    output = Path("test_result/e2_residual_exchange_routing_v1") / mode
    if output.exists():
        raise FileExistsError(output)
    meta = read_json(RESIDUAL.result / "run_metadata.json")
    split = manifest_pairs(DATA, RESIDUAL.manifest)[0]["val"]
    pairs = {a.stem:(a,b) for a,b in split}
    ids = list(pairs)[:1] if mode == "first" else read_json(ORIGINAL_LOSS_PILOT.manifest)["splits"]["val"]
    if (list(pairs) != meta["scene_ids"] or meta["metric_profile"] != DEFAULT_PROFILE
            or meta["split_manifest_sha256"] != sha256_file(RESIDUAL.manifest)
            or meta["checkpoint_hashes"] != {k:sha256_file(exp.final) for k,exp in
                                             (("candidate", RESIDUAL), ("e2", BASE_ANCHOR))}):
        raise ValueError("Validation provenance differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RESIDUAL.build_model().to(device).eval()
    state = torch.load(RESIDUAL.final, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    scores = {k:read_scores(RESIDUAL.result/k/"metrics.csv") for k in ("candidate","e2")}
    original = matching.reciprocal_weights
    def uniform(*args):
        return tuple(equalize(w) for w in original(*args))
    def rendered(o, s):
        return model(o,s)[0,0].clamp(0,1).mul(255).round().byte().cpu().numpy()
    rows, regions, hashes = [], [], {}
    for scene in ids:
        optical, sar = read_pair(pairs[scene])
        o, s = (to_tensor(x,device) for x in (optical,sar))
        path = RESIDUAL.result/"candidate/gray"/f"{scene}.png"
        base = BASE_ANCHOR.result/"gray"/f"{scene}.png"
        hashes[scene] = [sha256_file(p) for p in (*pairs[scene], base, path)]
        on = rendered(o,s)
        if not np.array_equal(on, read_image(path,cv2.IMREAD_UNCHANGED)):
            raise ValueError(f"Archived residual output differs: {scene}")
        hook = model.detail_fusion.residual_projection.register_forward_hook(lambda mod,args,y: torch.zeros_like(y))
        try:
            off = rendered(o,s)
        finally:
            hook.remove()
        with patch.object(matching, "reciprocal_weights", uniform):
            flat = rendered(o,s)
        edges = tuple(FusionEvaluator._kumar_edges(im) for im in (optical,sar))
        masks = source_regions(optical,sar,edges[0][0],edges[1][0])
        maps = {}
        for name, image in (("e2",read_image(base,cv2.IMREAD_UNCHANGED)),("off",off),("uniform",flat),("on",on)):
            contribution = contribution_map(image,*edges)[0]
            maps[name] = contribution
            nabf = float(contribution.sum())
            if name in ("e2","on"):
                np.testing.assert_allclose(nabf,scores["e2" if name=="e2" else "candidate"][scene],atol=1e-9,rtol=0)
            rows.append(dict(scene=scene, mode=name, Nabf=nabf, Qabf=FusionEvaluator.Qabf(image,optical,sar),
                             VIF=FusionEvaluator.VIFF(image.astype(float),optical.astype(float),sar.astype(float)),
                             output_mae_from_on=float(np.abs(image.astype(float)-on).mean())))
        for region,mask in masks.items():
            regions.append(dict(scene=scene,region=region,area=float(mask.mean()),
                                on_minus_e2=float((maps["on"]-maps["e2"])[mask].sum()),
                                on_minus_off=float((maps["on"]-maps["off"])[mask].sum()),
                                on_minus_uniform=float((maps["on"]-maps["uniform"])[mask].sum())))
    means = {m:float(np.mean([r["Nabf"] for r in rows if r["mode"]==m])) for m in ("e2","on","off","uniform")}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"per_scene.csv",rows)
    write_csv(output/"regions.csv",regions)
    write_json(output/"summary.json",means)
    write_json(output/"run_metadata.json",dict(scene_ids=ids, inputs=hashes, checkpoint_hashes=meta["checkpoint_hashes"],
               device=str(device), source_sha256=sha256_file(Path(__file__)),
               definition="uniform over original nonzero neighbors with original per-query weight sum"))
    key="QXS-RESIDUAL-EXCHANGE-ROUTING-V1-"+mode.upper()
    register(dict(id=key, comparable_group=key, task="engineering",
                  status="diagnostic_single_scene" if len(ids)==1 else "diagnostic_fixed_subset",
                  method="Frozen residual exchange write and support-matched uniform control, NAB/F",
                  dataset="QXS-SAROPT",split="val",offset=0,sample_count=len(ids),selected_scene_ids=ids,
                  checkpoint=RESIDUAL.final.as_posix(),checkpoint_sha256=meta["checkpoint_hashes"]["candidate"],
                  parent_checkpoint=BASE_ANCHOR.final.as_posix(),parent_checkpoint_sha256=meta["checkpoint_hashes"]["e2"],
                  split_manifest=RESIDUAL.manifest.as_posix(),split_manifest_sha256=meta["split_manifest_sha256"],
                  metric_profile=DEFAULT_PROFILE,output_policy="clamp_then_uint8",metrics=means,result_dir=output.as_posix(),
                  metadata_path=(output/"run_metadata.json").as_posix(),metric_source=dict(
                      type="json_paths",path=(output/"summary.json").as_posix(),paths={k:[k] for k in means})))
    print(means)


if __name__ == "__main__":
    main()
