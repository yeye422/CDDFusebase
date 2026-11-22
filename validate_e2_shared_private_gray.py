"""Resume full B validation from per-scene gray PNGs and reference-profile scores."""
import argparse
import os
from pathlib import Path

import cv2
from filelock import FileLock
import numpy as np
import torch

from diagnose_e2_information_path import read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_cross_scale import check_training_settings, seal_result, training_lock
from run_e2_matched_no_pac import register_many
from run_e2_shared_private import FULL, paired_report, preflight
from run_e2_source_interval_val import BASE_ANCHOR, RCCR, read_metric_rows
from run_e2_v3_val936 import bind
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


OUTPUT = FULL.result.parent / "seed42_gray_v1"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "val"))
    action = parser.parse_args().action
    os.chdir(Path(__file__).resolve().parent)
    manifest, _, expected = preflight(FULL)
    lock = training_lock(FULL, expected)
    check_training_settings(expected["config"], read_json(BASE_ANCHOR.run_dir/"config.json"), resolved=True, run=FULL.run_dir)
    ids = manifest["splits"]["val"]
    pairs = manifest_pairs(DATA, FULL.manifest)[0]["val"]
    if [a.stem for a, b in pairs] != ids:
        raise ValueError("Validation scene order differs")
    base_table = RCCR.result/"e2/metrics.csv"
    baseline = read_metric_rows(base_table, ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group = "QXS-E2-SHARED-PRIVATE-PATCH-FULL-V1-VAL1000-GRAY-REFERENCE-FMI"
    meta = dict(scene_ids=ids, split="val", offset=0, metric_profile=DEFAULT_PROFILE,
                output_policy="clamp_then_uint8", stored_images="candidate_gray", split_manifest=FULL.manifest.as_posix(),
                split_manifest_sha256=sha256_file(FULL.manifest), training_lock_sha256=sha256_file(lock),
                baseline_metrics_sha256=sha256_file(base_table), baseline_metrics=base_table.as_posix(),
                checkpoint_hashes={k:sha256_file(exp.final) for k,exp in (("e2",BASE_ANCHOR),("candidate",FULL))},
                device=str(device), torch=str(torch.__version__), script_sha256=sha256_file(Path(__file__)))
    work = OUTPUT.with_suffix(".work")
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    with FileLock(str(OUTPUT.with_suffix(".lock")), timeout=0):
        if OUTPUT.exists():
            from run_e2_cross_scale import verify_result
            verify_result(OUTPUT, meta)
        else:
            work.mkdir(exist_ok=True)
            bind(work, "run_metadata.json", meta)
            gray_dir = work / "candidate/gray"
            gray_dir.mkdir(parents=True, exist_ok=True)
            (work/"e2").mkdir(exist_ok=True)
            (work/"scenes").mkdir(exist_ok=True)
            model = FULL.build_model().to(device).eval()
            state = torch.load(FULL.final, map_location="cpu", weights_only=False)
            if (state["variant"], state["epoch"], state["stage"]) != (FULL.variant, 15, "fusion"):
                raise ValueError("Expected the final B checkpoint")
            model.load_state_dict(state["model"])
            rows = []
            selected = [ids.index("4880")] if action == "smoke" else range(len(ids))
            for i in selected:
                scene, pair = ids[i], pairs[i]
                record = work/"scenes"/f"{scene}.json"
                target = gray_dir/f"{scene}.png"
                inputs = [sha256_file(path) for path in pair]
                if record.exists():
                    saved = read_json(record)
                    if saved["inputs"] != inputs or saved["image_sha256"] != sha256_file(target):
                        raise ValueError(f"Cached data changed: {scene}")
                else:
                    optical, sar = read_pair(pair)
                    out = model(to_tensor(optical,device), to_tensor(sar,device))
                    if not torch.isfinite(out).all():
                        raise ValueError(f"Nonfinite model output: {scene}")
                    gray = out[0,0].clamp(0,1).mul(255).round().byte().cpu().numpy()
                    if not cv2.imwrite(str(target), gray):
                        raise OSError(target)
                    scores = FusionEvaluator.metric_vector(gray,optical,sar)
                    saved = dict(inputs=inputs, image_sha256=sha256_file(target),
                                 scores=[float(x) if np.isfinite(x) else None for x in scores])
                    pending = record.with_suffix(".pending")
                    write_json(pending, saved)
                    pending.replace(record)
                rows.append((np.asarray(saved["scores"],dtype=float), (), ()))
                if action=="smoke" or (i+1)%25==0:
                    print(f"Verified {scene}" if action=="smoke" else f"val {i+1}/{len(ids)}", flush=True)
            if action=="smoke":
                return
            verify_images(gray_dir, ids)
            means = paired_report(work, ids, dict(e2=baseline,candidate=rows), [("e2","candidate")],
                                  experiment=FULL, baseline_gray=BASE_ANCHOR.result/"gray",
                                  diagnostic_fields=(), source_metrics=())
            write_json(work/"registry_metrics.json", means)
            seal_result(work)
            work.rename(OUTPUT)
        means = read_json(OUTPUT/"registry_metrics.json")
        entries = []
        for key, exp in (("e2", BASE_ANCHOR), ("candidate", FULL)):
            folder, metrics = OUTPUT/key, means[key]
            entries.append(dict(id=f"{group}-{key}",comparable_group=group,task="fusion",status="development_validation",
                                method=exp.method,dataset="QXS-SAROPT",split="val",offset=0,sample_count=len(ids),
                                split_manifest=FULL.manifest.as_posix(),split_manifest_sha256=meta["split_manifest_sha256"],
                                checkpoint=exp.final.as_posix(),checkpoint_sha256=meta["checkpoint_hashes"][key],
                                result_dir=folder.as_posix(),metadata_path=(OUTPUT/"run_metadata.json").as_posix(),
                                metric_profile=DEFAULT_PROFILE,output_policy=meta["output_policy"],metrics=metrics,
                                metric_source=dict(type="csv_row",path=(folder/"metrics.csv").as_posix(),
                                                   match={"image":"mean"},columns={k:k for k in metrics})))
        register_many(entries)
        print(OUTPUT/"paired_comparison.csv")


if __name__ == "__main__":
    main()
