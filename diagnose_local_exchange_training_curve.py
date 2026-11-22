"""Compare saved full-training checkpoints on the first archived 936-set scene."""
import csv
import os
from pathlib import Path

import numpy as np
import torch
import cv2

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import write_csv
from run_e2_local_correspondence_full import FULL
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import BASE_ANCHOR, RECON_EPOCHS, TOTAL_EPOCHS
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


@torch.inference_mode()
def main():
    os.chdir(Path(__file__).resolve().parent)
    parent = Path("test_result/e2_local_correction_pilot_v3/val936/seed42")
    split = read_json(parent / "split_manifest.json")
    scene = split["splits"]["val"][0]
    paired = read_json(FULL.result / "run_metadata.json")
    out = Path("test_result/e2_local_exchange_training_curve_v1") / f"scene{scene}"
    if out.exists():
        raise FileExistsError(out)
    pair = locate_scene(DATA, split, scene)
    input_hashes = [sha256_file(p) for p in pair]
    if (input_hashes != read_json(parent / "scenes" / f"{scene}.json")["inputs"]
            or sha256_file(FULL.manifest) != paired["split_manifest_sha256"]):
        raise ValueError("Archive provenance mismatch")
    optical, sar = read_pair(pair)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o, s = (to_tensor(image, device) for image in (optical, sar))
    names = ("Qabf", "Nabf", "VIFF")
    rows, hashes, endpoints = {}, {}, {}
    for key, run in (("e2", BASE_ANCHOR), ("candidate", FULL)):
        model = run.build_model().to(device).eval()
        hashes[key] = {}
        for epoch in range(RECON_EPOCHS + 1, TOTAL_EPOCHS + 1):
            checkpoint = run.run_dir / f"epoch_{epoch:03d}.pth"
            digest = sha256_file(checkpoint)
            hashes[key][checkpoint.as_posix()] = digest
            st = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (st["epoch"], st["stage"], st["variant"]) != (epoch-1, "fusion", run.variant):
                raise ValueError(f"Unexpected checkpoint: {checkpoint}")
            model.load_state_dict(st["model"], strict=True)
            image = model(o,s)[0,0].clamp(0,1).mul(255).round().byte().cpu().numpy().astype(float)
            scores = [getattr(FusionEvaluator, name)(image, optical.astype(float), sar.astype(float)) for name in names]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite scene metrics")
            rows[key,epoch] = scores
            if epoch == TOTAL_EPOCHS:
                if digest != paired["checkpoint_hashes"][key]:
                    raise ValueError("Wrong final weights")
                with (FULL.result / key / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
                    ref = next(r for r in csv.DictReader(f) if r["image"] == scene)
                archived_path = (BASE_ANCHOR.result if key == "e2" else FULL.result/"candidate")/"gray"/f"{scene}.png"
                archived = cv2.imread(str(archived_path), cv2.IMREAD_UNCHANGED).astype(float)
                archived_scores = [getattr(FusionEvaluator,n)(archived,optical.astype(float),sar.astype(float)) for n in names]
                np.testing.assert_allclose(archived_scores, [float(ref[n]) for n in ("Qabf","Nabf","VIF")], rtol=0, atol=1e-7)
                error = np.abs(image-archived)
                endpoints[key] = dict(different_pixels=int(np.count_nonzero(error)), maximum_gray_difference=float(error.max()),
                                      archived_metrics=archived_scores, rerun_metrics=scores)
    out.mkdir(parents=True)
    table, better_n, joint = [], [], []
    for epoch in range(RECON_EPOCHS + 1, TOTAL_EPOCHS + 1):
        a,b = rows["e2",epoch], rows["candidate",epoch]
        diff = np.subtract(b,a)
        if diff[1] < 0:
            better_n.append(epoch)
            if diff[0] > 0 and diff[2] > 0:
                joint.append(epoch)
        table.append((epoch, *a, *b, *diff))
    fields = ["epoch"] + [key+"_"+n for key in ("e2","exchange","delta") for n in ("Qabf","Nabf","VIF")]
    write_csv(out/"trajectory.csv",fields,table)
    metrics = dict(fusion_epochs=len(table), lower_nabf_epochs=len(better_n), jointly_better_epochs=len(joint))
    write_json(out/"summary.json",dict(**metrics, lower_nabf_at=better_n, jointly_better_at=joint))
    write_json(out/"run_metadata.json",dict(scene=scene, epoch_checkpoint_sha256=hashes, input_hashes=input_hashes,
               endpoint_checks=endpoints, expected_final_source=(FULL.result/"run_metadata.json").as_posix(), script_sha256=sha256_file(Path(__file__)),
               metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", device=str(device)))
    group = "QXS-FULL-LOCAL-EXCHANGE-TRAINING-CURVE-V1-SCENE"+scene
    register_many([dict(id=group, comparable_group=group, task="engineering", status="diagnostic_single_scene",
        method="Full E2 vs local exchange frozen fusion-epoch trajectory", dataset="QXS-SAROPT",
        split="val", offset=paired["scene_ids"].index(scene), sample_count=1, selected_scene_ids=[scene],
        checkpoint=FULL.final.as_posix(), checkpoint_sha256=sha256_file(FULL.final),
        split_manifest=FULL.manifest.as_posix(), split_manifest_sha256=paired["split_manifest_sha256"],
        metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", result_dir=out.as_posix(),
        metadata_path=(out/"run_metadata.json").as_posix(), metrics=metrics,
        metric_source=dict(type="json_paths", path=(out/"summary.json").as_posix(), paths={k:[k] for k in metrics}),
        notes="One preselected scene; archived PNG metrics verified and endpoint pixel differences recorded.")])
    print(metrics)


if __name__ == "__main__":
    main()
