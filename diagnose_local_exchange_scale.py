"""Single frozen scene: suppress exchanged features in the pilot and full checkpoints."""
import os
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from e2_local_correspondence import LocalCorrespondenceE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, write_csv
from run_e2_local_correspondence import PILOT
from run_e2_local_correspondence_full import FULL
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


@torch.inference_mode()
def main():
    os.chdir(Path(__file__).resolve().parent)
    archive = Path("test_result/e2_local_correction_pilot_v3/val936/seed42")
    manifest = archive / "split_manifest.json"
    split = read_json(manifest)
    scene = split["splits"]["val"][0]
    pair = locate_scene(DATA, split, scene)
    optical, sar = read_pair(pair)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o, s = (to_tensor(x, device) for x in (optical, sar))
    output = Path("test_result/e2_local_exchange_scale_diagnosis_v1") / f"scene{scene}"
    if output.exists():
        raise FileExistsError(output)
    pilot_meta = read_json(Path("test_result/e2_local_correspondence_val936_v1/seed42/candidate/run_metadata.json"))
    full_meta = read_json(FULL.result / "run_metadata.json")
    group = f"QXS-LOCAL-EXCHANGE-SCALE-V1-SCENE{scene}"
    rows, entries, provenance = [], [], {}
    for size, exp, folder, ck_hash in (
        ("pilot", PILOT, Path("test_result/e2_local_correspondence_val936_v1/seed42/candidate"), pilot_meta["checkpoint_sha256"]),
        ("full", FULL, FULL.result/"candidate", full_meta["checkpoint_hashes"]["candidate"]),
    ):
        if ck_hash != sha256_file(exp.final):
            raise ValueError("Checkpoint changed")
        model = LocalCorrespondenceE2().to(device).eval()
        model.load_state_dict(torch.load(exp.final, map_location="cpu", weights_only=False)["model"])
        layer = model.detail_fusion
        source = folder/"gray"/f"{scene}.png"
        archived = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        provenance[size] = dict(checkpoint=exp.final.as_posix(), sha256=ck_hash, archive=source.as_posix(),
                                archive_sha256=sha256_file(source))
        for mode, disabled in (("on", ()), ("no_optical_to_sar", (layer.optical_value,)),
                               ("no_sar_to_optical", (layer.sar_value,)), ("off", (layer.optical_value, layer.sar_value))):
            with ExitStack() as stack:
                for projection in disabled:
                    handle = projection.register_forward_hook(lambda mod, args, y: torch.zeros_like(y))
                    stack.callback(handle.remove)
                image = model(o, s)[0,0].clamp(0,1).mul(255).round().byte().cpu().numpy()
            if mode == "on" and not np.array_equal(image, archived):
                raise ValueError(f"Original forward differs from archived {size} output")
            metrics = dict(zip(METRICS, map(float, FusionEvaluator.metric_vector(image, optical, sar))))
            rows.append(dict(budget=size, mode=mode, **metrics,
                             mae_from_on=float(np.abs(image.astype(float)-archived).mean())))
            print(size, mode, {k:round(metrics[k],6) for k in ("Qabf","Nabf","VIF")}, flush=True)
        values = {r["mode"]:r["Nabf"] for r in rows if r["budget"]==size}
        key = f"{size}_on_minus_off_nabf"
        entries.append(dict(id=f"{group}-{size}", comparable_group=f"{group}-{size}", task="engineering",
                            status="diagnostic_single_scene", method=f"{size} fixed local exchange on/off",
                            dataset="QXS-SAROPT", split="val", offset=0, sample_count=1, selected_scene_ids=[scene],
                            checkpoint=exp.final.as_posix(), checkpoint_sha256=ck_hash, split_manifest=manifest.as_posix(),
                            split_manifest_sha256=sha256_file(manifest), metric_profile=DEFAULT_PROFILE,
                            output_policy="clamp_then_uint8", result_dir=output.as_posix(),
                            metadata_path=(output/"run_metadata.json").as_posix(),metrics={key:values["on"]-values["off"]},
                            metric_source=dict(type="json_paths",path=(output/"summary.json").as_posix(),paths={key:[key]}),
                            notes="Single archived scene; zeroed value projection(s), no retraining."))
    output.mkdir(parents=True)
    write_csv(output/"metrics.csv", list(rows[0]), (list(r.values()) for r in rows))
    write_json(output/"run_metadata.json",dict(scene=scene, inputs=[sha256_file(x) for x in pair],
               checkpoints=provenance, script_sha256=sha256_file(Path(__file__)), device=str(device)))
    write_json(output/"summary.json", {k:v for e in entries for k,v in e["metrics"].items()})
    register_many(entries)


if __name__=="__main__":
    main()
