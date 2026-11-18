"""Frozen E2 equal-final-effect optical/SAR control.

Calibrate two separate outcomes: (1) fused Detail delta RMS; (2) float
image MAE in gray levels. For each scene and outcome, set one common target
at 90% of the smallest response among all 12 original-4% perturbations.
Only shrink inputs, never increase them. Do not optimize quality metrics.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from diagnose_e2_causal_interventions import (
    DIRECTIONS, METRICS, ROOT, SELECTED_METRICS, SIGNS, STAGES,
    average, evaluate_metrics, intervene_detail, projection_context,
    tensor_rms, transfer_values, uint8_output, write_csv)
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from diagnose_fcsr_decoder import gradients, source_specific_mask
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT
from run_qxs_reference_test1000 import DATA
from utils.dataset_os import evenly_spaced
from utils.fusion_metrics import DEFAULT_PROFILE

MODES=("detail_rms","image_mae")
OUTPUT=Path("test_result/e2_equal_final_effect_v1/val16_strength004_seed42")
PAIRS=(("optical","sar"),("optical","shuffled_optical"),("sar","shuffled_optical"))
KEY_METRICS=(*SELECTED_METRICS,"source_optical_transfer","source_sar_transfer")


def conditions():
    return [(stage,direction,sign) for stage in STAGES
            for direction in DIRECTIONS for sign in SIGNS]


def measure_effect(model,ctx,optical,base,baseline,stage,direction,sign,
                   input_rms,mode,full=False):
    new_detail=intervene_detail(model,ctx,stage,direction,sign,input_rms)
    detail_rms=tensor_rms(new_detail-ctx["detail"])
    if mode=="detail_rms" and not full:
        return detail_rms
    fused=model.reconstruct(optical,base,new_detail)
    image_mae=float((fused-baseline).abs().mean()*255)
    if full:
        return dict(fused=fused,detail_rms=detail_rms,image_mae=image_mae)
    return detail_rms if mode=="detail_rms" else image_mae


def nominal_measurements(model,ctx,optical,base,baseline,upper):
    values={}
    for stage,direction,sign in conditions():
        result=measure_effect(model,ctx,optical,base,baseline,
                              stage,direction,sign,upper,"detail_rms",full=True)
        pair=dict(detail_rms=result["detail_rms"],image_mae=result["image_mae"])
        if not all(math.isfinite(x) and x>1e-12 for x in pair.values()):
            raise ValueError("Zero or nonfinite nominal response")
        values[(stage,direction,sign)]=pair
    return values


def targets_from_nominal(nominal, fraction):
    if not 0<fraction<1 or set(nominal)!=set(conditions()):
        raise ValueError("Invalid nominal experiments or common target fraction")
    return {mode:min(x[mode] for x in nominal.values())*fraction for mode in MODES}


def calibrate(measure,target,upper,steps=18,tolerance=0.02):
    """Root bracket search independent of QAB/F, NAB/F and other metrics."""
    if not (math.isfinite(target) and math.isfinite(upper) and
            target>0 and upper>0 and 8<=steps<=32 and 0<tolerance<=0.10):
        raise ValueError("Invalid calibration bounds")
    hi=float(upper)
    observed=float(measure(hi))
    if not math.isfinite(observed) or observed<target:
        raise ValueError("Common target is unreachable without amplification")
    lo,lower_value=0.0,0.0
    high,upper_value=hi,observed
    best=(hi,observed)
    for _ in range(steps):
        # Safeguarded interpolation converges in 1-3 evaluations for near-linear
        # small perturbations; keeps a valid bracket for nonlinear responses.
        fraction=(target-lower_value)/max(upper_value-lower_value,1e-30)
        fraction=max(0.05,min(0.95,fraction))
        mid=lo+fraction*(high-lo)
        value=float(measure(mid))
        if not math.isfinite(value) or value<0:
            raise ValueError("Invalid calibration response")
        if abs(value-target)<abs(best[1]-target):
            best=(mid,value)
        if abs(value-target)/target<=tolerance:
            return mid,value,abs(value-target)/target
        if value<target:
            lo,lower_value=mid,value
        else:
            high,upper_value=mid,value
    amplitude,achieved=best
    error=abs(achieved-target)/target
    if error>tolerance:
        raise ValueError("Calibration relative error %.5f > %.5f"%(error,tolerance))
    return amplitude,achieved,error


def finite_difference(left,right,metric):
    a,b=left.get(metric),right.get(metric)
    if a is None or b is None:
        return None
    a,b=float(a),float(b)
    return a-b if math.isfinite(a) and math.isfinite(b) else None


def paired_results(rows,tolerance):
    by={}
    for row in rows:
        key=(row["scene"],row["match_mode"],row["stage"],
             row["direction"],row["sign"])
        if key in by:
            raise ValueError("Duplicate matched rows")
        by[key]=row
    contrasts=[]
    for scene in sorted({x["scene"] for x in rows}):
        for mode in MODES:
            for stage in STAGES:
                for sign in SIGNS:
                    for left,right in PAIRS:
                        a=by[(scene,mode,stage,left,sign)]
                        b=by[(scene,mode,stage,right,sign)]
                        if a["calibration_error"]>tolerance or b["calibration_error"]>tolerance:
                            raise ValueError("Unmatched source pair")
                        if abs(a["target_effect"]-b["target_effect"])>1e-10:
                            raise ValueError("Targets differ")
                        result=dict(scene=scene,match_mode=mode,
                                    stage=stage,sign=sign,
                                    comparison=left+"_minus_"+right,
                                    target_effect=a["target_effect"],
                                    realized_effect_difference=(
                                        a["realized_effect"]-b["realized_effect"]),
                                    detail_rms_difference=(
                                        a["realized_detail_delta_rms"]-
                                        b["realized_detail_delta_rms"]),
                                    image_mae_difference=(
                                        a["image_mae_8bit_float"]-b["image_mae_8bit_float"]))
                        for metric in KEY_METRICS:
                            result[metric+"_difference"]=finite_difference(a,b,metric)
                        contrasts.append(result)
    return contrasts


def summary_stats(values,lower_is_better=False):
    good=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(n=len(good),mean=average(good),
                median=float(np.median(good)) if good else None,
                improved=sum((x<0 if lower_is_better else x>0) for x in good))


def summarize(rows,contrasts):
    by={}
    baseline={(r["scene"],r["match_mode"]):r for r in rows
              if r["stage"]=="baseline"}
    for mode in MODES:
        for stage,direction,sign in conditions():
            subset=[r for r in rows if
                    (r["match_mode"],r["stage"],r["direction"],r["sign"])==
                    (mode,stage,direction,sign)]
            key=f"{mode}:{stage}:{direction}:{sign:+d}"
            by[key]=dict(
                count=len(subset),
                mean_input_fraction=average(r["input_fraction"] for r in subset),
                mean_image_mae=average(r["image_mae_8bit_float"] for r in subset),
                mean_final_detail_rms=average(
                    r["realized_detail_delta_rms"] for r in subset),
                mean_changed_pixel_fraction=average(
                    r["changed_u8_fraction"] for r in subset),
                max_match_error=max(r["calibration_error"] for r in subset),
                delta={metric:summary_stats(
                    [finite_difference(r,baseline[(r["scene"],mode)],metric)
                     for r in subset],metric=="Nabf") for metric in KEY_METRICS})
    pair_stats={}
    for mode in MODES:
        for stage in STAGES:
            for sign in SIGNS:
                for left,right in PAIRS:
                    name=left+"_minus_"+right
                    subset=[r for r in contrasts if (
                        r["match_mode"],r["stage"],r["sign"],r["comparison"]
                    )==(mode,stage,sign,name)]
                    pair_stats[f"{mode}:{stage}:{sign:+d}:{name}"]={
                        metric:summary_stats([r[metric+"_difference"] for r in subset],
                                              metric=="Nabf")
                        for metric in KEY_METRICS}
    return by,pair_stats


def one_scene(model,scene,manifest,device,args,image_dir=None):
    ou,su=read_pair(locate_scene(DATA,manifest,scene))
    o,s=to_tensor(ou,device),to_tensor(su,device)
    rows,original=[],[]
    with torch.inference_mode():
        enc=model.encode_pair(o,s)
        base,detail=model.fuse_features(*enc)
        ctx=projection_context(model,enc,base)
        torch.testing.assert_close(ctx["detail"],detail,atol=1e-6,rtol=0)
        y=model.reconstruct(o,base,detail)
        torch.testing.assert_close(y,model(o,s),atol=1e-6,rtol=0)
        gray0=uint8_output(y)
        metrics0=evaluate_metrics(gray0,ou,su)
        source_g={"optical":gradients(o),"sar":gradients(s)}
        valid=s>0
        masks={
            "optical":source_specific_mask(source_g["optical"],source_g["sar"],valid),
            "sar":source_specific_mask(source_g["sar"],source_g["optical"],valid)}
        t0=transfer_values(y,source_g,masks)
        input_rms=args.strength*math.hypot(
            tensor_rms(ctx["op"]),tensor_rms(ctx["sp"]))/math.sqrt(2)
        if input_rms<=1e-12:
            raise ValueError("Degenerate projection input")
        nominal=nominal_measurements(model,ctx,o,base,y,input_rms)
        targets=targets_from_nominal(nominal,args.target_fraction)
        if image_dir is not None:
            import cv2
            if not cv2.imwrite(str(image_dir/"baseline.png"),gray0):
                raise OSError("Failed to save baseline")
        for stage,direction,sign in conditions():
            item=nominal[(stage,direction,sign)]
            original.append(dict(scene=scene,stage=stage,direction=direction,sign=sign,
                                 original_input_rms=input_rms,
                                 nominal_detail_rms=item["detail_rms"],
                                 nominal_image_mae=item["image_mae"],
                                 target_detail_rms=targets["detail_rms"],
                                 target_image_mae=targets["image_mae"]))
        for mode in MODES:
            common=dict(scene=scene,match_mode=mode,
                        nominal_input_rms=input_rms,target_effect=targets[mode],
                        source_optical_transfer=t0["optical"],
                        source_sar_transfer=t0["sar"],
                        optical_candidates=int(masks["optical"].sum()),
                        sar_candidates=int(masks["sar"].sum()))
            rows.append(dict(**common,stage="baseline",direction="none",sign=0,
                applied_input_rms=0.,input_fraction=0.,realized_effect=0.,
                realized_detail_delta_rms=0.,image_mae_8bit_float=0.,
                calibration_error=0.,changed_u8_fraction=0.,**metrics0))
            for stage,direction,sign in conditions():
                def probe(amplitude):
                    return measure_effect(model,ctx,o,base,y,stage,direction,sign,
                                          amplitude,mode)
                amplitude,achieved,error=calibrate(
                    probe,targets[mode],input_rms,args.steps,args.tolerance)
                final=measure_effect(model,ctx,o,base,y,stage,direction,sign,
                                     amplitude,mode,full=True)
                actual=final["detail_rms"] if mode=="detail_rms" else final["image_mae"]
                error=abs(actual-targets[mode])/targets[mode]
                if error>args.tolerance:
                    raise AssertionError("Calibration changed on full evaluation")
                img=uint8_output(final["fused"])
                grad_transfer=transfer_values(final["fused"],source_g,masks)
                row=dict(scene=scene,match_mode=mode,stage=stage,
                    direction=direction,sign=sign,nominal_input_rms=input_rms,
                    target_effect=targets[mode],applied_input_rms=amplitude,
                    input_fraction=amplitude/input_rms,realized_effect=actual,
                    realized_detail_delta_rms=final["detail_rms"],
                    image_mae_8bit_float=final["image_mae"],
                    calibration_error=error,
                    changed_u8_fraction=float(np.mean(img!=gray0)),
                    source_optical_transfer=grad_transfer["optical"],
                    source_sar_transfer=grad_transfer["sar"],
                    optical_candidates=int(masks["optical"].sum()),
                    sar_candidates=int(masks["sar"].sum()),
                    **evaluate_metrics(img,ou,su))
                rows.append(row)
                if image_dir is not None:
                    import cv2
                    filename=f"{mode}_{stage}_{direction}_{'p' if sign>0 else 'm'}.png"
                    if not cv2.imwrite(str(image_dir/filename),img):
                        raise OSError("Failed to save matched image")
    return rows,original


def register(out,report):
    """Register as diagnostic, not in any formal fusion-comparison group."""
    location=(out.relative_to(ROOT)).as_posix()
    registry_path=ROOT/"EXPERIMENT_REGISTRY.json"
    old=registry_path.read_text(encoding="utf-8")
    doc=json.loads(old)
    identity="E2-EQUAL-FINAL-DIAG-"+hashlib.sha256(location.encode()).hexdigest()[:12].upper()
    if any(e["id"]==identity for e in doc["experiments"]):
        raise FileExistsError("Already registered: "+identity)
    entry=dict(id=identity,method="Frozen E2 equal final-effect source intervention",
        task="engineering",status="diagnostic_fixed_subset",
        comparable_group=identity,dataset="QXS-SAROPT",split="val",offset=0,
        sample_count=report["sample_count"],selected_scene_ids=report["scene_ids"],
        checkpoint=report["checkpoint"],
        checkpoint_sha256=report["checkpoint_sha256"],
        split_manifest=report["split_manifest"],
        split_manifest_sha256=report["split_manifest_sha256"],
        metric_profile=DEFAULT_PROFILE,output_policy="clamp_round_uint8",
        result_dir=location,metadata_path=location+"/run_metadata.json",
        metrics={"calibrated_conditions":float(report["calibrated_conditions"])},
        metric_source=dict(type="json_paths",path=location+"/summary.json",
                           paths={"calibrated_conditions":["calibrated_conditions"]}),
        notes="Frozen diagnostic. Matched output RMS/MAE. Not trained model ranking.")
    doc["experiments"].append(entry)
    doc["updated_at"]=datetime.now().strftime("%Y-%m-%d")
    registry_path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",
                             encoding="utf-8")
    try:
        subprocess.run([sys.executable,"maintain_experiment_index.py","--check",
                        "--render"],cwd=ROOT,check=True)
    except Exception:
        registry_path.write_text(old,encoding="utf-8")
        raise
    return identity


def run(args):
    os.chdir(ROOT)
    if not 1<=args.val_scenes<=64 or not 0<args.strength<=0.2:
        raise ValueError("val scenes 1..64 and strength (0,0.2]")
    if not 0<args.target_fraction<1 or not 0<args.tolerance<=0.1:
        raise ValueError("Invalid calibration settings")
    if not 0<=args.save_images<=args.val_scenes:
        raise ValueError("Invalid save count")
    out=(ROOT/(args.output_dir or OUTPUT)).resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    manifest=source_manifest()
    scenes=[str(x) for x in evenly_spaced(manifest["splits"]["val"],args.val_scenes)]
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model,checkpoint=frozen_checkpoint(device)
    all_rows,nominal_rows=[],[]
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".matched-e2-",dir=out.parent) as tmp:
        staged=Path(tmp)/"result"
        staged.mkdir()
        images=staged/"images"
        if args.save_images:images.mkdir()
        for i,scene in enumerate(scenes):
            folder=None
            if i<args.save_images:
                folder=images/scene
                folder.mkdir()
            sample,raw=one_scene(model,scene,manifest,device,args,folder)
            all_rows.extend(sample)
            nominal_rows.extend(raw)
            print(f"equal-output matching {i+1}/{len(scenes)} {scene}",flush=True)
        contrasts=paired_results(all_rows,args.tolerance)
        conditions_summary,pair_summary=summarize(all_rows,contrasts)
        max_error=max(r["calibration_error"] for r in all_rows)
        report=dict(status="frozen_equal_final_effect_no_training",
          dataset="QXS-SAROPT",split="matched_pilot_val64",
          sample_count=len(scenes),scene_ids=scenes,
          checkpoint=checkpoint.as_posix(),
          checkpoint_sha256=sha256_file(checkpoint),
          split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
          split_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
          script_sha256=sha256_file(Path(__file__)),seed=42,device=str(device),
          nominal_strength=args.strength,target_fraction=args.target_fraction,
          calibration_steps=args.steps,relative_tolerance=args.tolerance,
          max_matching_error=max_error,
          calibrated_conditions=len(all_rows)-2*len(scenes),
          modes=MODES,metric_profile=DEFAULT_PROFILE,
          quantization="clamp_round_uint8",
          metric_names=METRICS,condition_summary=conditions_summary,
          matched_contrast_summary=pair_summary,
          caveats=[
            "Calibrate by effect size only, never by image-quality metric.",
            "Per-scene shared target spans all stages, signs and sources within one mode.",
            "Detail-RMS and image-MAE modes are distinct and not mutually equivalent.",
            "All interventions shrink nominal 4% input; frozen paths may still be off-distribution.",
            "SAR edge candidates may be speckle; source-gradient scores are not structure labels.",
            "No architecture trained, and observed local improvements do not imply generalizable gains."])
        write_csv(staged/"matched_all_metrics.csv",all_rows)
        write_csv(staged/"matched_contrasts.csv",contrasts)
        write_csv(staged/"nominal_effects.csv",nominal_rows)
        (staged/"summary.json").write_text(json.dumps(report,ensure_ascii=False,
                                         indent=2,allow_nan=False),encoding="utf-8")
        (staged/"run_metadata.json").write_text(json.dumps({
            k:report[k] for k in ("dataset","split","sample_count","scene_ids",
                "checkpoint","checkpoint_sha256","split_manifest",
                "split_manifest_sha256","script_sha256","nominal_strength",
                "target_fraction","relative_tolerance","max_matching_error",
                "metric_profile","quantization")},ensure_ascii=False,indent=2),
                encoding="utf-8")
        staged.rename(out)
    reg_id=register(out,report)
    print(json.dumps({"output":str(out),"registration":reg_id,
                      "samples":len(scenes),"max_matching_error":max_error},
                     ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-scenes",type=int,default=16)
    p.add_argument("--strength",type=float,default=0.04)
    p.add_argument("--target-fraction",type=float,default=0.90)
    p.add_argument("--tolerance",type=float,default=0.02)
    p.add_argument("--steps",type=int,default=18)
    p.add_argument("--save-images",type=int,default=4)
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--output-dir",type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()
