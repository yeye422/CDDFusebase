"""Matched, frozen causal interventions at two E2 Detail Fusion boundaries.

PRE-PROJECTION-AGGREGATION ("early"): perturb post-GELU optical/SAR
projection and recompute scale branches plus final Detail output.
PRE-OUTPUT ("late"): perturb the identical direct source slot of the
aggregate just before detail_fusion.output, keeping scale branches fixed.

Each source/control perturbation has the same absolute projected-channel RMS.
Positive and negative interventions, SAR and deterministic shuffled-optical
null controls share images, fused Base, checkpoint and Decoder.

No weights are trained/changed. This diagnoses *local response* and tradeoffs;
it does not establish that an intervention is a useful trainable architecture.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from diagnose_fcsr_decoder import gradients, signed_transfer, source_specific_mask
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT
from run_qxs_reference_test1000 import DATA
from utils.dataset_os import evenly_spaced
from utils.fusion_metrics import FusionEvaluator, DEFAULT_PROFILE

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("test_result/e2_detail_causal_interventions_v1/val16_strength004_seed42")
STAGES = ("early", "late")
DIRECTIONS = ("optical", "sar", "shuffled_optical")
SIGNS = (-1, 1)
SELECTED_METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF", "SCD")
METRICS = tuple(FusionEvaluator.METRIC_NAMES)
EPS = 1e-12


def tensor_rms(x):
    return float(x.detach().float().square().mean().sqrt())


def projection_context(model, encoded, bf):
    """Verify cached internal node is exactly the archived forward path."""
    _, do, _, ds = encoded
    module = model.detail_fusion
    op, sp, bp = module._project_inputs(do, ds, bf)
    scales = [block(torch.cat((op, sp), 1)) for block in module.scale_branches]
    agg = torch.cat((op, sp, *scales, bp), 1)
    detail = module.output(agg)
    actual = module(do, ds, bf)
    torch.testing.assert_close(detail, actual, atol=1e-6, rtol=0)
    return dict(op=op, sp=sp, bp=bp, scales=scales, agg=agg,
                detail=detail)


def shuffled_direction(x, seed=42):
    """Deterministic channel permutation + nonzero spatial roll.

    Preserves global RMS and channel-wise spatial content, disrupts alignment
    and channel identity. This is a null intervention, NOT independent noise.
    """
    c, h, w = x.shape[-3:]
    if c < 3 or h < 8 or w < 8:
        raise ValueError("Need at least three channels and 8x8 feature grid")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(c, generator=gen).to(x.device)
    return torch.roll(x[:, perm], shifts=(max(1,h//3), max(1,w//4)), dims=(-2,-1))


def direction_tensors(ctx):
    return {
        "optical": ctx["op"],
        "sar": ctx["sp"],
        "shuffled_optical": shuffled_direction(ctx["op"]),
    }


def normalized_direction(source, target_rms):
    actual = tensor_rms(source)
    if actual <= 1e-10:
        raise ValueError("Feature response direction is degenerate")
    delta = source * (target_rms / actual)
    if not math.isclose(tensor_rms(delta),target_rms,rel_tol=0.001):
        raise AssertionError("Source direction RMS matching failed")
    return delta


def intervene_detail(model, ctx, stage, direction, sign, target_rms):
    """Same absolute RMS at the targeted 32-channel optical/SAR node.

    Early changes the source projection and lets scale branches recompute.
    Late changes ONLY the direct source slot after concatenation; scale branch
    outputs stay fixed. Thus early/late differences cannot be read as a single
    neuron's causal contribution without accounting for changed pathways.
    """
    if stage not in STAGES or direction not in DIRECTIONS or sign not in SIGNS:
        raise ValueError("Unknown intervention definition")
    delta = normalized_direction(direction_tensors(ctx)[direction],target_rms)
    op, sp, bp = ctx["op"], ctx["sp"], ctx["bp"]
    direct_optical = direction != "sar"
    if stage == "early":
        if direct_optical:
            op = op + sign * delta
        else:
            sp = sp + sign * delta
        joint = torch.cat((op,sp),1)
        scales = [block(joint) for block in model.detail_fusion.scale_branches]
        agg = torch.cat((op,sp,*scales,bp),1)
    else:
        agg = ctx["agg"].clone()
        offset = 0 if direct_optical else op.shape[1]
        agg[:,offset:offset+delta.shape[1]] += sign*delta
    detail = model.detail_fusion.output(agg)
    return detail


def uint8_output(image):
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("Expected B1HW gray output")
    if not torch.isfinite(image).all():
        raise FloatingPointError("Nonfinite fused output")
    return image[0,0].clamp(0,1).mul(255).round().byte().cpu().numpy()


def safe_float(value):
    v = float(value)
    return v if math.isfinite(v) else None


def evaluate_metrics(gray, optical_u8, sar_u8):
    values = FusionEvaluator.metric_vector(gray,optical_u8,sar_u8)
    if len(values) != len(METRICS):
        raise ValueError("Metric-vector length changed")
    return dict(zip(METRICS,(safe_float(x) for x in values)))


def transfer_values(output, gradients_source, masks):
    gf = gradients(output)
    return {
        name: signed_transfer(gradients_source[name],gf,masks[name])
        for name in ("optical","sar")
    }


def evaluate_scene(model, scene, manifest, device, strength, save_gray=None):
    ou,su = read_pair(locate_scene(DATA,manifest,scene))
    o,s = to_tensor(ou,device),to_tensor(su,device)
    with torch.inference_mode():
        encoded = model.encode_pair(o,s)
        bo,do,bs,ds = encoded
        bf,df = model.fuse_features(*encoded)
        ctx = projection_context(model,encoded,bf)
        torch.testing.assert_close(ctx["detail"],df,atol=1e-6,rtol=0)
        baseline = model.reconstruct(o,bf,df)
        torch.testing.assert_close(baseline,model(o,s),atol=1e-6,rtol=0)
        gray0 = uint8_output(baseline)
        original_metrics = evaluate_metrics(gray0,ou,su)
        source_gradient = {"optical":gradients(o),"sar":gradients(s)}
        valid = s>0
        masks = {
            "optical":source_specific_mask(source_gradient["optical"],
                                           source_gradient["sar"],valid),
            "sar":source_specific_mask(source_gradient["sar"],
                                       source_gradient["optical"],valid),
        }
        transfer0=transfer_values(baseline,source_gradient,masks)
        if save_gray:
            import cv2
            if not cv2.imwrite(str(save_gray/"baseline.png"),gray0):
                raise OSError(f"Cannot save baseline of {scene}")
        anchor = (tensor_rms(ctx["op"])**2+
                  tensor_rms(ctx["sp"])**2)**0.5/2**0.5
        target_rms=strength*anchor
        if target_rms<1e-10:
            raise ValueError("Degenerate projection scale")
        rows=[dict(scene=scene,stage="baseline",direction="none",sign=0,
                   strength=strength,target_projected_rms=target_rms,
                   realized_detail_delta_rms=0.0,
                   image_mae_8bit_unquantized=0.0,
                   quantized_changed_pixel_fraction=0.0,
                   source_optical_transfer=transfer0["optical"],
                   source_sar_transfer=transfer0["sar"],
                   candidate_optical_count=int(masks["optical"].sum()),
                   candidate_sar_count=int(masks["sar"].sum()),
                   **original_metrics)]
        for stage in STAGES:
            for direction in DIRECTIONS:
                for sign in SIGNS:
                    dnew=intervene_detail(model,ctx,stage,direction,sign,target_rms)
                    fused=model.reconstruct(o,bf,dnew)
                    gray=uint8_output(fused)
                    metrics=evaluate_metrics(gray,ou,su)
                    transfer=transfer_values(fused,source_gradient,masks)
                    row=dict(scene=scene,stage=stage,direction=direction,
                             sign=sign,strength=strength,
                             target_projected_rms=target_rms,
                             realized_detail_delta_rms=tensor_rms(dnew-df),
                             image_mae_8bit_unquantized=float((fused-baseline).abs().mean()*255),
                             quantized_changed_pixel_fraction=float(np.mean(gray!=gray0)),
                             source_optical_transfer=transfer["optical"],
                             source_sar_transfer=transfer["sar"],
                             candidate_optical_count=int(masks["optical"].sum()),
                             candidate_sar_count=int(masks["sar"].sum()),
                             **metrics)
                    rows.append(row)
                    if save_gray:
                        import cv2
                        filename=f"{stage}_{direction}_{'plus' if sign>0 else 'minus'}.png"
                        if not cv2.imwrite(str(save_gray/filename),gray):
                            raise OSError(f"Cannot save {filename}")
        return rows


def average(items):
    items=[float(v) for v in items if v is not None and math.isfinite(float(v))]
    return sum(items)/len(items) if items else None


def paired_summary(rows):
    grouped = {}
    by_scene={}
    for row in rows:
        by_scene.setdefault(row["scene"],{})[(row["stage"],row["direction"],row["sign"])]=row
    pairs=[(stage,direction,sign) for stage in STAGES for direction in DIRECTIONS
          for sign in SIGNS]
    for key in pairs:
        comparisons=[]
        for scene,items in by_scene.items():
            if ("baseline","none",0) not in items or key not in items:
                raise ValueError(f"Missing complete paired rows for {scene}/{key}")
            base=items[("baseline","none",0)]
            candidate=items[key]
            delta={}
            for name in (*SELECTED_METRICS,
                         "source_optical_transfer","source_sar_transfer"):
                x,y=base.get(name),candidate.get(name)
                delta[name]=None if x is None or y is None else y-x
            comparisons.append(dict(scene=scene,delta=delta,raw=candidate))
        # Only finite paired results are counted for every separate indicator.
        changes={}
        for metric in (*SELECTED_METRICS,"source_optical_transfer",
                       "source_sar_transfer"):
            v=[x["delta"][metric] for x in comparisons
               if x["delta"][metric] is not None and
                  math.isfinite(x["delta"][metric])]
            changes[metric]=dict(
                n=len(v),mean_delta=average(v),improved_scene_count=sum(
                    ((z<0) if metric=="Nabf" else (z>0)) for z in v),
                median_delta=(float(np.median(v)) if v else None))
        changes["mean_quant_changed_fraction"]=average(
            p["raw"]["quantized_changed_pixel_fraction"] for p in comparisons)
        changes["mean_unquantized_image_mae_8bit"]=average(
            p["raw"]["image_mae_8bit_unquantized"] for p in comparisons)
        changes["mean_realized_detail_delta_rms"]=average(
            p["raw"]["realized_detail_delta_rms"] for p in comparisons)
        grouped[f"{key[0]}:{key[1]}:{key[2]:+d}"]=changes
    return grouped


def matched_contrasts(rows):
    """Optical minus matched SAR/null, same scene/stage/sign/strength."""
    by = {}
    for row in rows:
        by[(row["scene"],row["stage"],row["direction"],row["sign"])]=row
    contrasts=[]
    for scene,stage,direction,sign in sorted(by):
        if direction!="optical" or stage not in STAGES:
            continue
        candidate=by[(scene,stage,direction,sign)]
        for other in ("sar","shuffled_optical"):
            control=by.get((scene,stage,other,sign))
            if control is None:
                raise ValueError("Missing matched source/null control")
            result=dict(scene=scene,stage=stage,sign=sign,
                        comparison=f"optical_minus_{other}")
            for metric in (*SELECTED_METRICS,
                           "source_optical_transfer","source_sar_transfer",
                           "quantized_changed_pixel_fraction",
                           "realized_detail_delta_rms"):
                x,y=candidate.get(metric),control.get(metric)
                result[metric+"_contrast"]=(x-y if x is not None and
                                             y is not None else None)
            contrasts.append(result)
    return contrasts


def write_csv(path, rows):
    with path.open("x",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    if not 1 <= args.val_scenes <= 64:
        raise ValueError("--val-scenes must be 1..64")
    if not 0 < args.strength <= 0.20:
        raise ValueError("--strength must be in (0,0.20]")
    if not 0<=args.save_images<=args.val_scenes:
        raise ValueError("--save-images must be 0..val-scenes")
    output=args.output_dir or OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    manifest=source_manifest()
    val_ids=[str(x) for x in evenly_spaced(
        manifest["splits"]["val"],args.val_scenes)]
    if len(val_ids)!=len(set(val_ids)):
        raise ValueError("Repeated val ID")
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model,checkpoint=frozen_checkpoint(device)
    rows=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".e2-causal-",dir=output.parent) as tmp:
        pending=Path(tmp)/"result"
        pending.mkdir()
        images=pending/"images"
        if args.save_images:
            images.mkdir()
        for i,scene in enumerate(val_ids):
            image_dir=None
            if i<args.save_images:
                image_dir=images/scene
                image_dir.mkdir()
            sample=evaluate_scene(model,scene,manifest,device,args.strength,image_dir)
            rows.extend(sample)
            print(f"causal intervention {i+1}/{len(val_ids)} scene={scene}",flush=True)
        summary=paired_summary(rows)
        contrasts=matched_contrasts(rows)
        contrast_summary={
            f"{stage}:{sign:+d}:optical_minus_{other}":{
                metric:average(row[metric+"_contrast"] for row in contrasts
                               if row["stage"]==stage and row["sign"]==sign and
                                  row["comparison"]==f"optical_minus_{other}")
                for metric in ("Qabf","Nabf","FMI","VIF","source_optical_transfer",
                               "source_sar_transfer")}
            for stage in STAGES for sign in SIGNS
            for other in ("sar","shuffled_optical")}
        report=dict(
            status="frozen_e2_causal_intervention_no_training",
            dataset="QXS-SAROPT",split="matched_pilot_val64",
            checkpoint=checkpoint.as_posix(),
            checkpoint_sha256=sha256_file(checkpoint),
            script_sha256=sha256_file(Path(__file__)),
            split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
            split_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
            val_scene_ids=val_ids,seed=42,device=str(device),strength=args.strength,
            metric_profile=DEFAULT_PROFILE,quantization="clamp_round_uint8",
            interventions={
                "early":"post-GELU source projection; recompute three scale branches and output",
                "late":"direct optical/SAR slice immediately before Detail.output; scales frozen",
                "random_null":"spatially rolled and channel-permuted optical projected feature",
                "amplitude":"equal absolute perturbation RMS for every source/stage/sign within scene",
                "sign":"both positive and negative, no best-sign selection"},
            metric_names=list(METRICS),
            summary=summary,matched_contrast_summary=contrast_summary,
            caveats=[
                "Frozen interventions may be out of training distribution.",
                "Postprojection and pre-output stages have different paths; equal node RMS does not imply equal fused Detail displacement.",
                "SAR high gradients may be speckle, and automatically selected source-specific gradients are not ground-truth structures.",
                "Metric effects of quantized images may be zero at small perturbation amplitudes.",
                "This diagnosis cannot demonstrate the benefits of a retrained fusion architecture."
            ])
        write_csv(pending/"all_scenes_metrics.csv",rows)
        write_csv(pending/"matched_contrasts.csv",contrasts)
        (pending/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),
            encoding="utf-8")
        pending.rename(output)
    print(json.dumps({"per_intervention":summary,
                      "optical_minus_controls":contrast_summary},
                     ensure_ascii=False,indent=2),flush=True)
    print(f"saved {output.resolve()}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-scenes",type=int,default=16)
    parser.add_argument("--strength",type=float,default=0.04)
    parser.add_argument("--save-images",type=int,default=4)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
