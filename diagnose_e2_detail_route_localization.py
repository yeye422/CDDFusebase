"""Frozen E2 optical-detail attenuation localization, aligned to FCSR val pilot.

Four complementary readouts:
(1) exact Detail Fusion intermediate trace vs archived model forward;
(2) equal-RMS optical/SAR Detail input perturbation through source projections,
    multi-scale branches, fused Detail, and the unchanged Decoder;
(3) original E2 hard-max source-supervision winners, including optical-only ROIs;
(4) original-source objective vs optical-candidate gradient disagreement
    at the *fused Detail tensor*, not a claim of training causality.

No model parameter update, no new model training, no test set access.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint, loss_partition
from diagnose_fcsr_decoder import (gradients, rms, signed_transfer,
                                    source_specific_mask)
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, training_args
from run_qxs_reference_test1000 import DATA
from train_direct_fusion import DirectFusionLosses
from utils.dataset_os import evenly_spaced

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("test_result/e2_detail_route_localization_v1/val16_seed42")
EPS = 1e-12
BRANCHES = ("optical", "sar")
TRACE_STAGES = ("projection", "scales", "detail", "image")
DELTA = 0.015


def route_trace(model, optical_detail, sar_detail, fused_base, optical_image):
    """Exact same MultiScaleDetailFusion path as actual model.forward.

    Intermediate optical/sar projections have the same channel count.
    The scale stage concatenates all 3 equal-width branches.
    """
    module = model.detail_fusion
    op, sp, bp = module._project_inputs(optical_detail, sar_detail, fused_base)
    joined = torch.cat((op, sp), dim=1)
    scale = [block(joined) for block in module.scale_branches]
    detail = module.output(torch.cat((op, sp, *scale, bp), dim=1))
    output = model.reconstruct(optical_image, fused_base, detail)
    return dict(optical_projection=op, sar_projection=sp,
                scales=torch.cat(scale,dim=1),
                detail=detail, image=output)


def unit_rms_perturbation(value, delta_rms):
    """Matched absolute RMS in the two source input feature spaces."""
    n = rms(value)
    if n < 1e-10:
        raise ValueError("Degenerate Detail source tensor")
    out = value * (float(delta_rms)/n)
    if abs(rms(out)/float(delta_rms)-1) > 1e-3:
        raise AssertionError("RMS mismatch")
    return out


def cosine(a, b):
    aa, bb = a.detach().float().flatten(), b.detach().float().flatten()
    da = aa.norm()
    db = bb.norm()
    if float(da) <= 1e-15 or float(db) <= 1e-15:
        return None
    return float(torch.dot(aa,bb)/(da*db))


def stage_gain(response, input_delta_rms):
    return rms(response) / max(input_delta_rms,EPS)


def roi_gradient_gain(image_response, roi):
    if int(roi.sum()) < 32:
        return None
    g = gradients(image_response)
    return rms(g.square().sum(1,keepdim=True).sqrt()[roi])


def route_probe(model, optical, sar, optical_detail, sar_detail, base,
                baseline, masks, delta_fraction):
    """Two-sided source perturbations with matched input RMS.

    No source is turned off, and fused Base remains identical.
    """
    input_scale=(rms(optical_detail)+rms(sar_detail))*0.5
    delta_rms=delta_fraction*input_scale
    if delta_rms <= 1e-10:
        raise ValueError("Degenerate source Detail RMS")
    rows=[]
    for name in BRANCHES:
        source=optical_detail if name=="optical" else sar_detail
        d=unit_rms_perturbation(source,delta_rms)
        if name=="optical":
            plus=route_trace(model,optical_detail+d,sar_detail,base,optical)
            minus=route_trace(model,optical_detail-d,sar_detail,base,optical)
            proj_key="optical_projection"
        else:
            plus=route_trace(model,optical_detail,sar_detail+d,base,optical)
            minus=route_trace(model,optical_detail,sar_detail-d,base,optical)
            proj_key="sar_projection"
        responses={
            "projection":(plus[proj_key]-minus[proj_key])*0.5,
            "scales":(plus["scales"]-minus["scales"])*0.5,
            "detail":(plus["detail"]-minus["detail"])*0.5,
            "image":(plus["image"]-minus["image"])*0.5,
        }
        row=dict(
            branch=name, input_perturbation_rms=delta_rms,
            source_feature_rms=rms(source), delta_fraction=delta_fraction)
        for stage,value in responses.items():
            row[stage+"_response_rms"]=rms(value)
            row[stage+"_gain"]=stage_gain(value,delta_rms)
        row["decoder_gain_from_actual_detail_direction"]=(
            rms(responses["image"])/max(rms(responses["detail"]),EPS))
        row["image_optical_roi_grad_rms"]=roi_gradient_gain(
            responses["image"],masks["optical"])
        row["image_sar_roi_grad_rms"]=roi_gradient_gain(
            responses["image"],masks["sar"])
        rows.append(row)
    return rows


def original_winner_stats(optical, sar, valid, masks, losses):
    """Measure source-winner selection in ORIGINAL E2 max-gradient target."""
    with torch.no_grad():
        raw_opt=losses.sobel(optical.float())
        raw_sar=losses.sobel(sar.float())
        winner=(raw_sar>raw_opt)&valid
        win_int=(sar>optical)&valid
        results={}
        for region,mask in (
            ("optical_candidate",masks["optical"]),
            ("sar_candidate",masks["sar"]),
            ("valid_all",valid)):
            n=int(mask.sum())
            if n<32:
                results[region+"_pixels"]=n
                results[region+"_sar_gradient_winner_fraction"]=None
                results[region+"_sar_intensity_winner_fraction"]=None
            else:
                results[region+"_pixels"]=n
                results[region+"_sar_gradient_winner_fraction"]=float(winner[mask].float().mean())
                results[region+"_sar_intensity_winner_fraction"]=float(win_int[mask].float().mean())
        return results


def optical_edge_loss(output, optical, mask):
    if int(mask.sum())<32:
        return None
    difference=(gradients(output)-gradients(optical)).square().sum(1,keepdim=True)
    return difference[mask].mean()


def gradient_conflict(model, losses, optical, sar, valid, fused_base, fused_detail,
                      masks, weight):
    """Gradients at the fused Detail interface; all network weights frozen.

    Edge objective is diagnostic-only. Signed transfer and optimizer effects are
    NOT estimated as inference-quality or ground-truth measures.
    """
    if int(masks["optical"].sum())<32:
        return dict(status="insufficient_optical_candidates")
    with torch.enable_grad():
        d=fused_detail.detach().clone().requires_grad_(True)
        result=model.reconstruct(optical,fused_base.detach(),d)
        loss_o,loss_s,_=loss_partition(optical,sar,result,valid,losses,weight)
        oracle=optical_edge_loss(result,optical,masks["optical"])
        grad_o=torch.autograd.grad(loss_o,d,retain_graph=True)[0]
        grad_s=torch.autograd.grad(loss_s,d,retain_graph=True)[0]
        grad_oracle=torch.autograd.grad(oracle,d)[0]
        total=grad_o+grad_s
    with torch.no_grad():
        true_total,_,_=losses.fusion(optical,sar,result.detach(),valid,weight)
        assert abs(float((loss_o+loss_s).detach())-float(true_total))<=1e-4, "Original E2 loss partition differs"
    return dict(status="ok",
        loss_original_optical=float(loss_o.detach()),
        loss_original_sar=float(loss_s.detach()),
        loss_optical_edge_proxy=float(oracle.detach()),
        optical_vs_sar_gradient_cosine=cosine(grad_o,grad_s),
        original_vs_optical_edge_gradient_cosine=cosine(total,grad_oracle),
        original_optical_vs_edge_gradient_cosine=cosine(grad_o,grad_oracle),
        original_sar_vs_edge_gradient_cosine=cosine(grad_s,grad_oracle),
        original_sar_to_optical_gradient_norm=rms(grad_s)/max(rms(grad_o),EPS),
        detail_original_gradient_norm=rms(total),
        detail_edge_gradient_norm=rms(grad_oracle))


def image_transfer_rows(model, optical, sar, encoded, fused_base, image, masks):
    bo,do,bs,ds=encoded
    with torch.no_grad():
        recon_o=model.reconstruct(optical,bo,do)
        recon_s=model.reconstruct(sar,bs,ds)
        go,gs=gradients(optical),gradients(sar)
        gf=gradients(image)
        gr_o,gr_s=gradients(recon_o),gradients(recon_s)
        return [
          dict(branch="optical",candidate_pixels=int(masks["optical"].sum()),
               reconstructed_transfer=signed_transfer(go,gr_o,masks["optical"]),
               fused_transfer=signed_transfer(go,gf,masks["optical"])),
          dict(branch="sar",candidate_pixels=int(masks["sar"].sum()),
               reconstructed_transfer=signed_transfer(gs,gr_s,masks["sar"]),
               fused_transfer=signed_transfer(gs,gf,masks["sar"])),
        ]


def mean_values(rows, field):
    values=[float(row[field]) for row in rows
            if isinstance(row.get(field),(int,float)) and math.isfinite(float(row[field]))]
    return sum(values)/len(values) if values else None


def summary(routes,winners,losses,transfers):
    grouped={}
    for branch in BRANCHES:
        selected=[r for r in routes if r["branch"]==branch]
        grouped[branch]={
            stage:mean_values(selected,stage+"_gain") for stage in TRACE_STAGES
        }
        grouped[branch]["decoder_gain"]=mean_values(
            selected,"decoder_gain_from_actual_detail_direction")
        grouped[branch]["optical_roi_response"]=mean_values(
            selected,"image_optical_roi_grad_rms")
        grouped[branch]["sar_roi_response"]=mean_values(
            selected,"image_sar_roi_grad_rms")
    ratios={}
    by_scene={}
    for row in routes:
        by_scene.setdefault(row["scene"],{})[row["branch"]]=row
    for stage in TRACE_STAGES:
        ratios["optical_to_sar_"+stage+"_gain"]=mean_values(
            [dict(ratio=pair["optical"][stage+"_gain"]/
                  max(pair["sar"][stage+"_gain"],EPS))
             for pair in by_scene.values() if len(pair)==2],"ratio")
    ratios["optical_to_sar_decoder_gain"]=mean_values(
        [dict(ratio=pair["optical"]["decoder_gain_from_actual_detail_direction"]/
              max(pair["sar"]["decoder_gain_from_actual_detail_direction"],EPS))
         for pair in by_scene.values() if len(pair)==2],"ratio")
    transfer_summary={}
    for branch in BRANCHES:
        selected=[r for r in transfers if r["branch"]==branch]
        transfer_summary[branch]={
            "mean_reconstruction":mean_values(selected,"reconstructed_transfer"),
            "mean_fused":mean_values(selected,"fused_transfer"),
            "median_fused":(float(median([r["fused_transfer"] for r in selected
                                      if r["fused_transfer"] is not None]))
                            if any(r["fused_transfer"] is not None for r in selected) else None)}
    return dict(
        sample_count=len(by_scene),
        routes=grouped,
        scene_mean_source_gain_ratios=ratios,
        optical_candidate_original_sar_gradient_winner_fraction=mean_values(
            winners,"optical_candidate_sar_gradient_winner_fraction"),
        valid_all_original_sar_gradient_winner_fraction=mean_values(
            winners,"valid_all_sar_gradient_winner_fraction"),
        feature_gradient_conflict={
            "valid_scenes":sum(r["status"]=="ok" for r in losses),
            "mean_original_vs_optical_edge_cosine":mean_values(
                losses,"original_vs_optical_edge_gradient_cosine"),
            "mean_optical_vs_sar_cosine":mean_values(
                losses,"optical_vs_sar_gradient_cosine"),
            "mean_sar_to_optical_gradient_norm":mean_values(
                losses,"original_sar_to_optical_gradient_norm")},
        transfer=transfer_summary)


def write_csv(path,rows):
    if not rows:
        raise ValueError("Empty results")
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x",newline="",encoding="utf-8-sig") as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    if not (1<=args.val_scenes<=64 and 0<args.delta<=0.03):
        raise ValueError("val scenes 1..64; --delta must be in (0,0.03]")
    output=args.output_dir or DEFAULT_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    manifest=source_manifest()
    ids=[str(x) for x in evenly_spaced(manifest["splits"]["val"],args.val_scenes)]
    if len(ids)!=len(set(ids)):
        raise ValueError("Repeated val IDs")
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model,checkpoint=frozen_checkpoint(device)
    metric_loss=DirectFusionLosses().to(device).eval()
    metric_loss.requires_grad_(False)
    weight=float(training_args(ORIGINAL_LOSS_PILOT).gradient_weight)
    routes,winners,gradients_out,transfers=[],[],[],[]
    for index,scene in enumerate(ids):
        ou,su=read_pair(locate_scene(DATA,manifest,scene))
        optical,sar=to_tensor(ou,device),to_tensor(su,device)
        valid=(sar>0)
        with torch.no_grad():
            encoded=model.encode_pair(optical,sar)
            bf,df=model.fuse_features(*encoded)
            trace=route_trace(model,encoded[1],encoded[3],bf,optical)
            torch.testing.assert_close(df,trace["detail"],rtol=0,atol=1e-6)
            torch.testing.assert_close(model.reconstruct(optical,bf,df),
                                       trace["image"],rtol=0,atol=1e-6)
            go,gs=gradients(optical),gradients(sar)
            masks={
              "optical":source_specific_mask(go,gs,valid),
              "sar":source_specific_mask(gs,go,valid)
            }
            r=route_probe(model,optical,sar,encoded[1],encoded[3],bf,trace,masks,args.delta)
            win=original_winner_stats(optical,sar,valid,masks,metric_loss)
            trans=image_transfer_rows(model,optical,sar,encoded,bf,trace["image"],masks)
        grad=gradient_conflict(model,metric_loss,optical,sar,valid,bf,df,masks,weight)
        for row in r: row["scene"]=scene
        for row in trans: row["scene"]=scene
        win["scene"]=scene;grad["scene"]=scene
        routes.extend(r);transfers.extend(trans);winners.append(win)
        gradients_out.append(grad)
        print(f"Detail route localization {index+1}/{len(ids)} scene={scene}",
              flush=True)
    grouped=summary(routes,winners,gradients_out,transfers)
    report=dict(
        status="frozen_detail_route_and_loss_target_diagnosis_no_training",
        dataset="QXS-SAROPT",split="matched_pilot_val64",
        checkpoint=checkpoint.as_posix(),
        checkpoint_sha256=sha256_file(checkpoint),
        script_sha256=sha256_file(Path(__file__)),
        split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
        split_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
        val_scene_ids=ids,device=str(device),delta=args.delta,
        gradient_weight=weight,
        protocols={
          "inputs":"same frozen optical-Base E2 fused_base; source Detail probes match input perturbation RMS",
          "stages":"projected source, concatenated multi-scale branch, fused Detail, decoded image",
          "gradient_target":"original E2 hard-max Sobel and intensity, valid SAR masking",
          "optical_edge_proxy":"source-only smoothed-gradient candidates, not verified true contours",
          "feature_gradient":"autograd wrt fused_detail only; does not update weights"},
        summary=grouped,
        caveats=[
          "Scalar route gains measure local sensitivity, not information utility.",
          "Original-source winner fractions are label pressure, not learned causal importance.",
          "Gradient conflict with optical proxy is a local result, not full training causality.",
          "Optical/SAR candidates are unlabeled; SAR high gradients may be speckle.",
          "No new model, no FCSR, and no fusion-quality metric changes were evaluated."
        ])
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".e2-route-",dir=output.parent) as temp:
        pending=Path(temp)/"result"
        pending.mkdir()
        for filename,rows in (
            ("route_stages.csv",routes),
            ("original_loss_winners.csv",winners),
            ("feature_gradient_conflict.csv",gradients_out),
            ("edge_transfer.csv",transfers)):
            write_csv(pending/filename,rows)
        (pending/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),
            encoding="utf-8")
        pending.rename(output)
    print(json.dumps(grouped,ensure_ascii=False,indent=2),flush=True)
    print(f"Saved: {output.resolve()}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-scenes",type=int,default=16)
    ap.add_argument("--delta",type=float,default=DELTA)
    ap.add_argument("--cpu",action="store_true")
    ap.add_argument("--output-dir",type=Path)
    run(ap.parse_args())


if __name__=="__main__":
    main()
