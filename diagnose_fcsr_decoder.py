"""Frozen FCSR preflight: locate frequency mixing and source-edge transmission.

Uses archived optical-Base E2 original-loss pilot, locked pilot val64 IDs.
No model training, new decoder, SAR ground-truth structure claims or test data.
Frequency readouts describe local *response*, not whether mixing is harmful.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_qxs_reference_test1000 import DATA
from utils.dataset_os import evenly_spaced

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("test_result/e2_fcsr_decoder_diagnosis_v1/val16_seed42")
STAGES = ("reduce", "transform", "logits", "image")
BRANCHES = ("base", "detail")
BANDS = ("coarse", "middle", "fine")
EPS = 1e-12


def feature_bands(x):
    """Approximate spatial scale split, exactly additive, NOT strict DWT."""
    h, w = x.shape[-2:]
    if h % 4 or w % 4:
        raise ValueError("Feature height and width must be divisible by 4")
    p2 = F.interpolate(F.avg_pool2d(x, 2), size=(h, w),
                       mode="bilinear", align_corners=False)
    p4 = F.interpolate(F.avg_pool2d(x, 4), size=(h, w),
                       mode="bilinear", align_corners=False)
    return {"coarse": p4, "middle": p2 - p4, "fine": x - p2}


def haar_detail_fraction(x):
    """Energy share of single-level orthonormal Haar highpass coefficients."""
    if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError("Expected BCHW with even spatial dimensions")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a+b+c+d) * 0.5
    lh = (a-b+c-d) * 0.5
    hl = (a+b-c-d) * 0.5
    hh = (a-b-c+d) * 0.5
    total = (ll.square()+lh.square()+hl.square()+hh.square()).sum()
    high = (lh.square()+hl.square()+hh.square()).sum()
    return float((high / total.clamp_min(EPS)).detach())


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def decoder_taps(model, base, detail):
    """Exact no-image-skip E2 decoder computation, exposing internal stages."""
    dec = model.decoder
    z = dec.reduce_channel(torch.cat((base, detail), dim=1))
    t = dec.encoder_level2(z)
    logits = dec.output(t)
    image = dec.sigmoid(logits)
    return {"reduce": z, "transform": t, "logits": logits, "image": image}


def equal_rms_delta(component, target_rms):
    strength = rms(component)
    if strength < 1e-10:
        raise ValueError("Degenerate frequency component")
    return component * (target_rms / strength)


def probe_response(model, bf, df, fraction=0.015):
    """Central-difference decoder Jacobian probes, equal input perturb RMS."""
    reference = math.sqrt((rms(bf)**2 + rms(df)**2)/2)
    if reference <= 1e-10:
        raise ValueError("Degenerate fused input")
    target_rms = fraction * reference
    results = []
    for branch, tensor in (("base", bf), ("detail", df)):
        parts = feature_bands(tensor)
        torch.testing.assert_close(
            sum(parts.values()), tensor, rtol=0, atol=1e-5)
        for band_name, part in parts.items():
            magnitude = rms(part)
            if magnitude < 1e-10:
                results.append(dict(branch=branch, band=band_name,
                                    status="degenerate_band", band_rms=magnitude))
                continue
            displacement = equal_rms_delta(part, target_rms)
            if branch == "base":
                positive = decoder_taps(model, bf + displacement, df)
                negative = decoder_taps(model, bf - displacement, df)
            else:
                positive = decoder_taps(model, bf, df + displacement)
                negative = decoder_taps(model, bf, df - displacement)
            row = dict(branch=branch, band=band_name, status="ok",
                       branch_rms=rms(tensor), band_rms=magnitude,
                       perturb_rms=rms(displacement),
                       perturb_fraction=fraction,
                       input_band_high_fraction=haar_detail_fraction(displacement))
            for stage in STAGES:
                # The difference is signal caused by the controlled perturbation.
                response = (positive[stage] - negative[stage]) * 0.5
                row[stage + "_response_rms"] = rms(response)
                row[stage + "_normalized_gain"] = rms(response) / max(target_rms, EPS)
                row[stage + "_high_fraction"] = haar_detail_fraction(response)
                row[stage + "_highpass_gain"] = (row[stage + "_normalized_gain"] *
                    math.sqrt(row[stage + "_high_fraction"]))
            results.append(row)
    return results


def mixing_stats(model, bf, df):
    """Inspect the actual learned 1x1 channel reducer, not a new fusion rule."""
    kernel = model.decoder.reduce_channel.weight
    channels = bf.shape[1]
    base = F.conv2d(bf, kernel[:, :channels], bias=None)
    detail = F.conv2d(df, kernel[:, channels:], bias=None)
    joint = base + detail
    if model.decoder.reduce_channel.bias is not None:
        joint = joint + model.decoder.reduce_channel.bias.reshape(1,-1,1,1)
    actual = model.decoder.reduce_channel(torch.cat((bf, df), dim=1))
    torch.testing.assert_close(joint, actual, rtol=1e-4, atol=1e-4)
    dot = float((base*detail).mean())
    rb, rd = rms(base), rms(detail)
    return dict(base_projected_rms=rb, detail_projected_rms=rd,
                projection_cosine=dot/max(rb*rd, EPS),
                combined_rms=rms(actual),
                combined_to_component_rms=rms(actual)/max(rb+rd, EPS),
                base_projection_high_fraction=haar_detail_fraction(base),
                detail_projection_high_fraction=haar_detail_fraction(detail),
                combined_projection_high_fraction=haar_detail_fraction(actual))


def gradients(x):
    """Smoothed source-image Sobel (display-domain, no speckle labels)."""
    smooth = F.avg_pool2d(F.pad(x, (2,2,2,2), mode="reflect"), 5, stride=1)
    kernel = x.new_tensor([[-1.,0.,1.], [-2.,0.,2.], [-1.,0.,1.]]) / 8
    return F.conv2d(F.pad(smooth, (1,1,1,1), mode="reflect"),
                    torch.stack((kernel,kernel.T))[:,None])


def source_specific_mask(g_source, g_other, valid, high_q=0.85, low_q=0.5):
    """Unsupervised contrast-discrepancy candidates, not ground truth edges."""
    s = g_source.square().sum(1,keepdim=True).sqrt()
    o = g_other.square().sum(1,keepdim=True).sqrt()
    interior = torch.zeros_like(valid, dtype=torch.bool)
    interior[..., 8:-8, 8:-8] = True
    permitted = interior & valid
    if int(permitted.sum()) < 256:
        return torch.zeros_like(permitted)
    high = torch.quantile(s[permitted], high_q)
    low = torch.quantile(o[permitted], low_q)
    # Do not call zero-contrast homogeneous regions 'high-gradient' edges.
    if float(high) <= 1e-7:
        return torch.zeros_like(permitted)
    return permitted & (s >= high) & (o <= low)


def signed_transfer(gradient_source, gradient_candidate, mask):
    if mask.sum() < 32:
        return None
    src, dst = gradient_source, gradient_candidate
    # dot over both gradient components, only at the selected pixels
    numerator = ((src*dst).sum(1,keepdim=True)[mask]).sum()
    denominator = (src.square().sum(1,keepdim=True)[mask]).sum()
    if float(denominator) < EPS:
        return None
    return float(numerator / denominator)


def edge_probe(model, encoded, bf, df, optical, sar, image, fraction=0.04):
    """Compare source reconstruction to fused output and modest source-directed
    feature interpolation at the *same* frozen decoder. No causal loss label.
    """
    bo, do, bs, ds = encoded
    recon_o = decoder_taps(model, bo, do)["image"]
    recon_s = decoder_taps(model, bs, ds)["image"]
    ga, gb = gradients(optical), gradients(sar)
    go, gs = gradients(recon_o), gradients(recon_s)
    gf = gradients(image)
    valid = (sar > 0)
    candidate_masks = {
        "optical": source_specific_mask(ga,gb,valid),
        "sar": source_specific_mask(gb,ga,valid),
    }
    shifted_b = decoder_taps(model, bf + fraction*(bo-bf), df)["image"]
    shifted_d = decoder_taps(model, bf, df + fraction*(do-df))["image"]
    shifted_both = decoder_taps(
        model, bf + fraction*(bo-bf), df + fraction*(do-df))["image"]
    grads_changed = {
        "optical_base_direction": gradients(shifted_b),
        "optical_detail_direction": gradients(shifted_d),
        "both_optical_directions": gradients(shifted_both)
    }
    rows = []
    for name, src_g, src_recon_g in (
        ("optical", ga, go), ("sar", gb, gs)):
        mask = candidate_masks[name]
        row = dict(source=name, candidate_pixels=int(mask.sum()),
                   selection="source_gradient_top15pct_other_bottom50pct",
                   source_reconstruction_transfer=signed_transfer(src_g,src_recon_g,mask),
                   fused_transfer=signed_transfer(src_g,gf,mask),
                   source_feature_interpolation=fraction)
        # These counterfactual directions are optical-only and only have
        # a direct source interpretation for optical candidates.
        if name == "optical":
            for key, changed in grads_changed.items():
                row[key + "_transfer"] = signed_transfer(src_g,changed,mask)
        rows.append(row)
    return rows, recon_o, recon_s


def save_preview(path, optical, sar, image, recon_o, recon_s, scene):
    """Five aligned grayscale panels; no inferred edge labels."""
    def uint8(x):
        return (x.detach().clamp(0,1)[0,0].cpu().numpy()*255).round().astype(np.uint8)
    cells = [
        ("Optical",uint8(optical)), ("SAR",uint8(sar)),
        ("E2 fused",uint8(image)), ("O recon",uint8(recon_o)),
        ("SAR recon",uint8(recon_s))]
    canvas = np.concatenate(
        [cv2.copyMakeBorder(cell,24,0,0,0,cv2.BORDER_CONSTANT,value=0)
         for _,cell in cells], axis=1)
    for i, (label, _) in enumerate(cells):
        cv2.putText(canvas,label,(i*256+8,17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,255,1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas):
        raise OSError(f"Could not save preview for {scene}")


def mean_valid(rows, key):
    values = [float(row[key]) for row in rows
              if row.get(key) is not None and
              isinstance(row[key], (int,float)) and
              math.isfinite(float(row[key]))]
    return sum(values)/len(values) if values else None


def summarize(frequency, edges, mixes):
    by_probe={}
    for branch in BRANCHES:
        for band in BANDS:
            items=[x for x in frequency if x["branch"]==branch and
                   x["band"]==band and x["status"]=="ok"]
            by_probe[f"{branch}:{band}"]={
                "scenes":len(items),
                "mean_normalized_image_gain":mean_valid(items,"image_normalized_gain"),
                "mean_image_high_fraction":mean_valid(items,"image_high_fraction"),
                "mean_image_highpass_gain":mean_valid(items,"image_highpass_gain"),
                "mean_reduce_high_fraction":mean_valid(items,"reduce_high_fraction"),
                "mean_transform_high_fraction":mean_valid(items,"transform_high_fraction"),
                "mean_logits_high_fraction":mean_valid(items,"logits_high_fraction")}
    weak={}
    for source in ("optical","sar"):
        items=[x for x in edges if x["source"]==source]
        weak[source]={
            "scenes":len(items), "mean_candidate_pixels":mean_valid(items,"candidate_pixels"),
            "mean_source_reconstruction_transfer":mean_valid(items,"source_reconstruction_transfer"),
            "mean_fused_transfer":mean_valid(items,"fused_transfer")}
        if source=="optical":
            for name in ("optical_base_direction","optical_detail_direction","both_optical_directions"):
                weak[source]["mean_"+name+"_transfer"]=mean_valid(
                    items,name+"_transfer")
    return dict(
        frequency_response=by_probe,
        edge_candidates=weak,
        projection=dict(
            mean_projection_cosine=mean_valid(mixes,"projection_cosine"),
            mean_combined_to_component_rms=mean_valid(mixes,"combined_to_component_rms")),
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty diagnostic results")
    fieldnames=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x",newline="",encoding="utf-8-sig") as fp:
        writer=csv.DictWriter(fp,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    if not (1<=args.val_scenes<=64):
        raise ValueError("val scenes must be 1..64")
    if not (0<args.delta<=0.05 and 0<args.interpolation<=0.1):
        raise ValueError("delta must be <=0.05 and interpolation <=0.1")
    output=args.output_dir or OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest=source_manifest()
    ids=[str(x) for x in evenly_spaced(manifest["splits"]["val"],args.val_scenes)]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate scene IDs")
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model,checkpoint=frozen_checkpoint(device)
    frequency,edges,mixing=[],[],[]
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fcsr-preflight-",dir=output.parent) as tmp:
        pending=Path(tmp)/"result"
        pending.mkdir()
        preview=pending/"previews"
        if args.preview_count:
            preview.mkdir()
        with torch.inference_mode():
            for index,scene in enumerate(ids):
                ou,su=read_pair(locate_scene(DATA,manifest,scene))
                o,s=to_tensor(ou,device),to_tensor(su,device)
                encoded=model.encode_pair(o,s)
                bf,df=model.fuse_features(*encoded)
                baseline=decoder_taps(model,bf,df)
                torch.testing.assert_close(
                    baseline["image"],model.reconstruct(o,bf,df),
                    atol=1e-6,rtol=0)
                f=probe_response(model,bf,df,args.delta)
                m=mixing_stats(model,bf,df)
                e,ro,rs=edge_probe(model,encoded,bf,df,o,s,baseline["image"],
                                   args.interpolation)
                for row in f: row["scene"]=scene
                for row in e: row["scene"]=scene
                frequency.extend(f)
                edges.extend(e)
                m["scene"]=scene
                m["output_saturation_fraction"]=float(
                    ((baseline["image"]<0.05)|(baseline["image"]>0.95)).float().mean())
                mixing.append(m)
                if index<args.preview_count:
                    save_preview(preview/f"{scene}.png",o,s,baseline["image"],ro,rs,scene)
                print(f"FCSR decoder diagnosis {index+1}/{len(ids)} scene={scene}",
                      flush=True)
        summary=summarize(frequency,edges,mixing)
        report=dict(
            status="frozen_e2_decoder_interventions_no_training",
            dataset="QXS-SAROPT",
            split="matched_pilot_val64",
            checkpoint=checkpoint.as_posix(),
            checkpoint_sha256=sha256_file(checkpoint),
            script_sha256=sha256_file(Path(__file__)),
            source_manifest=str(
                __import__("run_e2_scratch_twostage").ORIGINAL_LOSS_PILOT.manifest),
            val_scene_ids=ids, device=str(device), delta=args.delta,
            optical_interpolation=args.interpolation,
            preview_count=args.preview_count,
            protocols={
                "band_split":"nonorthogonal additive 2x/4x spatial-pooling decomposition",
                "response":"symmetric finite difference at exactly equal perturbation RMS",
                "frequency":"single-level orthonormal Haar highpass energy fraction of response",
                "edge":"source-gradient top15pct and other-source bottom50pct; no truth labels",
                "source_recon":"same final E2 decoder, original source Base/Detail",
            },
            summary=summary,
            caveats=[
                "Cross-band Jacobian response does not prove harmful frequency mixing.",
                "One-stage Haar energy is an image/feature spatial-band proxy.",
                "Single-source reconstruction contrast loss does not localize the defect to the decoder.",
                "Optical source-directed interpolation is out-of-distribution; diagnostic only.",
                "SAR source edges may reflect speckle, not genuine scatterer structure.",
                "FCSR and trainable alternative decoders are NOT evaluated."
            ])
        write_csv(pending/"frequency_response.csv",frequency)
        write_csv(pending/"channel_mix.csv",mixing)
        write_csv(pending/"edge_candidates.csv",edges)
        (pending/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),
            encoding="utf-8")
        pending.rename(output)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(f"Saved: {output.resolve()}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-scenes",type=int,default=16)
    ap.add_argument("--delta",type=float,default=0.015)
    ap.add_argument("--interpolation",type=float,default=0.04)
    ap.add_argument("--preview-count",type=int,default=4)
    ap.add_argument("--cpu",action="store_true")
    ap.add_argument("--output-dir",type=Path)
    run(ap.parse_args())


if __name__=="__main__":
    main()
