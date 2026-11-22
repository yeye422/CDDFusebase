"""V3: screen whether MaRS adds usable structure beyond archived frozen E2.

No Adapter training, no fusion changes, no test set. Ridge *readout* trained
only on locked pilot train scenes to predict source Optical-Y gradient magnitude.
Default validation uses locked val64: eight manually audited STG scenes and
56 unaudited scenes reported separately. No test-set access.

Compare E2 rich (shared + signed/absolute/energy Haar Detail) against E2 plus
equal-dimensional TRAIN-only PCA branches: pretrained MaRS, three random
SwinV2 initializations, shifted MaRS and inexpensive source-structure features.
Both 128/64 grids must pass fixed incremental and coherent-line screening.
RGB requires a separate exploratory run; only small Adapter ablation admission
can be recommended, never actual fusion improvement.

Official upstream: WanderRainy/MaRS, timm==1.0.15, SwinV2 Base,
mars_base_rgb_encoder_only.pth (Zenodo record 17800805).
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diagnose_e2_information_path import read_pair, to_tensor
from diagnose_stg_no_train import (
    DATASET, DEFAULT_LABELS, endpoints, load_inputs, load_visibility,
)
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from mars_probe_evidence import (
    DECISION_RULES, admission_decision, coherent_path_metrics, fit_projection, fit_residualizers,
    paired_evidence, project_rows, resize_channels, source_structure_features,
)

ROOT=Path(__file__).resolve().parent
DEFAULT_OUT=Path("test_result/mars_frozen_prior_probe_v3/train32_val64_gray_d32")
DEFAULT_WEIGHTS=Path("models/external/mars_base_rgb_encoder_only.pth")
METRIC_PROFILE="mars_incremental_structure_screen_v3"
OUTPUT_POLICY="frozen_e2_rich_plus_train_pca_residualized_controls_v3"
STAGES=(0,1)
GRIDS={0:128,1:64}
RNG_SEED=20261118
PRETRAIN_MEAN=np.array([87.01,91.52,83.51],np.float32)
PRETRAIN_STD=np.array([62.66,54.58,53.11],np.float32)


def edge_target(gray):
    """Source-derived unsigned optical edge label, not ground truth of fusion."""
    x=gray.astype(np.float32)/255.0
    smooth=cv2.GaussianBlur(x,(0,0),1.25)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    return np.clip(np.sqrt(gx*gx+gy*gy)/0.25,0,1).astype(np.float32)


def feature_grid(x,size):
    """Area-match shared features or coefficient channels to the MaRS grid.

    Never apply this directly to reconstructed signed Haar Detail: its 2x2
    block means are zero. Extract its directional coefficient channels first.
    """
    if x.ndim!=4 or x.shape[0]!=1:
        raise ValueError("Expected B,C,H,W singleton feature")
    if x.shape[-2:]!=(size,size):
        x=F.interpolate(x,size=(size,size),mode="area")
    result=x[0].detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite backbone features")
    return result


def standardize_mars_rgb(rgb_0_255,gray_0_255,mode):
    """Official MaRS raw-0..255 RGB mean/std; no ImageNet normalization."""
    if mode not in ("gray","rgb"):
        raise ValueError("Unsupported input mode")
    if rgb_0_255.shape[:2]!=gray_0_255.shape or rgb_0_255.shape[-1]!=3:
        raise ValueError("Misaligned Optical RGB/Y")
    image=(np.repeat(gray_0_255[...,None],3,axis=-1) if mode=="gray"
           else rgb_0_255).astype(np.float32)
    image=cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
    image=np.clip(image,0,255)
    image=(image-PRETRAIN_MEAN)/PRETRAIN_STD
    if not np.isfinite(image).all():
        raise FloatingPointError("Invalid MaRS image input")
    return torch.from_numpy(np.transpose(image,(2,0,1)).copy())[None]


def mars_nchw(outputs,channels):
    """SwinV2 timm can emit NHWC; disambiguate stage0 using stage1 shape."""
    if len(outputs)!=2 or len(channels)!=2:
        raise ValueError("Expected first two SwinV2 feature stages")
    second=outputs[1]
    if second.ndim!=4:
        raise ValueError("SwinV2 feature is not 4D")
    if second.shape[1]==channels[1] and second.shape[-1]!=channels[1]:
        layout="nchw"
    elif second.shape[-1]==channels[1] and second.shape[1]!=channels[1]:
        layout="nhwc"
    else:
        raise ValueError("Cannot disambiguate SwinV2 feature layout")
    maps=[]
    for index,(x,c) in enumerate(zip(outputs,channels)):
        if x.ndim!=4:
            raise ValueError("Invalid feature rank")
        x=x if layout=="nchw" else x.permute(0,3,1,2).contiguous()
        if x.shape[1]!=c or x.shape[-1]!=GRIDS[index] or x.shape[-2]!=GRIDS[index]:
            raise ValueError(f"Unexpected SwinV2 feature size at stage {index}: {tuple(x.shape)}")
        maps.append(x)
    return maps


def shuffled_grid(x):
    """Same feature values; wrong spatial correspondence at train AND val."""
    if x.ndim!=3:
        raise ValueError("Expected CHW")
    return np.roll(x,(max(1,x.shape[1]//3),max(1,x.shape[2]//4)),
                   axis=(1,2)).copy()


class RidgeStats:
    """Streaming centered ridge readout; no storage of training images."""
    def __init__(self,channels,alpha=0.01):
        self.c=int(channels)
        self.alpha=float(alpha)
        if not 0<alpha<1:
            raise ValueError("Positive ridge alpha must be below 1")
        self.n=0
        self.sx=np.zeros(self.c,np.float64)
        self.xx=np.zeros((self.c,self.c),np.float64)
        self.sy=0.
        self.xy=np.zeros(self.c,np.float64)
        self.yy=0.

    def add(self,feat,target,sample_count,seed):
        if feat.ndim!=3 or target.shape!=feat.shape[1:]:
            raise ValueError("Mismatched target and feature grid")
        n=target.size
        rng=np.random.default_rng(seed)
        idx=rng.choice(n,min(sample_count,n),replace=False)
        x=feat.reshape(self.c,n).T[idx].astype(np.float64)
        y=target.ravel()[idx].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise FloatingPointError("Nonfinite ridge inputs")
        self.n+=len(idx)
        self.sx+=x.sum(0)
        self.sy+=y.sum()
        self.yy+=np.dot(y,y)
        self.xx+=x.T@x
        self.xy+=x.T@y

    def solve(self):
        if self.n<=self.c*2:
            raise ValueError("Insufficient ridge training pixels")
        mean=self.sx/self.n
        std=np.sqrt(np.maximum(np.diag(self.xx)/self.n-mean**2,0))
        std=np.maximum(std,1e-4)
        covariance=self.xx/self.n-np.outer(mean,mean)
        cross=self.xy/self.n-mean*(self.sy/self.n)
        a=covariance/std[:,None]/std[None,:]+self.alpha*np.eye(self.c)
        b=cross/std
        w=np.linalg.solve(a,b)
        # predict_ridge already centers x. Its intercept is therefore y_mean,
        # not the intercept for an uncentered x/std design matrix.
        intercept=self.sy/self.n
        if not np.isfinite(w).all() or not math.isfinite(intercept):
            raise FloatingPointError("Unstable ridge fit")
        return dict(mean=mean,std=std,coef=w,intercept=intercept,
                    count=self.n,alpha=self.alpha)


def predict_ridge(fitted,feat):
    c,h,w=feat.shape
    if c!=len(fitted["coef"]):
        raise ValueError("Ridge channel mismatch")
    x=feat.reshape(c,-1).T.astype(np.float64)
    prediction=((x-fitted["mean"])/fitted["std"])@fitted["coef"]+fitted["intercept"]
    return np.clip(prediction.reshape(h,w),0,1).astype(np.float32)


def pixel_metrics(pred,target):
    if pred.shape!=target.shape:
        raise ValueError("Need matching source-grid predictions")
    err=pred-target
    truth=target.ravel()
    upper=target>=np.quantile(target,0.8)
    return dict(mse=float(np.mean(err**2)),
                mae=float(np.mean(np.abs(err))),
                high_edge_mae=float(np.mean(np.abs(err[upper]))),
                r2=float(1.-np.mean(err**2)/(np.var(truth)+1e-10)),
                target_positive_fraction=float(np.mean(upper)))


def line_score(map256,row):
    """Sample fixed midpoint-to-midpoint corridor, no per-method adjustment."""
    a,b=endpoints(row).astype(np.float32)
    direction=b-a
    length=float(np.linalg.norm(direction))
    if length<8:
        raise ValueError("Degenerate archived line")
    normal=np.array([-direction[1],direction[0]],dtype=np.float32)/length
    samples=a[None]+np.linspace(.15,.85,33,dtype=np.float32)[:,None]*direction
    coords=samples[None]+np.arange(-4,5,dtype=np.float32)[:,None,None]*normal
    x=coords[:,:,0].astype(np.float32)
    y=coords[:,:,1].astype(np.float32)
    corridor=cv2.remap(map256,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    best=corridor.max(0)
    return dict(line_mean=float(best.mean()),
                line_continuity=float(np.mean(best>=0.20)),
                line_p25=float(np.quantile(best,0.25)),
                **coherent_path_metrics(corridor))


def summary_rows(rows,key_fields,metric_fields,reference):
    grouped={}
    for row in rows:
        key=tuple(row[k] for k in key_fields)
        grouped.setdefault(key,{})[row["method"]]=row
    out=[]
    for stage in STAGES:
        for method in sorted({r["method"] for r in rows}):
            if method==reference:
                continue
            matches=[(pair[method],pair[reference]) for key,pair in grouped.items()
                     if key[0]==stage and method in pair and reference in pair]
            if not matches:
                continue
            result=dict(stage=stage,method=method,reference=reference,n_pairs=len(matches))
            for metric in metric_fields:
                diff=[float(a[metric])-float(b[metric]) for a,b in matches
                      if np.isfinite(a[metric]) and np.isfinite(b[metric])]
                result[metric+"_mean_delta"]=float(np.mean(diff)) if diff else None
                result[metric+"_median_delta"]=float(np.median(diff)) if diff else None
                result[metric+"_positive"]=sum(x>0 for x in diff)
                result[metric+"_finite_n"]=len(diff)
            out.append(result)
    return out


def rgb_loader(paths):
    optical_path=paths[0]
    bgr=cv2.imread(str(optical_path),cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Unreadable GF2 RGB: {optical_path}")
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def train_ids(manifest,n):
    source=[str(s) for s in manifest["splits"]["train"]]
    if not 1<=n<=len(source):
        raise ValueError("Invalid number of train scenes")
    from utils.dataset_os import evenly_spaced
    result=evenly_spaced(source,n)
    if len(set(result))!=n:
        raise ValueError("Duplicate training scene IDs")
    return result


def source_paths(scene,manifest):
    base=DATASET
    return (base/manifest["optical_dir"]/f"{scene}.png",
            base/manifest["sar_dir"]/f"{scene}.png")


def extract_e2(model,paths,device):
    o,s=read_pair(paths)
    with torch.no_grad():
        pair=model.encode_pair(to_tensor(o,device),to_tensor(s,device))
        shared=model.encoder.encoder_level1(model.encoder.patch_embed(to_tensor(o,device)))
        # Detail is IDWT(0, gated_LH, gated_HL, gated_HH). Direct area pooling
        # destroys it by averaging each signed 2x2 Haar block to zero. Recover
        # the three signed high bands before any spatial averaging. This fixed
        # linear transform retains direction/phase without adding Base or
        # absolute-value/energy features. For 64 source channels it yields 192.
        detail_bands,_=model.lwpd.haar(pair[1])
        detail=torch.cat(detail_bands[1:],dim=1)
        absolute=detail.abs()
        energy=torch.sqrt(sum(band.square() for band in detail_bands[1:]))
    return {"e2_detail":{stage:feature_grid(detail,GRIDS[stage]) for stage in STAGES},
            "e2_absolute":{stage:feature_grid(absolute,GRIDS[stage]) for stage in STAGES},
            "e2_energy":{stage:feature_grid(energy,GRIDS[stage]) for stage in STAGES},
            "e2_shared":{stage:feature_grid(shared,GRIDS[stage]) for stage in STAGES}}


def extract_mars(model,paths,device,mode):
    gray,_=read_pair(paths)
    rgb=rgb_loader(paths)
    image=standardize_mars_rgb(rgb,gray,mode).to(device)
    with torch.no_grad():
        maps=mars_nchw(model(image),model.feature_info.channels())
    return {stage:feature_grid(maps[stage],GRIDS[stage]) for stage in STAGES}


def load_truncated_mars_checkpoint(model, checkpoint):
    """Load only requested SwinV2 stages without silently skipping needed keys.

    The official encoder-only MaRS checkpoint holds stages 0..3, while timm's
    FeatureListNet(out_indices=(0,1)) discards stages 2/3. Passing checkpoint_path
    to timm therefore raises Unexpected key(s) for the unused layers. Filter
    ONLY those unused stages, and load retained stages with strict=True.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and "state_dict" in saved:
        saved = saved["state_dict"]
    elif isinstance(saved, dict) and "model" in saved:
        saved = saved["model"]
    if not isinstance(saved, dict) or not saved:
        raise ValueError("MaRS checkpoint is not a nonempty state dictionary")
    # Official checkpoint keys are unprefixed; allow common DDP wrappers.
    if all(str(key).startswith("module.") for key in saved):
        saved = {key[len("module."):]: value for key, value in saved.items()}
    expected = model.state_dict()
    missing = sorted(set(expected).difference(saved))
    mismatched = sorted(
        key for key in set(expected).intersection(saved)
        if not isinstance(saved[key], torch.Tensor)
        or saved[key].shape != expected[key].shape
    )
    unused = sorted(set(saved).difference(expected))
    unexpected = [
        key for key in unused
        if not key.startswith(("layers_2.", "layers_3."))
    ]
    if missing or mismatched or unexpected:
        raise RuntimeError(
            "MaRS checkpoint incompatible with retained stages: "
            f"missing={missing[:8]} (n={len(missing)}), "
            f"shape_mismatch={mismatched[:8]} (n={len(mismatched)}), "
            f"unexpected={unexpected[:8]} (n={len(unexpected)})"
        )
    if not unused:
        raise RuntimeError(
            "Expected official full MaRS encoder stages 2/3 to be omitted "
            "by the two-stage feature extractor; verify checkpoint format."
        )
    keep = {key: saved[key] for key in expected}
    model.load_state_dict(keep, strict=True)
    print(
        f"MaRS encoder loaded: retained {len(keep)} tensors; "
        f"ignored {len(unused)} tensors from unused stages 2/3; "
        "no missing retained tensors.",
        flush=True,
    )


def load_mars(device,weight,random_model,seed=RNG_SEED):
    try:
        import timm
    except ImportError as error:
        raise RuntimeError(
            "Missing timm==1.0.15; run: .venv\\Scripts\\python.exe -m pip install timm==1.0.15"
        ) from error
    if timm.__version__!="1.0.15":
        raise RuntimeError(f"Use verified timm==1.0.15, got {timm.__version__}")
    if not random_model and not weight.is_file():
        raise FileNotFoundError(
            f"Download OFFICIAL mars_base_rgb_encoder_only.pth from "
            f"https://zenodo.org/records/17800805 into {weight}; "
            f"checkpoint is NOT downloaded or guessed automatically.")
    torch.manual_seed(seed)
    # Construct only the first two stages for the 128/64-pixel probes.
    # Do not let timm load the four-stage checkpoint with strict=True.
    model=timm.create_model(
        "swinv2_base_window8_256",pretrained=False,features_only=True,
        in_chans=3,img_size=512,out_indices=(0,1))
    if not random_model:
        load_truncated_mars_checkpoint(model,weight)
    return model.to(device).eval().requires_grad_(False)


def write(path,rows):
    if not rows:raise ValueError("Empty probe result")
    with path.open("x",encoding="utf-8-sig",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def checkpoint_path_for_metadata(path, root):
    """Python 3.8+ compatible relative-path display, never touch checkpoint."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def register(out,meta):
    relative=out.relative_to(ROOT).as_posix()
    registry_path=ROOT/"EXPERIMENT_REGISTRY.json"
    before=registry_path.read_text(encoding="utf-8")
    registry=json.loads(before)
    key="MARS-FROZEN-PRIOR-"+hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    if any(e["id"]==key for e in registry["experiments"]):
        raise FileExistsError(f"Already registered {key}")
    registry["experiments"].append(dict(
        id=key,method="Frozen E2 + MaRS incremental structure admission screen (v3)",
        task="engineering",status="diagnostic_fixed_subset",
        comparable_group=key,dataset="QXS-SAROPT",
        split="val",offset=0,sample_count=meta["val_count"],
        selected_scene_ids=meta["val_ids"],checkpoint=meta["e2_checkpoint"],
        checkpoint_sha256=meta["e2_sha256"],
        split_manifest=meta["manifest"],split_manifest_sha256=meta["manifest_sha256"],
        metric_profile=METRIC_PROFILE,
        output_policy=OUTPUT_POLICY,
        result_dir=relative,metadata_path=relative+"/run_metadata.json",
        metrics={"val_count":float(meta["val_count"])},
        metric_source=dict(type="json_paths",path=relative+"/summary.json",
                           paths={"val_count":["val_count"]}),
        notes="V3 incremental source-structure screen: E2 rich signed/absolute/energy baseline plus train-only PCA/residualized equal-dimensional branches, three random initializations, unaudited scenes and coherent-line proxies. Not fusion metrics or proof of fusion improvement."))
    registry["updated_at"]=datetime.now().strftime("%Y-%m-%d")
    registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    try:
        subprocess.run([sys.executable,"maintain_experiment_index.py","--check",
                        "--render"],cwd=ROOT,check=True)
    except Exception:
        registry_path.write_text(before,encoding="utf-8")
        raise


def validation_ids(manifest, audited, count):
    from utils.dataset_os import evenly_spaced, validate_manifest_splits
    validate_manifest_splits(manifest)
    source=[str(scene) for scene in manifest["splits"]["val"]]
    if not set(audited).issubset(source) or not len(audited)<=count<=len(source):
        raise ValueError("Validation must include all audited scenes within locked val")
    rest=[scene for scene in source if scene not in audited]
    extra=evenly_spaced(rest,count-len(audited)) if count>len(audited) else []
    selected=set(audited)|set(extra)
    return [scene for scene in source if scene in selected]


def method_branches(random_seeds):
    return {"e2_signed":None,"e2_rich":None,"e2_plus_mars":"mars",
            "e2_plus_shifted":"shifted","e2_plus_source":"source",
            **{f"e2_plus_random_{seed}":f"random_{seed}" for seed in random_seeds}}


def scene_features(paths, models, device, input_mode, runtime):
    """One extraction per backbone and scene; no multi-GB validation cache."""
    def measured(name, function):
        if device.type=="cuda":torch.cuda.synchronize(device)
        start=time.perf_counter()
        value=function()
        if device.type=="cuda":torch.cuda.synchronize(device)
        entry=runtime.setdefault(name,dict(calls=0,seconds=0.0))
        entry["calls"]+=1;entry["seconds"]+=time.perf_counter()-start
        return value
    gray,_=read_pair(paths)
    encoded=measured("e2",lambda:extract_e2(models["e2"],paths,device))
    mars=measured("mars",lambda:extract_mars(models["mars"],paths,device,input_mode))
    random={name:measured(name,lambda model=model:extract_mars(model,paths,device,input_mode))
            for name,model in models.items() if name.startswith("random_")}
    source=measured("source",lambda:source_structure_features(gray,GRIDS[0]))
    result={}
    for stage,size in GRIDS.items():
        signed=np.concatenate((encoded["e2_shared"][stage],encoded["e2_detail"][stage]))
        rich=np.concatenate((signed,encoded["e2_absolute"][stage],encoded["e2_energy"][stage]))
        result[stage]={"e2_signed":signed,"e2_rich":rich,"mars":mars[stage],
                       "shifted":shuffled_grid(mars[stage]),"source":resize_channels(source,size),
                       **{name:values[stage] for name,values in random.items()}}
    return result,edge_target(gray)


def fit_incremental_heads(sampled, targets, branches, dimensions):
    """Fit all projections/heads on identical TRAIN rows; no val access."""
    projections={}
    for branch in sorted(set(branches.values())-{None,"shifted"}):
        projections[branch]=fit_projection(sampled[branch],dimensions)
    # The shifted control has the very same transform as the aligned branch.
    projections["shifted"]=projections["mars"]
    projected={branch:project_rows(projections[branch],sampled[branch])
               for branch in sorted(set(branches.values())-{None})}
    residualizers=fit_residualizers(sampled["e2_rich"],projected)
    heads={}
    for method,branch in branches.items():
        if branch is None:
            x=sampled[method]
        else:
            residualizer=residualizers[branch]
            normalized=(sampled["e2_rich"]-residualizer["mean"])/residualizer["std"]
            residual=projected[branch]-residualizer["branch_mean"]-normalized@residualizer["coef"]
            x=np.concatenate((sampled["e2_rich"],residual),axis=1)
        stats=RidgeStats(x.shape[1])
        stats.add(x.T[:,:,None],targets[:,None],len(targets),RNG_SEED)
        head=stats.solve()
        if branch is not None:
            # Fold PCA + residualization into linear coefficients for fast
            # validation. This avoids an E2[512] -> branch[32] product per pixel.
            base_channels=sampled["e2_rich"].shape[1]
            slopes=head["coef"]/head["std"]
            branch_slopes=slopes[base_channels:]
            explained=residualizer["coef"]@branch_slopes
            projection=projections[branch]
            raw_slopes=projection["basis"]@branch_slopes
            head["effective_base_coef"]=slopes[:base_channels]-explained/residualizer["std"]
            head["effective_branch_coef"]=raw_slopes/projection["std"]
            head["effective_intercept"]=(head["intercept"]-np.dot(head["mean"],slopes)
                -np.dot(residualizer["branch_mean"],branch_slopes)
                +np.dot(residualizer["mean"]/residualizer["std"],explained)
                -np.dot(projection["mean"]/projection["std"],raw_slopes))
            head["residualizer"]=residualizer
        heads[method]=head
    return heads,projections


def predict_incremental_heads(features, branches, heads, projections):
    size=features["e2_rich"].shape[-1]
    rows={name:values.reshape(values.shape[0],-1).T for name,values in features.items()}
    out={}
    for method,branch in branches.items():
        if branch is None:
            prediction=predict_ridge(heads[method],rows[method].T[:,:,None])
        else:
            head=heads[method]
            prediction=(rows["e2_rich"]@head["effective_base_coef"]
                        +rows[branch]@head["effective_branch_coef"]+head["effective_intercept"])
        out[method]=np.clip(prediction,0,1).astype(np.float32).reshape(size,size)
    return out


def render_screen_report(summary):
    decision=summary["decision"]
    lines=["# MaRS 接入价值筛查（v3）","",f"判断：`{decision['status']}`。", "",
           decision["message"],"",f"训练探针：{summary['train_count']}图；验证：{summary['val_count']}图，"
           f"其中人工审计{summary['audit_count']}图，未人工审计{summary['unaudited_count']}图。", "",
           "两类验证分别报告；未人工审计集仅相对固定人工8图分离，不是外部独立数据集。", "",
           "## MaRS相对各对照的增量（未人工审计场景）", "",
           "| 尺度 | 对照 | 指标 | 共同场景数 | 相对改善 | 改善场景 | 配对改善95%区间 |",
           "|---|---|---|---:|---:|---:|---|"]
    for row in summary["pixel_comparisons"]:
        if row["cohort"]!="unaudited" or row["metric"] not in ("mse","high_edge_mae"):
            continue
        if row["n_scenes"]:
            lines.append(f"| {row['stage']} | {row['reference']} | {row['metric']} | {row['n_scenes']} | "
                         f"{row['relative_gain']:.2%} | {row['wins']} | "
                         f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] |")
    lines.extend(["","## 人工线段路径代理", "",
                  "| 尺度 | 类别 | 对照 | 指标 | 线段数 | 共同场景数 | 改善均值 |",
                  "|---|---|---|---|---:|---:|---:|"])
    for row in summary["line_comparisons"]:
        if row["reference"] not in ("e2_rich","e2_plus_source") or row["metric"] not in ("longest_run_fraction","max_gap_fraction"):
            continue
        if row["n_scenes"]:
            lines.append(f"| {row['stage']} | {row['cohort']} | {row['reference']} | {row['metric']} | "
                         f"{row['n_items']} | {row['n_scenes']} | {row['mean_gain']:.6g} |")
    lines.extend(["","所有改善值统一为正数更好；disconnected上响应下降为改善。", "",
                  "## 解释边界", ""]+["- "+text for text in summary["caveats"]])
    return "\n".join(lines)+"\n"


def run(args):
    os.chdir(ROOT)
    if not 1<=args.train_scenes<=256 or not 32<=args.samples_per_scene<=4096:
        raise ValueError("Invalid training readout sample count")
    if not 1<=args.augmentation_dim<=32 or not 0<=args.save_images<=8:
        raise ValueError("Require augmentation-dim 1..32 and save-images 0..8")
    if not args.random_seeds or len(set(args.random_seeds))!=len(args.random_seeds):
        raise ValueError("Provide distinct random seeds")
    weight=(ROOT/args.mars_weights).resolve()
    default=DEFAULT_OUT.parent/f"train{args.train_scenes}_val{args.val_scenes}_{args.input_mode}_d{args.augmentation_dim}"
    out=(ROOT/(args.output_dir or default)).resolve()
    out.relative_to(ROOT)
    if out.exists():raise FileExistsError(f"Output exists: {out}; choose a new --output-dir")
    if not weight.is_file():raise FileNotFoundError(f"Missing official optical checkpoint: {weight}")
    weight_sha=sha256_file(weight)
    torch.manual_seed(RNG_SEED);torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    archive,sample,labels,audit_paths=load_inputs(DEFAULT_LABELS,DATASET)
    visibility=load_visibility(DEFAULT_LABELS/"e2_visibility.csv",labels)
    manifest_path=ROOT/sample["source_manifest"]
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    val=validation_ids(manifest,audit_paths,args.val_scenes)
    train=train_ids(manifest,args.train_scenes)
    if set(train)&set(val):raise ValueError("Train/val scene leakage")
    ckpt=ROOT/archive["checkpoints"]["e2"]
    e2_sha=sha256_file(ckpt)
    if e2_sha!=archive["checkpoint_hashes"]["e2"]:
        raise ValueError("E2 checkpoint differs from archived audit")
    branches=method_branches(args.random_seeds)
    runtime={};models={};all_images=[];all_lines=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".mars-screen-",dir=out.parent) as tmp:
        pending=Path(tmp)/"results";pending.mkdir()
        model=OpticalBaseAnchoredE2()
        model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=False)["model"],strict=True)
        models["e2"]=model.to(device).eval().requires_grad_(False)
        models["mars"]=load_mars(device,weight,False)
        for seed in args.random_seeds:
            models[f"random_{seed}"]=load_mars(device,weight,True,seed)
        parameters={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
        samples={stage:{} for stage in STAGES};targets={stage:[] for stage in STAGES}
        for index,scene in enumerate(train):
            features,target=scene_features(source_paths(scene,manifest),models,device,args.input_mode,runtime)
            for stage,size in GRIDS.items():
                truth=cv2.resize(target,(size,size),interpolation=cv2.INTER_AREA).ravel()
                rng=np.random.default_rng(RNG_SEED+index*37+stage*173)
                idx=rng.choice(len(truth),min(args.samples_per_scene,len(truth)),replace=False)
                targets[stage].append(truth[idx])
                for name,feature in features[stage].items():
                    rows=feature.reshape(feature.shape[0],-1).T[idx].copy()
                    samples[stage].setdefault(name,[]).append(rows)
            print(f"feature sampling train {index+1}/{len(train)} scene={scene}",flush=True)
        heads={};projections={};saved_arrays={};head_info={};projection_info={}
        for stage in STAGES:
            data={name:np.concatenate(values) for name,values in samples[stage].items()}
            heads[stage],projections[stage]=fit_incremental_heads(
                data,np.concatenate(targets[stage]),branches,args.augmentation_dim)
            for name,head in heads[stage].items():
                head_info[f"stage{stage}:{name}"]={"train_pixels":head["count"],
                    "channels":len(head["coef"]),"alpha":head["alpha"]}
                for field in ("mean","std","coef","intercept"):
                    saved_arrays[f"stage{stage}__{name}__{field}"]=head[field]
                if "residualizer" in head:
                    head_info[f"stage{stage}:{name}"]["e2_conditioning_rank"]=head["residualizer"]["e2_effective_rank"]
                    for field in ("effective_base_coef","effective_branch_coef","effective_intercept"):
                        saved_arrays[f"stage{stage}__{name}__{field}"]=head[field]
                    for field in ("mean","std","branch_mean","coef"):
                        saved_arrays[f"stage{stage}__{name}__condition_{field}"]=head["residualizer"][field]
            for name,projection in projections[stage].items():
                projection_info[f"stage{stage}:{name}"]={key:projection[key] for key in ("explained_variance","effective_rank")}
                for field in ("mean","std","basis"):
                    saved_arrays[f"stage{stage}__projection_{name}__{field}"]=projection[field]
        del samples,targets,data
        for index,scene in enumerate(val):
            features,target=scene_features(source_paths(scene,manifest),models,device,args.input_mode,runtime)
            cohort="audited" if scene in audit_paths else "unaudited"
            for stage,size in GRIDS.items():
                predictions=predict_incremental_heads(features[stage],branches,heads[stage],projections[stage])
                truth=cv2.resize(target,(size,size),interpolation=cv2.INTER_AREA)
                for method,prediction in predictions.items():
                    all_images.append(dict(scene=scene,cohort=cohort,stage=stage,method=method,
                                           **pixel_metrics(prediction,truth)))
                    map256=cv2.resize(prediction,(256,256),interpolation=cv2.INTER_LINEAR)
                    if index<args.save_images or scene in list(audit_paths)[:args.save_images]:
                        folder=pending/"previews"/scene;folder.mkdir(parents=True,exist_ok=True)
                        cv2.imwrite(str(folder/f"{method}_stage{stage}.png"),
                                    (np.clip(map256,0,1)*255).round().astype(np.uint8))
                    for row in labels:
                        if row["scene"]!=scene:continue
                        group=(visibility[(scene,row["id"])] if row["label"]=="connected"
                               else "disconnected" if row["label"]=="disconnected" else "uncertain")
                        if group not in ("missing","present","disconnected"):continue
                        all_lines.append(dict(scene=scene,id=row["id"],stage=stage,
                            method=method,cohort=group,**line_score(map256,row)))
            print(f"incremental validation {index+1}/{len(val)} scene={scene}",flush=True)
        references=[name for name in branches if name not in ("e2_signed","e2_plus_mars")]
        bootstrap=DECISION_RULES["bootstrap_replicates"]
        pixel_pairs=paired_evidence(all_images,list(branches),"e2_plus_mars",references,
            {"mse":False,"mae":False,"high_edge_mae":False,"r2":True},bootstrap=bootstrap)
        line_pairs=paired_evidence(all_lines,list(branches),"e2_plus_mars",references,
            {"path_mean":True,"path_coverage":True,"longest_run_fraction":True,"max_gap_fraction":False},bootstrap=bootstrap)
        representation=paired_evidence(all_images,list(branches),"e2_rich",["e2_signed"],
                                      {"mse":False,"high_edge_mae":False},bootstrap=bootstrap)
        decision=admission_decision(pixel_pairs,line_pairs,
            [name for name in branches if name.startswith("e2_plus_random_")],len(train),args.input_mode)
        summary=dict(metric_profile=METRIC_PROFILE,train_count=len(train),val_count=len(val),
            audit_count=len(audit_paths),unaudited_count=len(set(val)-set(audit_paths)),
            methods=list(branches),pixel_comparisons=pixel_pairs,line_comparisons=line_pairs,
            representation_checks=representation,decision=decision,
            caveats=[
                "这是冻结的光学源梯度可读性及互补性筛查，没有生成新融合图，也没有测量融合质量收益。",
                "E2 rich包含shared、带符号LH/HL/HH、逐方向绝对值和三方向能量；幅值特征在池化前计算。E2 Detail含SAR门控条件。",
                "所有增量分支均为训练集PCA投影的同维通道；有效秩可能不同。移位MaRS与对齐MaRS共用同一训练投影。",
                "增量分支先用训练集E2 rich做无标签线性残差化，避免重复已有特征通过改变Ridge正则化产生假增益。",
                "低成本源结构与目标接近；其像素MSE只作报告，人工线段对照参与门槛。",
                "未人工审计场景仅与固定人工8图分离，仍来自既有pilot validation，不代表外部独立验证。",
                "人工missing仅3条；路径连续性仍是走廊内代理，不能等同真实拓扑恢复。",
                "线性读出未通过不能排除其他非线性Adapter、其他投影维度或MaRS深层特征的价值。",
                "两个尺度须同时通过；正向结论仅允许进入小规模受控Adapter对照。RGB模式含额外信息。",
                "协议不同于v1/v2，不得混用；v1结果已作废。冻结E2及MaRS权重均复用，不重训融合基线。"])
        np.savez_compressed(pending/"readout_and_projection.npz",**saved_arrays)
        meta=dict(val_count=len(val),val_ids=val,train_count=len(train),train_ids=train,
            audited_val_ids=list(audit_paths),unaudited_val_ids=[s for s in val if s not in audit_paths],
            metric_profile=METRIC_PROFILE,output_policy=OUTPUT_POLICY,input_mode=args.input_mode,
            random_seeds=args.random_seeds,augmentation_dim=args.augmentation_dim,
            decision_rules=DECISION_RULES,samples_per_train_scene=args.samples_per_scene,
            ridge_prediction="((x-train_mean)/train_std) @ coef + train_target_mean; clip [0,1]",
            pca_fit_split="train_only; shared sample indices across methods",
            branch_conditioning="train-only OLS on standardized E2 rich; eigenvalue relative cutoff 1e-8; no target/val labels",
            source_labels_sha256=sha256_file(DEFAULT_LABELS/"source_labels.csv"),
            visibility_sha256=sha256_file(DEFAULT_LABELS/"e2_visibility.csv"),
            manifest=sample["source_manifest"],manifest_sha256=sha256_file(manifest_path),
            e2_checkpoint=archive["checkpoints"]["e2"],e2_sha256=e2_sha,
            mars_checkpoint=checkpoint_path_for_metadata(weight,ROOT),mars_sha256=weight_sha,
            model="swinv2_base_window8_256",model_source="https://github.com/WanderRainy/MaRS",
            timm_version="1.0.15",input_size=512,stages=list(STAGES),resolution=GRIDS,
            torch_version=torch.__version__,numpy_version=np.__version__,opencv_version=cv2.__version__,
            preprocessing="raw 0..255; official RGB mean/std; gray replicates GF2 Y; bicubic 256->512",
            seed=RNG_SEED,device=str(device),script_sha256=sha256_file(Path(__file__)),
            evidence_module_sha256=sha256_file(ROOT/"mars_probe_evidence.py"),
            methods=list(branches),readout_fits=head_info,projection_fits=projection_info,
            readout_archive="readout_and_projection.npz",
            readout_archive_sha256=sha256_file(pending/"readout_and_projection.npz"),
            runtime=runtime,backbone_parameters=parameters,
            bootstrap_unit="scene; per-metric finite intersection across ALL methods; 2000 paired resamples")
        for name,rows in (("pixel_probe",all_images),("line_probe",all_lines),
                          ("pixel_comparisons",pixel_pairs),("line_comparisons",line_pairs),
                          ("representation_checks",representation)):
            write(pending/(name+".csv"),rows)
        for name,value in (("summary",summary),("run_metadata",meta),("decision",decision)):
            (pending/(name+".json")).write_text(json.dumps(value,indent=2,ensure_ascii=False,
                                                         allow_nan=False),encoding="utf-8")
        (pending/"接入价值筛查.md").write_text(render_screen_report(summary),encoding="utf-8")
        pending.rename(out)
    register(out,meta)
    print(json.dumps(dict(output=str(out),decision=decision),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mars-weights",type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument("--train-scenes",type=int,default=32)
    parser.add_argument("--val-scenes",type=int,default=64)
    parser.add_argument("--samples-per-scene",type=int,default=512)
    parser.add_argument("--augmentation-dim",type=int,default=32)
    parser.add_argument("--random-seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--input-mode",choices=("gray","rgb"),default="gray")
    parser.add_argument("--save-images",type=int,default=4)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
