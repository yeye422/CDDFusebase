"""Frozen MaRS-Base optical feature probe against archived optical-Base E2.

No Adapter training, no fusion changes, no test set. Ridge *readout* trained
only on locked pilot train scenes to predict source Optical-Y gradient magnitude.
Evaluation uses disjoint eight manually audited STG validation scenes, including
E2-missing connected edges and disconnected hard-negative links.

Compare same 1/4 & 1/8 original-image spatial grids:
  E2 shared / E2 optical Detail,
  MaRS optical pretrained (Y->RGB replicated),
  same-architecture random SwinV2,
  spatially shifted pretrained features.
Optionally evaluate MaRS on real optical RGB (changes input information).

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

ROOT=Path(__file__).resolve().parent
DEFAULT_OUT=Path("test_result/mars_frozen_prior_probe_v1/train32_val8_gray_seed42")
DEFAULT_WEIGHTS=Path("models/external/mars_base_rgb_encoder_only.pth")
STAGES=(0,1)
GRIDS={0:128,1:64}
RNG_SEED=20261118
PRETRAIN_MEAN=np.array([87.01,91.52,83.51],np.float32)
PRETRAIN_STD=np.array([62.66,54.58,53.11],np.float32)
METHODS=("e2_shared","e2_detail","mars_gray","mars_random","mars_shifted")
EXTRA_METHODS=("mars_rgb","mars_rgb_random")


def edge_target(gray):
    """Source-derived unsigned optical edge label, not ground truth of fusion."""
    x=gray.astype(np.float32)/255.0
    smooth=cv2.GaussianBlur(x,(0,0),1.25)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    return np.clip(np.sqrt(gx*gx+gy*gy)/0.25,0,1).astype(np.float32)


def feature_grid(x,size):
    """Area-match E2 features to the exact MaRS stage resolution."""
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
        intercept=self.sy/self.n-np.dot(mean/std,w)
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
                line_p25=float(np.quantile(best,0.25)))


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
    return {"e2_detail":{stage:feature_grid(pair[1],GRIDS[stage]) for stage in STAGES},
            "e2_shared":{stage:feature_grid(shared,GRIDS[stage]) for stage in STAGES}}


def extract_mars(model,paths,device,mode):
    gray,_=read_pair(paths)
    rgb=rgb_loader(paths)
    image=standardize_mars_rgb(rgb,gray,mode).to(device)
    with torch.no_grad():
        maps=mars_nchw(model(image),model.feature_info.channels())
    return {stage:feature_grid(maps[stage],GRIDS[stage]) for stage in STAGES}


def load_mars(device,weight,random_model):
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
    torch.manual_seed(RNG_SEED)
    model=timm.create_model(
        "swinv2_base_window8_256",pretrained=False,features_only=True,
        in_chans=3,img_size=512,out_indices=(0,1),
        **({} if random_model else {"checkpoint_path":str(weight)}))
    return model.to(device).eval().requires_grad_(False)


def fit_and_score(method,model,manifest,train,paths,labels,visibility,
                  samples,device,output,save_images):
    """Train linear readouts on train only; every test image strictly held out."""
    stages={stage:None for stage in STAGES}
    family="e2" if method.startswith("e2") else "mars"
    def get_maps(pair):
        return (extract_e2(model,pair,device)[method] if family=="e2"
                else extract_mars(model,pair,device,
                                  "rgb" if "_rgb" in method else "gray"))
    for i,scene in enumerate(train):
        optical,sar=source_paths(scene,manifest)
        g,_=read_pair((optical,sar))
        features=get_maps((optical,sar))
        ground=edge_target(g)
        for stage in STAGES:
            feat=features[stage]
            if method=="mars_shifted":
                feat=shuffled_grid(feat)
            target=cv2.resize(ground,(GRIDS[stage],GRIDS[stage]),interpolation=cv2.INTER_AREA)
            if stages[stage] is None:
                stages[stage]=RidgeStats(feat.shape[0])
            stages[stage].add(feat,target,samples,RNG_SEED+i*37+stage*173)
        if (i+1)%8==0 or i+1==len(train):
            print(f"{method} ridge train {i+1}/{len(train)}",flush=True)
    fitted={stage:stages[stage].solve() for stage in STAGES}
    image_rows,line_rows=[],[]
    for i,(scene,pair) in enumerate(paths.items()):
        gray,_=read_pair(pair[:2])
        reference=edge_target(gray)
        features=get_maps(pair[:2])
        probes=[row for row in labels if row["scene"]==scene]
        for stage in STAGES:
            feat=features[stage]
            if method=="mars_shifted":
                feat=shuffled_grid(feat)
            pred=predict_ridge(fitted[stage],feat)
            truth=cv2.resize(reference,(GRIDS[stage],GRIDS[stage]),interpolation=cv2.INTER_AREA)
            values=pixel_metrics(pred,truth)
            image_rows.append(dict(scene=scene,stage=stage,method=method,**values))
            map256=cv2.resize(pred,(256,256),interpolation=cv2.INTER_LINEAR)
            if i<save_images:
                preview=output/"previews"/str(scene)
                preview.mkdir(parents=True,exist_ok=True)
                png=(np.clip(map256,0,1)*255).round().astype(np.uint8)
                cv2.imwrite(str(preview/f"{method}_stage{stage}.png"),png)
            for row in probes:
                key=(scene,row["id"])
                cohort=("missing" if row["label"]=="connected" and visibility[key]=="missing"
                        else "present" if row["label"]=="connected" and visibility[key]=="present"
                        else "uncertain" if row["label"]=="connected"
                        else "disconnected" if row["label"]=="disconnected" else "uncertain")
                if cohort=="uncertain":
                    continue
                line_rows.append(dict(scene=scene,id=row["id"],stage=stage,
                    method=method,cohort=cohort,**line_score(map256,row)))
        print(f"{method} val {i+1}/{len(paths)} scene={scene}",flush=True)
    return image_rows,line_rows,{f"stage{stage}":dict(
        train_pixels=fitted[stage]["count"],channels=len(fitted[stage]["coef"]),
        alpha=fitted[stage]["alpha"]) for stage in STAGES}


def write(path,rows):
    if not rows:raise ValueError("Empty probe result")
    with path.open("x",encoding="utf-8-sig",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_lines(lines):
    out={}
    for stage in STAGES:
        for method in sorted({r["method"] for r in lines}):
            chosen=[r for r in lines if r["stage"]==stage and r["method"]==method]
            def summary(cohort):
                values=[r["line_mean"] for r in chosen if r["cohort"]==cohort]
                return dict(n=len(values),mean=float(np.mean(values)) if values else None)
            missing=summary("missing")
            negative=summary("disconnected")
            out[f"stage{stage}:{method}"]=dict(missing=missing,
                disconnected=negative,present=summary("present"),
                missing_minus_disconnected=(missing["mean"]-negative["mean"]
                   if missing["n"] and negative["n"] else None))
    return out


def register(out,meta):
    relative=out.relative_to(ROOT).as_posix()
    registry_path=ROOT/"EXPERIMENT_REGISTRY.json"
    before=registry_path.read_text(encoding="utf-8")
    registry=json.loads(before)
    key="MARS-FROZEN-PRIOR-"+hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    if any(e["id"]==key for e in registry["experiments"]):
        raise FileExistsError(f"Already registered {key}")
    registry["experiments"].append(dict(
        id=key,method="Frozen MaRS vs E2 optical edge linear probe",
        task="engineering",status="diagnostic_fixed_subset",
        comparable_group=key,dataset="QXS-SAROPT",
        split="val",offset=0,sample_count=meta["val_count"],
        selected_scene_ids=meta["val_ids"],checkpoint=meta["e2_checkpoint"],
        checkpoint_sha256=meta["e2_sha256"],
        split_manifest=meta["manifest"],split_manifest_sha256=meta["manifest_sha256"],
        metric_profile="mars_optical_source_edge_probe_v1",
        output_policy="frozen_feature_ridge_train_source_edge_target",
        result_dir=relative,metadata_path=relative+"/run_metadata.json",
        metrics={"val_count":float(meta["val_count"])},
        metric_source=dict(type="json_paths",path=relative+"/summary.json",
                           paths={"val_count":["val_count"]}),
        notes="Source-gradient self-supervised linear readout, not fusion metrics. MaRS official optical checkpoint hashed in metadata."))
    registry["updated_at"]=datetime.now().strftime("%Y-%m-%d")
    registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    try:
        subprocess.run([sys.executable,"maintain_experiment_index.py","--check",
                        "--render"],cwd=ROOT,check=True)
    except Exception:
        registry_path.write_text(before,encoding="utf-8")
        raise


def run(args):
    os.chdir(ROOT)
    weight=(ROOT/args.mars_weights).resolve()
    out=(ROOT/(args.output_dir or DEFAULT_OUT)).resolve()
    out.relative_to(ROOT)
    if out.exists():raise FileExistsError(f"Output exists: {out}")
    if not weight.is_file():
        raise FileNotFoundError(
            f"Missing official MaRS optical checkpoint {weight}; "
            "download from https://zenodo.org/records/17800805")
    if not 1<=args.train_scenes<=256 or not 32<=args.samples_per_scene<=4096:
        raise ValueError("Invalid ridge sample count")
    if not 0<=args.save_images<=8:raise ValueError("Invalid save image count")
    torch.manual_seed(RNG_SEED)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    archive,sample,labels,paths=load_inputs(DEFAULT_LABELS,DATASET)
    visibility=load_visibility(DEFAULT_LABELS/"e2_visibility.csv",labels)
    manifest_path=ROOT/sample["source_manifest"]
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train=train_ids(manifest,args.train_scenes)
    if set(train)&set(paths):
        raise ValueError("Train/val scene leakage")
    ckpt=ROOT/archive["checkpoints"]["e2"]
    if sha256_file(ckpt)!=archive["checkpoint_hashes"]["e2"]:
        raise ValueError("E2 checkpoint differs from archived visual audit")
    weight_sha=sha256_file(weight)
    all_images,all_lines,heads=[],[],{}
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".mars-prior-",dir=out.parent) as tmp:
        pending=Path(tmp)/"results"
        pending.mkdir()
        model=OpticalBaseAnchoredE2()
        saved=torch.load(ckpt,map_location="cpu",weights_only=False)
        model.load_state_dict(saved["model"],strict=True)
        model.to(device).eval().requires_grad_(False)
        for method in ("e2_shared","e2_detail"):
            img,lines,info=fit_and_score(method,model,manifest,train,paths,labels,
                visibility,args.samples_per_scene,device,pending,args.save_images)
            all_images+=img;all_lines+=lines;heads[method]=info
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
        model=load_mars(device,weight,False)
        for method in (("mars_gray","mars_shifted","mars_rgb") if args.include_rgb
                       else ("mars_gray","mars_shifted")):
            img,lines,info=fit_and_score(method,model,manifest,train,paths,labels,
                visibility,args.samples_per_scene,device,pending,args.save_images)
            all_images+=img;all_lines+=lines;heads[method]=info
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
        model=load_mars(device,weight,True)
        for method in (("mars_random","mars_rgb_random") if args.include_rgb
                       else ("mars_random",)):
            img,lines,info=fit_and_score(method,model,manifest,train,paths,labels,
                visibility,args.samples_per_scene,device,pending,args.save_images)
            all_images+=img;all_lines+=lines;heads[method]=info
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
        summary=dict(val_count=len(paths),
            image_comparisons=summary_rows(all_images,("stage","scene"),
                  ("mse","mae","high_edge_mae","r2"),"e2_detail"),
            line_comparisons=summary_rows(all_lines,("stage","scene","id"),
                  ("line_mean","line_continuity","line_p25"),"e2_detail"),
            line_groups=summarize_lines(all_lines),
            caveats=[
                "Ridge predicts Optical source gradient only, not fusion quality or true edges.",
                "MaRS original pretraining used true RGB whereas E2 consumed Y only; gray3 and RGB results are separate.",
                "The released encoder-only checkpoints do not instantiate MMA fusion.",
                "MaRS 512-pixel input is upsampled from QXS 256; stages are compared at identical 128/64 original-grid resolution.",
                "Random and spatially-shifted controls probe whether any gain requires pretrained and spatially aligned representation.",
                "Only 8 archived visual scenes and very few E2-missing manually confirmed optical segments.",
                "Selecting the best stage post hoc is not a validated architecture improvement."
            ])
        meta=dict(val_count=len(paths),val_ids=list(paths),train_count=len(train),
            train_ids=train,source_labels_sha256=sha256_file(DEFAULT_LABELS/"source_labels.csv"),
            visibility_sha256=sha256_file(DEFAULT_LABELS/"e2_visibility.csv"),
            manifest=sample["source_manifest"],manifest_sha256=sha256_file(manifest_path),
            e2_checkpoint=archive["checkpoints"]["e2"],e2_sha256=sha256_file(ckpt),
            mars_checkpoint=str(weight.relative_to(ROOT)) if weight.is_relative_to(ROOT) else str(weight),
            mars_sha256=weight_sha,model="swinv2_base_window8_256",
            model_source="https://github.com/WanderRainy/MaRS",
            model_license="academic research only; inspect upstream terms",
            timm_version="1.0.15",input_size=512,stages=list(STAGES),
            resolution={str(x):GRIDS[x] for x in STAGES},
            preprocessing="RGB raw 0..255; normalize by official MaRS RGB mean/std; gray mode replicates GF2 Y",
            samples_per_train_scene=args.samples_per_scene,
            seed=RNG_SEED,device=str(device),
            script_sha256=sha256_file(Path(__file__)),
            methods=list(heads),readout_fits=heads)
        write(pending/"pixel_probe.csv",all_images)
        write(pending/"line_probe.csv",all_lines)
        write(pending/"nominal_pairwise.csv",summary["image_comparisons"])
        (pending/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,
                                            allow_nan=False),encoding="utf-8")
        (pending/"run_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False,
                                                 allow_nan=False),encoding="utf-8")
        pending.rename(out)
    register(out,meta)
    print(json.dumps({"output":str(out),"val_count":len(paths),
                      "methods":list(heads),"mars_sha256":weight_sha},
                     ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mars-weights",type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument("--train-scenes",type=int,default=32)
    parser.add_argument("--samples-per-scene",type=int,default=512)
    parser.add_argument("--include-rgb",action="store_true")
    parser.add_argument("--save-images",type=int,default=4)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
