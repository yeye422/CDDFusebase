"""Basis-independent local relations and frozen E2 routing directions.

This is a fixed diagnostic operator, not the released MaRS MMA or a trained
confidence estimator. No raw RGB/SAR channel-space cosine is compared.
"""
import numpy as np
import torch
import torch.nn.functional as F

STAGES=(2,3)
ARMS=("dual_relation","independent_relations","optical_relation","sar_relation",
      "shifted_sar","permuted_relation","source_relation")
OFFSETS=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))


class UninformativeScene(ValueError):
    """Shared, quality-blind scene exclusion; never remove one control only."""


def sar_input(gray,stats):
    if gray.ndim!=2 or not 0<=float(gray.min())<=float(gray.max())<=255:
        raise ValueError("Expected QXS display-domain SAR in 0..255")
    import cv2
    if not np.isfinite(stats["std"]) or stats["std"]<1e-3:
        raise ValueError("Degenerate TRAIN SAR normalization")
    x=gray.astype(np.float32)
    x=np.where(x>0,x,stats["mean"])
    x=np.clip(cv2.resize(x,(512,512),interpolation=cv2.INTER_CUBIC),0,255)
    x=(x-stats["mean"])/stats["std"]
    return torch.from_numpy(x.copy())[None,None]


def deep_maps(outputs):
    if len(outputs)!=2:raise ValueError("Require MaRS stages 2/3")
    maps={}
    for stage,x,channels,size in zip(STAGES,outputs,(512,1024),(32,16)):
        if x.ndim!=4 or x.shape[0]!=1:raise ValueError("Expected singleton features")
        if x.shape[1:]==(size,size,channels):x=x.permute(0,3,1,2)
        if x.shape[1:]!=(channels,size,size):raise ValueError("Unexpected MaRS feature layout")
        if not torch.isfinite(x).all():raise FloatingPointError("Nonfinite MaRS features")
        maps[stage]=x[0].detach().float().cpu().numpy().copy()
    return maps


def grid_mask(valid,size):
    import cv2
    return cv2.resize(valid.astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)>=0.99


def neighborhood_relations(feature,center,valid):
    """Within-modality centered feature cosines, invariant to orthogonal basis."""
    x=np.asarray(feature,np.float64)-np.asarray(center,np.float64)[:,None,None]
    if x.ndim!=3 or valid.shape!=x.shape[1:] or not np.isfinite(x).all():
        raise ValueError("Invalid centered features/mask")
    norm=np.sqrt(np.sum(x*x,axis=0))
    unit=x/np.maximum(norm,1e-10)[None]
    keep=valid.copy()&(norm>1e-8)
    height,width=valid.shape
    yy,xx=np.indices(valid.shape)
    relations=[]
    for dy,dx in OFFSETS:
        neighbor=np.roll(unit,(-dy,-dx),axis=(1,2))
        inside=(yy+dy>=0)&(yy+dy<height)&(xx+dx>=0)&(xx+dx<width)
        keep&=inside&np.roll(valid&(norm>1e-8),(-dy,-dx),axis=(0,1))
        relations.append(np.sum(unit*neighbor,axis=0))
    signature=np.clip(np.stack(relations),-1,1)
    boundary=1-signature.mean(axis=0)
    return signature,boundary,keep


def agreement(left,right):
    if left.shape!=right.shape or left.shape[0]!=len(OFFSETS):
        raise ValueError("Relations need matching displacement coordinates")
    a=left-left.mean(axis=0,keepdims=True)
    b=right-right.mean(axis=0,keepdims=True)
    denominator=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
    return np.clip(np.sum(a*b,axis=0)/np.maximum(denominator,1e-10),-1,1)


def fit_relation_calibration(frames):
    """TRAIN feature centroids and 5/95 percentiles; no edge/quality labels."""
    result={}
    for family in ("mars","source"):
        result[family]={}
        for stage in STAGES:
            sums=[None,None];count=0
            for frame in frames:
                valid=frame["valid"][stage]
                number=int(valid.sum())
                if not number:continue
                for modality in (0,1):
                    value=frame[family][stage][modality][:,valid].astype(np.float64).sum(axis=1)
                    sums[modality]=value if sums[modality] is None else sums[modality]+value
                count+=number
            if count<32:raise ValueError("Insufficient TRAIN valid feature tokens")
            centers=[value/count for value in sums]
            values=[[],[]]
            for frame in frames:
                for modality in (0,1):
                    _,boundary,valid=neighborhood_relations(
                        frame[family][stage][modality],centers[modality],frame["valid"][stage])
                    values[modality].extend(boundary[valid].tolist())
            item={"centers":centers,"train_tokens":count,"ranges":[]}
            for value in values:
                if len(value)<32:raise ValueError("Too few TRAIN relation neighborhoods")
                low,high=np.quantile(value,[0.05,0.95])
                if high-low<1e-8:raise ValueError("TRAIN relation feature is degenerate")
                item["ranges"].append((float(low),float(high)))
            result[family][stage]=item
    return result


def relation_gates(frame,calibration,stage,seed=20261118):
    data={};shared=None
    for family in ("mars","source"):
        info=calibration[family][stage]
        signatures=[];boundaries=[]
        for modality in (0,1):
            sig,bound,mask=neighborhood_relations(frame[family][stage][modality],
                                                 info["centers"][modality],frame["valid"][stage])
            low,high=info["ranges"][modality]
            signatures.append(sig);boundaries.append(np.clip((bound-low)/(high-low),0,1))
            shared=mask if shared is None else shared&mask
        data[family]=(signatures,boundaries)
    size=frame["valid"][stage].shape[0]
    shift=(max(1,size//3),max(1,size//4))
    feature=np.roll(frame["mars"][stage][1],shift,axis=(1,2))
    shifted_mask=np.roll(frame["valid"][stage],shift,axis=(0,1))
    info=calibration["mars"][stage]
    ss,sb,sm=neighborhood_relations(feature,info["centers"][1],shifted_mask)
    shared&=sm
    if shared.sum()<16:raise UninformativeScene("fewer than 16 common valid relation neighborhoods")
    (osig,ssig),(ob,sb0)=data["mars"]
    rho=agreement(osig,ssig)
    wrong=agreement(osig,ss)
    low,high=info["ranges"][1]
    shifted_boundary=np.clip((sb-low)/(high-low),0,1)
    (co,cs),(bo,bs)=data["source"]
    cheap_agreement=agreement(co,cs)
    # Agreement changes the gain by at most +/-50%; modality-specific
    # boundary support is retained even when relations disagree.
    raw={
        "dual_relation":np.stack((ob*(1+0.5*rho),sb0*(1+0.5*rho))),
        "independent_relations":np.stack((ob,sb0)),
        "optical_relation":np.stack((ob,ob)),
        "sar_relation":np.stack((sb0,sb0)),
        "shifted_sar":np.stack((ob*(1+0.5*wrong),shifted_boundary*(1+0.5*wrong))),
        "source_relation":np.stack((bo*(1+0.5*cheap_agreement),bs*(1+0.5*cheap_agreement))),
    }
    permutation=np.random.default_rng(seed+stage).permutation(int(shared.sum()))
    shuffled=raw["dual_relation"].copy()
    shuffled[:,shared]=shuffled[:,shared][:,permutation]
    raw["permuted_relation"]=shuffled
    gates={}
    for name in ARMS:
        gate=raw[name]
        gate=(gate-gate[:,shared].mean(axis=1)[:,None,None])*shared[None]
        if float(np.sqrt(np.mean(gate[:,shared]**2)))<1e-8:
            raise UninformativeScene("constant routing field in "+name)
        gates[name]=gate.astype(np.float32)
    return gates,shared,dict(paired_agreement=float(rho[shared].mean()),
        shifted_agreement=float(wrong[shared].mean()),valid_tokens=int(shared.sum()),
        optical_boundary_mean=float(ob[shared].mean()),sar_boundary_mean=float(sb0[shared].mean()))


def routing_direction(ctx,gate,grid_valid,pixel_valid):
    """Native E2 values only; remove spatial DC and normalize joint RMS."""
    device=ctx["op"].device;size=ctx["op"].shape[-2:]
    field=F.interpolate(torch.from_numpy(gate.copy())[None].to(device),size=size,
                        mode="bilinear",align_corners=False)
    valid=F.interpolate(torch.from_numpy(grid_valid.astype(np.float32))[None,None].to(device),
                        size=size,mode="nearest")*pixel_valid.to(device).float()
    if float(valid.sum())<16:raise UninformativeScene("too few native valid pixels")
    deltas=[]
    for index,key in enumerate(("op","sp")):
        value=ctx[key]*field[:,index:index+1]
        mean=(value*valid).sum((-2,-1),keepdim=True)/valid.sum().clamp_min(1)
        deltas.append((value-mean)*valid)
    rms=torch.cat(deltas,dim=1).float().square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms)<1e-10:
        raise UninformativeScene("degenerate projected routing direction")
    return tuple(value/rms for value in deltas)


def routed_detail(model,ctx,direction,amplitude,sign):
    if sign not in (-1,1) or amplitude<0:raise ValueError("Invalid frozen route intervention")
    op=ctx["op"]+sign*amplitude*direction[0]
    sp=ctx["sp"]+sign*amplitude*direction[1]
    joint=torch.cat((op,sp),dim=1)
    scales=[block(joint) for block in model.detail_fusion.scale_branches]
    return model.detail_fusion.output(torch.cat((op,sp,*scales,ctx["bp"]),dim=1))
