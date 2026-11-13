"""S2 pre-training diagnostic: spatial heterogeneity of frozen E2 Detail responses.

Copy into the CDDFusebase repository root. It REQUIRES local E2 checkpoint and
QXS-SAROPT validation images; no training, parameter changes, or test usage.

Usage (Windows CMD, repository root):
.venv\\Scripts\\python.exe s2_region_diagnostic.py --scene-ids 4 44 1267 13273 \
  --tile-size 64 --output-dir test_result\\s2_region_diag_v1\\fixed4_tile64

The measured quantity is local finite-difference sensitivity of the EXISTING
fusion model, not source information utility, a true structure label, or proof
that a mixture of experts will outperform a single CNN.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_CHECKPOINT = Path('models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth')
DEFAULT_MANIFEST = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
DEFAULT_DATA_ROOT = Path('../datasets/QXSLAB_SAROPT')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def numeric(value):
    v = float(value)
    return v if math.isfinite(v) else None


def rms(tensor):
    return torch.sqrt(torch.mean(tensor.float().square())).item()


def local_response(positive, negative, roi, step, feat_region_rms):
    """Both signs use the same local relative gain; result is per-unit-feature RMS.

    Feature RMS is across channels and spatial pixels INSIDE the region.
    Output RMS is across the single-channel image in the same region.
    These are sensitivity normalizations, NOT physical or information units.
    """
    y0, y1, x0, x1 = roi
    derivative = (positive.float() - negative.float()) / (2. * step)
    local_rms = rms(derivative[..., y0:y1, x0:x1])
    whole_rms = rms(derivative)
    external = derivative.clone()
    external[..., y0:y1, x0:x1] = 0
    h, w = derivative.shape[-2:]
    local_area_fraction = (y1-y0)*(x1-x0)/(h*w)
    if local_area_fraction < 1:
        external_rms = float(torch.sqrt(external.square().mean() / (1-local_area_fraction)))
    else:
        external_rms = 0.
    return {
        'output_local_derivative_rms': numeric(local_rms),
        'output_global_derivative_rms': numeric(whole_rms),
        'output_outside_derivative_rms': numeric(external_rms),
        'gain_per_feature_rms': numeric(local_rms/max(feat_region_rms, 1e-8)),
        'feature_region_rms': numeric(feat_region_rms),
    }


def gradient_stats(optical_u8, sar_u8, roi):
    y0,y1,x0,x1 = roi
    o = optical_u8.astype(np.float32) / 255.
    s = sar_u8.astype(np.float32) / 255.
    gx_o = cv2.Sobel(o,cv2.CV_32F,1,0,ksize=3)
    gy_o = cv2.Sobel(o,cv2.CV_32F,0,1,ksize=3)
    gx_s = cv2.Sobel(s,cv2.CV_32F,1,0,ksize=3)
    gy_s = cv2.Sobel(s,cv2.CV_32F,0,1,ksize=3)
    a = np.stack((gx_o[y0:y1,x0:x1],gy_o[y0:y1,x0:x1]),axis=-1)
    b = np.stack((gx_s[y0:y1,x0:x1],gy_s[y0:y1,x0:x1]),axis=-1)
    dot = np.sum(a*b, axis=-1)
    aa = np.linalg.norm(a,axis=-1)
    bb = np.linalg.norm(b,axis=-1)
    valid = (aa > .01) & (bb > .01)
    return {
        'grad_rms_optical': numeric(np.sqrt(np.mean(aa**2))),
        'grad_rms_sar': numeric(np.sqrt(np.mean(bb**2))),
        'abs_grad_cosine_on_both_active': numeric(np.mean(np.abs(dot[valid])/(aa[valid]*bb[valid]+1e-8))) if np.any(valid) else None,
        'both_active_fraction':numeric(valid.mean()),
        'sar_valid_fraction':numeric((sar_u8[y0:y1,x0:x1]>0).mean()),
    }


def ranks(v):
    """Average ranks for tied observations; no SciPy requirement."""
    x = np.asarray(v, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=np.float64)
    sorted_v = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_v[j] == sorted_v[i]:
            j += 1
        result[order[i:j]] = 0.5*(i+j-1)
        i = j
    return result


def spearman(x, y):
    pairs = [(float(a),float(b)) for a,b in zip(x,y)
             if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 5:
        return None
    a,b = np.array(pairs).T
    ra,rb = ranks(a),ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return None
    return numeric(np.corrcoef(ra,rb)[0,1])


def region_rows(model, optical_u8, sar_u8, scene_id, *, tile_size=64, step=.05, min_valid_fraction=.90, device='cpu'):
    """Runs the SAME frozen encoder, LWPD, Base fusion and decoder for both probes."""
    if not (0 < step <= .1):
        raise ValueError('step must be in (0, .1]')
    h,w = optical_u8.shape
    if sar_u8.shape != (h,w) or h % tile_size or w % tile_size or tile_size % 2:
        raise ValueError('Require aligned HxW images divisible by an even tile_size')
    t = lambda img: torch.from_numpy(img.astype(np.float32)[None,None] / 255.).to(device)
    optical,sar = t(optical_u8),t(sar_u8)
    results = []
    with torch.inference_mode():
        encoded = model.encode_pair(optical,sar)
        bo,do,bs,ds = encoded
        base, detail = model.fuse_features(*encoded)
        original = model.reconstruct(optical,base,detail)
        torch.testing.assert_close(original,model(optical,sar),rtol=0,atol=1e-6)
        # Optical-base anchor and Base fusion weights are held constant in every intervention.
        for row in range(h//tile_size):
            for col in range(w//tile_size):
                roi = (row*tile_size,(row+1)*tile_size,col*tile_size,(col+1)*tile_size)
                y0,y1,x0,x1=roi
                r={'scene_id':str(scene_id),'row':row,'col':col,'y0':y0,'y1':y1,'x0':x0,'x1':x1}
                r.update(gradient_stats(optical_u8,sar_u8,roi))
                r['use_for_aggregate'] = r['sar_valid_fraction'] >= min_valid_fraction
                # Local spatial variations of all channels, not feature semantic labels.
                r['base_local_std']=numeric(base[...,y0:y1,x0:x1].float().std(dim=(-2,-1),unbiased=False).mean())
                for name, source in (('optical',do),('sar',ds)):
                    local_rms = rms(source[...,y0:y1,x0:x1])
                    mask = torch.zeros_like(source)
                    mask[...,y0:y1,x0:x1]=1
                    plus=source*(1+step*mask)
                    minus=source*(1-step*mask)
                    def evaluate(updated):
                        if name=='optical':
                            d=model.detail_fusion(updated,ds,base)
                        else:
                            d=model.detail_fusion(do,updated,base)
                        return model.reconstruct(optical,base,d)
                    response=local_response(evaluate(plus),evaluate(minus),roi,step,local_rms)
                    r.update({f'{name}_{k}':v for k,v in response.items()})
                og=r['optical_gain_per_feature_rms']; sg=r['sar_gain_per_feature_rms']
                r['log_sar_over_optical_gain']=numeric(math.log((sg+1e-9)/(og+1e-9))) if og is not None and sg is not None else None
                r['log_sar_over_optical_feature_rms']=numeric(math.log((r['sar_feature_region_rms']+1e-9)/(r['optical_feature_region_rms']+1e-9)))
                results.append(r)
    return results


def summary(rows):
    skipped=len([r for r in rows if not r.get('use_for_aggregate',True)])
    rows=[r for r in rows if r.get('use_for_aggregate',True)]
    arr=[r['log_sar_over_optical_gain'] for r in rows if r.get('log_sar_over_optical_gain') is not None]
    values=np.array(arr,dtype=float)
    stats={'regions_analyzed':len(rows),'regions_skipped_insufficient_sar_validity':skipped,'valid_gain_ratios':len(values)}
    if len(values):
        stats.update({
            'sar_gain_higher_fraction':numeric(np.mean(values>0)),
            'optical_gain_higher_fraction':numeric(np.mean(values<0)),
            'gain_log_ratio_q10_q50_q90':[numeric(x) for x in np.quantile(values,[.1,.5,.9])],
            'gain_log_ratio_std':numeric(values.std()),
        })
    for feature in ['log_sar_over_optical_feature_rms','base_local_std',
                    'grad_rms_optical','grad_rms_sar','abs_grad_cosine_on_both_active']:
        stats[f'spearman_gain_ratio_vs_{feature}']=spearman(
            [r.get('log_sar_over_optical_gain') for r in rows],
            [r.get(feature) for r in rows])
    by_scene={}
    for scene in sorted({r['scene_id'] for r in rows}):
        by_scene[scene]=summary_one([r for r in rows if r['scene_id']==scene])
    stats['per_scene']=by_scene
    return stats


def summary_one(rows):
    rows=[r for r in rows if r.get('use_for_aggregate',True)]
    values=np.array([r['log_sar_over_optical_gain'] for r in rows
                     if r['log_sar_over_optical_gain'] is not None],dtype=float)
    if len(values)==0: return {'count':len(rows)}
    return {'count':len(rows),'fraction_sar_gain_higher':numeric(np.mean(values>0)),
            'q10_q50_q90':[numeric(x) for x in np.quantile(values,[.1,.5,.9])],
            'range':numeric(values.max()-values.min())}


def find_scene(data_root,manifest,scene_id):
    paths=[]
    for k in ['optical_dir','sar_dir']:
        folder=data_root / manifest[k]
        m=[p for p in folder.glob(f'{scene_id}.*') if p.is_file()]
        if len(m)!=1:raise FileNotFoundError(f'Expected exactly one scene {scene_id} image in {folder}; got {len(m)}')
        paths.append(m[0])
    return paths


def main():
    from e2_base_anchor import OpticalBaseAnchoredE2
    from utils.dataset_os import validate_manifest_splits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-ids',nargs='+',required=True)
    parser.add_argument('--tile-size',type=int,default=64)
    parser.add_argument('--step',type=float,default=.05)
    parser.add_argument('--min-valid-fraction',type=float,default=.90)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Refusing to overwrite existing output directory')
    if not args.checkpoint.is_file():parser.error(f'Checkpoint not found: {args.checkpoint}')
    if not args.manifest.is_file():parser.error(f'Manifest not found: {args.manifest}')
    if not(0<args.step<=.1):parser.error('step must be in (0,.1]')
    if not(0 <= args.min_valid_fraction <= 1):parser.error('min-valid-fraction must be in [0,1]')
    with args.manifest.open(encoding='utf-8') as f:manifest=json.load(f)
    validate_manifest_splits(manifest)
    ids=[str(x) for x in args.scene_ids]
    if len(set(ids))!=len(ids) or not all(i in manifest['splits']['val'] for i in ids):
        parser.error('Scene IDs must be distinct members of the manifest val split')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('variant')!=OpticalBaseAnchoredE2.VARIANT:
        raise ValueError('Expected lwpd_e2_base_anchor_v1 checkpoint')
    model=OpticalBaseAnchoredE2(checkpoint['variant']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    all_rows=[]
    input_hashes={}
    for scene in ids:
        paths=find_scene(args.data_root,manifest,scene)
        input_hashes[scene]={p.name:sha256(p) for p in paths}
        obgr=cv2.imread(str(paths[0]),cv2.IMREAD_COLOR)
        su8=cv2.imread(str(paths[1]),cv2.IMREAD_GRAYSCALE)
        if obgr is None or su8 is None:raise OSError(f'Cannot read {paths}')
        ou8=cv2.cvtColor(obgr,cv2.COLOR_BGR2YCrCb)[...,0]
        rows=region_rows(model,ou8,su8,scene,tile_size=args.tile_size,step=args.step,min_valid_fraction=args.min_valid_fraction,device=args.device)
        all_rows.extend(rows)
        print(f'Scene {scene}: {len(rows)} regions, SAR-gain-higher fraction '
              f'{summary_one(rows).get("fraction_sar_gain_higher")}',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'regions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    metadata={'protocol':'frozen_e2_s2_regional_response_v1',
        'status':'measured_frozen_model_local_sensitivity',
        'split':'val','scene_ids':ids,'tile_size':args.tile_size,'step':args.step,'min_sar_valid_fraction':args.min_valid_fraction,
        'device':args.device,'precision':'fp32_no_tf32',
        'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256(args.checkpoint),
        'manifest':str(args.manifest),'manifest_sha256':sha256(args.manifest),
        'script_sha256':sha256(Path(__file__)),
        'input_sha256':input_hashes,'summary':summary(all_rows),
        'limitations':[
          'Sensitivity is not structure truth, expert oracle, or information utility.',
          'Regional differences in an input-conditioned CNN do not prove MoE superiority.',
          'Source feature perturbations have equal relative step; gain normalized by each local feature RMS.',
          'No training, no test split, no manual structure labels, no clean SAR.',
          'Sobel responses can reflect SAR speckle; not independently validated structure.',
          'The E2 loss may favor SAR strong gradients; this script does not estimate optimal fusion strategy.'
        ]}
    with (args.output_dir/'diagnosis.json').open('w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False,allow_nan=False)
    print('Saved:',args.output_dir/'regions.csv',args.output_dir/'diagnosis.json')


if __name__=='__main__':main()