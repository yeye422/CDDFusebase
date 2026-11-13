"""Single-val-scene LL-context on/off intervention. Never trains or scans images."""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from direct_fusion import DirectFusionCDDFuse
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT.parent/'datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json'
BASE=ROOT/'models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth'
CANDIDATE=ROOT/'models/lwpd_e2_llctx_v1/seed42/epoch_012.pth'


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-id',default='4')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--probe-repairs',action='store_true',help='Exploratory one-scene half-scale, spatial-mean and per-modality interventions; not tuning or training')
    args=parser.parse_args()
    out=(ROOT/args.output_dir).resolve();out.relative_to(ROOT)
    if out.exists():raise FileExistsError(out)
    splits,_=manifest_pairs(ROOT.parent/'datasets/QXSLAB_SAROPT',MANIFEST)
    index,pair=next(((i,p) for i,p in enumerate(splits['val']) if p[0].stem==args.scene_id),(None,None))
    if pair is None:raise ValueError('Only one explicit val scene is allowed; no test')
    torch.set_num_threads(2);torch.manual_seed(42)
    optical_bgr=cv2.imread(str(pair[0]),cv2.IMREAD_COLOR)
    optical=cv2.cvtColor(optical_bgr,cv2.COLOR_BGR2YCrCb)[...,0]
    sar=cv2.imread(str(pair[1]),cv2.IMREAD_GRAYSCALE)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tensor=lambda x:torch.from_numpy(x.astype(np.float32)/255)[None,None].to(device)
    x,s=tensor(optical),tensor(sar)
    out.mkdir(parents=True)
    images={};raws={};rows=[];cases=[]
    conditions=[('original_e2',BASE,'none'),('candidate_on',CANDIDATE,'none'),('candidate_off',CANDIDATE,'zero')]
    if args.probe_repairs:
        conditions += [(name,CANDIDATE,mode) for name,mode in [
            ('candidate_half','half'),('candidate_spatial_mean','spatial_mean'),
            ('candidate_optical_only','optical_only'),('candidate_sar_only','sar_only')]]
    for name,path,mode in conditions:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=DirectFusionCDDFuse(checkpoint['variant']).to(device).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        # Remove only the *current* LL-context output, leaving all trained old modules unchanged.
        hook=None
        if mode!='none':
            def intervene(module,inputs,output):
                a,b,stats=output
                if mode=='zero':a,b=torch.zeros_like(a),torch.zeros_like(b)
                elif mode=='half':a,b=.5*a,.5*b
                elif mode=='spatial_mean':
                    a=a.mean(dim=(-2,-1),keepdim=True).expand_as(a)
                    b=b.mean(dim=(-2,-1),keepdim=True).expand_as(b)
                elif mode=='optical_only':b=torch.zeros_like(b)
                elif mode=='sar_only':a=torch.zeros_like(a)
                else:raise ValueError(mode)
                return a,b,stats
            hook=model.lwpd.ll_context.register_forward_hook(intervene)
        with torch.no_grad():
            raw=model(x,s).float().squeeze().cpu().numpy()
        if hook is not None:hook.remove()
        if not np.isfinite(raw).all():raise ValueError('Non-finite output')
        gray=np.rint(np.clip(raw,0,1)*255).astype(np.uint8)
        metrics=dict(zip(PaperEvaluator.METRIC_NAMES,map(float,PaperEvaluator.metric_vector(gray,optical,sar))))
        rows.append(dict(condition=name,**metrics))
        np.save(out/f'{name}_raw.npy',raw)
        if not cv2.imwrite(str(out/f'{name}.png'),gray):raise OSError('Cannot write image')
        images[name]=gray;raws[name]=raw
        cases.append(dict(condition=name,checkpoint=str(path),checkpoint_sha256=sha256_file(path),
                          variant=checkpoint['variant'],intervention=mode))
        print(name,json.dumps(metrics),flush=True)
        del model,checkpoint
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    by={r['condition']:r for r in rows}
    summary=dict(on_off_raw_mae=float(np.abs(raws['candidate_on']-raws['candidate_off']).mean()),
                 on_off_u8_changed_fraction=float(np.mean(images['candidate_on']!=images['candidate_off'])),
                 on_nabf=by['candidate_on']['Nabf'],off_nabf=by['candidate_off']['Nabf'],
                 original_nabf=by['original_e2']['Nabf'])
    write_json(out/'summary.json',summary)
    meta=dict(scene_id=args.scene_id,split='val',offset=index,evaluated_pairs=1,cases=cases,probe_repairs=args.probe_repairs,
        metric_profile='paper',output_policy='clamp_then_round_uint8',precision='fp32',device=device,
        torch_version=torch.__version__,split_manifest=str(MANIFEST),split_manifest_sha256=sha256_file(MANIFEST),
        inputs={str(p):sha256_file(p) for p in pair},
        sources={name:sha256_file(ROOT/name) for name in ['diagnose_llctx_single.py','lwpd_ll_context.py','lwpd.py','direct_fusion.py','net.py','utils/Evaluator.py']},
        limitations=['Single previously used val scene, not held-out confirmation.',
                     'Turning off a co-adapted branch is an out-of-training intervention, not independently trained E2.',
                     'No decomposition of additive module contributions or training-seed uncertainty.',
                     'Repair probes are fixed exploratory interventions, not a search for a deployable scale or architecture.'])
    write_json(out/'run_metadata.json',meta)
    tiles=[]
    for name,gray in [('Optical',optical),('SAR',sar),*images.items()]:
        tile=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        tile=cv2.copyMakeBorder(tile,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(tile,name,(4,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
        tiles.append(tile)
    cv2.imwrite(str(out/'overview.png'),np.concatenate(tiles,axis=1))
    relative=out.relative_to(ROOT).as_posix()
    registry_path=ROOT/'EXPERIMENT_REGISTRY.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    eid='LLCTX-SINGLE-DIAG-'+sha256_file(out/'run_metadata.json')[:12].upper()
    registry['experiments'].append(dict(id=eid,status='diagnostic_single_scene',task='engineering',
        method='LL-context one-scene intervention probes; original E2 as reference',
        comparable_group=eid,dataset='QXS-SAROPT',split='val',offset=index,sample_count=1,
        split_manifest='../'+MANIFEST.relative_to(ROOT.parent).as_posix(),split_manifest_sha256=sha256_file(MANIFEST),
        checkpoint=CANDIDATE.relative_to(ROOT).as_posix(),checkpoint_sha256=sha256_file(CANDIDATE),
        parent_checkpoint=BASE.relative_to(ROOT).as_posix(),parent_checkpoint_sha256=sha256_file(BASE),
        metric_profile='single_scene_intervention_paper',output_policy='clamp_then_round_uint8',
        result_dir=relative,metadata_path=relative+'/run_metadata.json',metrics=summary,
        metric_source={'type':'json_paths','path':relative+'/summary.json','paths':{k:[k] for k in summary}},
        notes='单样本固定checkpoint干预，无训练。关闭LL-context不恢复独立E2；不作为总体性能确认。'))
    registry['updated_at']=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    write_json(registry_path,registry)
    subprocess.run([sys.executable,str(ROOT/'maintain_experiment_index.py'),'--check','--render'],cwd=ROOT,check=True)
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
