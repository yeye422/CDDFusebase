"""Diagnose E2-B/G1 targets. Default reuses images; optional one-scene FP32 check."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.dataset_os import manifest_pairs
from utils.e2_target_diagnostics import make_targets, output_diagnostics

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT.parent / 'datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json'
SOURCES = {
    'e2_b': ROOT / 'test_result/e2_general_loss_v1/val1000/b_seed42',
    'g1': ROOT / 'test_result/e2_g1_v1/val1000/g1_seed42',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def array(tensor):
    return tensor.detach().cpu().numpy().squeeze()


def tile(data, label, upper=1., signed=False):
    if signed:
        v = np.clip(data/max(upper, 1e-12), -1, 1)
        rgb = np.full((*v.shape, 3), 255, dtype=np.uint8)
        rgb[..., 0] = (255*(1-np.maximum(v,0))).astype(np.uint8)
        rgb[..., 1] = (255*(1-np.abs(v))).astype(np.uint8)
        rgb[..., 2] = (255*(1-np.maximum(-v,0))).astype(np.uint8)
    else:
        gray = np.rint(np.clip(data/max(upper, 1e-12), 0, 1)*255).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = cv2.copyMakeBorder(rgb, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(rgb,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    return rgb


def read_gray(path, shape=None):
    value=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if value is None or (shape is not None and value.shape!=shape):
        raise ValueError(f'Cannot read image with expected size: {path}')
    return value


def register(out, meta, summary):
    path=ROOT/'EXPERIMENT_REGISTRY.json'
    registry=json.loads(path.read_text(encoding='utf-8'))
    relative=out.relative_to(ROOT).as_posix()
    eid='E2-TARGET-DIAG-'+hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    if any(e['id']==eid for e in registry['experiments']):
        raise ValueError('Duplicate diagnostic registry id')
    base=meta['saved_sources']['e2_b']
    registry['experiments'].append(dict(
        id=eid,status='diagnostic_single_scene' if len(meta['scene_ids'])==1 else 'diagnostic_fixed_subset',
        task='engineering',method='Saved E2-B/G1 target-conflict and BGC1 diagnostic; no candidate training',
        comparable_group=eid,dataset='QXS-SAROPT',split='val',
        split_manifest=MANIFEST.relative_to(ROOT.parent).as_posix().replace('datasets/','../datasets/',1),
        split_manifest_sha256=meta['split_manifest_sha256'],offset=meta['scene_offsets'][0],
        sample_count=len(meta['scene_ids']),selected_scene_ids=meta['scene_ids'],
        checkpoint=Path(base['checkpoint']).relative_to(ROOT).as_posix(),checkpoint_sha256=base['checkpoint_sha256'],
        metric_profile='e2_target_diagnostic_v1',output_policy='reuse_saved_uint8_div255_cpu_fp32',
        result_dir=relative,metadata_path=relative+'/run_metadata.json',metrics=summary,
        metric_source={'type':'json_paths','path':relative+'/summary.json',
                       'paths':{k:[k] for k in summary}},
        notes=('复用已有灰度输出诊断目标；另做同一张图E2-B FP32前向核验。' if meta['raw_baseline'] else '仅复用已有灰度输出诊断目标；未做模型推理。')+
              '未训练候选；输出处的梯度不等于训练参数梯度；ROI未核查不得宣称噪声/真结构可分；不作正式模型排名。'))
    registry['updated_at']=datetime.now().strftime('%Y-%m-%d')
    write_json(path,registry)
    subprocess.run([sys.executable,str(ROOT/'maintain_experiment_index.py'),'--check','--render'],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-ids',nargs='+',default=['4'])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--raw-baseline',action='store_true',help='Single-scene FP32 E2-B forward to test saved-output quantization sensitivity; never trains')
    parser.add_argument('--roi-file',type=Path,help='Optional predeclared JSON {scene_id: [{name,x,y,width,height,label}]}')
    args=parser.parse_args()
    if len(set(args.scene_ids))!=len(args.scene_ids) or not 1<=len(args.scene_ids)<=12:
        raise ValueError('Choose 1 to 12 distinct explicit val scenes; no automatic scan')
    if args.raw_baseline and len(args.scene_ids)!=1:
        raise ValueError('--raw-baseline is restricted to one scene')
    out=(ROOT/args.output_dir).resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    torch.set_num_threads(2)
    splits,_=manifest_pairs(ROOT.parent/'datasets/QXSLAB_SAROPT',MANIFEST)
    val={pair[0].stem:(i,pair) for i,pair in enumerate(splits['val'])}
    if any(scene not in val for scene in args.scene_ids):
        raise ValueError('All scene IDs must belong to val; test/train are rejected')
    reg=json.loads((ROOT/'EXPERIMENT_REGISTRY.json').read_text(encoding='utf-8'))
    source_meta={}
    for name,path in SOURCES.items():
        m=json.loads((path/'run_metadata.json').read_text(encoding='utf-8'))
        entries=[e for e in reg['experiments'] if (ROOT/e['result_dir']).resolve()==path.resolve()]
        if len(entries)!=1 or entries[0]['comparable_group']!='QXS-VAL1000-E2-GENERAL-LOSS-v1':
            raise ValueError('Saved B/G1 comparable groups must match')
        if m['split']!='val' or m['split_manifest_sha256']!=sha(MANIFEST):
            raise ValueError('Saved source protocol mismatch')
        if any(scene not in m['selected_scene_ids'] for scene in args.scene_ids):
            raise ValueError('Scene absent from saved evaluation')
        if sha(m['checkpoint'])!=m['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        source_meta[name]=dict(checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],
                               metadata_path=str(path/'run_metadata.json'),metadata_sha256=sha(path/'run_metadata.json'))
    for key in ['metric_profile','output_range_policy','selected_scene_ids','offset']:
        a=json.loads((SOURCES['e2_b']/'run_metadata.json').read_text())[key]
        b=json.loads((SOURCES['g1']/'run_metadata.json').read_text())[key]
        if a!=b:raise ValueError('Saved B/G1 mismatch: '+key)
    rois=json.loads(args.roi_file.read_text(encoding='utf-8')) if args.roi_file else {}
    # Preflight all requested files and ROI bounds before creating the unique directory.
    for scene in args.scene_ids:
        shape=read_gray(val[scene][1][1]).shape
        for path in SOURCES.values():read_gray(path/'gray'/f'{scene}.png',shape)
        for roi in rois.get(scene,[]):
            x,y,w,h=[roi[k] for k in ['x','y','width','height']]
            if not all(type(v) is int for v in [x,y,w,h]) or min(x,y)<0 or min(w,h)<=0 or x+w>shape[1] or y+h>shape[0]:
                raise ValueError('Invalid ROI: '+str(roi))
    out.mkdir(parents=True)
    meta=dict(scene_ids=args.scene_ids,scene_offsets=[val[s][0] for s in args.scene_ids],
              split='val',split_manifest=str(MANIFEST),split_manifest_sha256=sha(MANIFEST),
              saved_sources=source_meta,eta=.10,coherence_window=5,coherence_sigma=1.,epsilon=1e-6,
              complete_support=7,gradient_weight=10.,precision='cpu_fp32',
              raw_baseline=args.raw_baseline,
              output_policy='saved_uint8_div255',metric_profile='e2_target_diagnostic_v1',
              source_sha256={name:sha(ROOT/name) for name in ['diagnose_e2_targets.py','utils/e2_target_diagnostics.py','utils/loss.py']},
              roi_file=str(args.roi_file.resolve()) if args.roi_file else None,
              roi_sha256=sha(args.roi_file) if args.roi_file else None,inputs=[],
              limitations=['No BGC1 model output; BGC1 changes diagnostic targets only.',
                           'Saved uint8 outputs have quantization kinks; output derivatives are not parameter updates.',
                           'No inference that coherence is a noise label; no ROI labels means mechanism screening is pending.',
                           'Fusion losses exclude reconstruction and are not full training losses.'])
    rows=[]; target_rows=[]; roi_rows=[]
    for scene in args.scene_ids:
        folder=out/scene;folder.mkdir()
        op,sp=val[scene][1]
        optical=cv2.cvtColor(cv2.imread(str(op),cv2.IMREAD_COLOR),cv2.COLOR_BGR2YCrCb)[...,0]
        sar=read_gray(sp,optical.shape)
        tensor=lambda v:torch.from_numpy(v.astype(np.float32)/255)[None,None]
        o,s=tensor(optical),tensor(sar);v=(s>0).float()
        t=make_targets(o,s,v,eta=.10)
        tr=dict(scene_id=scene,target_gap_abs_mean=float(t['target_gap'].abs().mean()),
                target_gap_nonzero_fraction=float((t['target_gap'].abs()>1e-6).float().mean()),
                support_fraction=float(t['support'].mean()),correction_mean=float(t['correction'].mean()),
                correction_p99=float(torch.quantile(t['correction'],.99)),
                correction_fraction=float((t['correction']>1e-6).float().mean()),
                sar_excess_sum=float(t['excess'].sum()),
                removed_excess_ratio=float(t['correction'].sum()/t['excess'].sum()) if t['excess'].sum()>0 else 0.)
        target_rows.append(tr)
        outputs=dict(optical=o,sar=s,mean=(o+s)/2,max_intensity=t['intensity'])
        paths={name:path/'gray'/f'{scene}.png' for name,path in SOURCES.items()}
        outputs.update({name:tensor(read_gray(path,optical.shape)) for name,path in paths.items()})
        if args.raw_baseline:
            from direct_fusion import DirectFusionCDDFuse
            device='cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint=torch.load(source_meta['e2_b']['checkpoint'],map_location='cpu',weights_only=False)
            model=DirectFusionCDDFuse(checkpoint['variant']).to(device).eval()
            model.load_state_dict(checkpoint['model'],strict=True)
            with torch.no_grad():
                raw=model(o.to(device),s.to(device)).float().cpu()
            if not torch.isfinite(raw).all():raise ValueError('Non-finite raw baseline')
            quantized=torch.round(raw.clamp(0,1)*255)/255
            outputs['e2_b_fp32']=raw
            outputs['e2_b_fp32_u8']=quantized
            np.save(folder/'e2_b_fp32_raw.npy',array(raw))
            meta['raw_baseline_check']=dict(device=device,precision='fp32_no_amp',torch_version=torch.__version__,
                model_source_sha256={name:sha(ROOT/name) for name in ['direct_fusion.py','lwpd.py','net.py']},
                saved_output_mae=float((raw-outputs['e2_b']).abs().mean()),
                changed_quantized_fraction=float((quantized!=outputs['e2_b']).float().mean()),
                caveat='Original saved output used its recorded inference environment; differences include precision/environment as well as quantization.')
            del model,checkpoint
        meta['inputs'].append(dict(scene_id=scene,files={str(p):sha(p) for p in [op,sp,*paths.values()]}))
        tile_data=[tile(array(o),'Optical'),tile(array(s),'SAR'),tile(array(outputs['e2_b']),'Saved E2-B'),
                   tile(array(outputs['g1']),'Saved G1'),tile(array(t['coherence']),'SAR coherence'),
                   tile(array(t['support']),'Full valid support'),
                   tile(array(t['correction']),'Correction (white >= 0.10)',.10),
                   tile(array(t['target_gap']),'A(max)-max(A) (+red)',1.,True)]
        np.savez_compressed(folder/'targets.npz',**{k:array(x) for k,x in t.items()})
        for name,image in outputs.items():
            stats,maps=output_diagnostics(image,o,s,v,t)
            rows.append(dict(scene_id=scene,output=name,**stats))
            np.savez_compressed(folder/f'{name}_maps.npz',**{k:array(x) for k,x in maps.items()})
            if name=='e2_b':
                tile_data += [tile(array(maps['conflict']),'E2-B loss conflict'),
                              tile(array(maps['derivative_changed']),'E2-B L1 derivative changed'),
                              tile(array(maps['euclidean_overshoot']),'E2-B overshoot (white >=1)',1.),
                              tile(array(t['excess']),'SAR excess (white >=1)',1.)]
            for roi in rois.get(scene,[]):
                x,y,w,h=[roi[k] for k in ['x','y','width','height']]
                cut=lambda value:value[...,y:y+h,x:x+w]
                roi_rows.append(dict(scene_id=scene,output=name,name=roi['name'],label=roi.get('label','unverified'),
                    correction_mean=float(cut(t['correction']).mean()),
                    correction_fraction=float((cut(t['correction'])>1e-6).float().mean()),
                    target_gap_abs_mean=float(cut(t['target_gap']).abs().mean()),
                    conflict_fraction=float(cut(maps['conflict']).mean()),
                    euclidean_overshoot_mean=float(cut(maps['euclidean_overshoot']).mean())))
        montage=np.concatenate([np.concatenate(tile_data[i:i+4],axis=1) for i in range(0,12,4)],axis=0)
        if not cv2.imwrite(str(folder/'overview.png'),montage):raise OSError('Image write failed')
    write_csv(out/'output_diagnostics.csv',rows);write_csv(out/'target_diagnostics.csv',target_rows)
    write_csv(out/'roi_diagnostics.csv',roi_rows)
    summary={k:float(np.mean([r[k] for r in target_rows])) for k in ['target_gap_abs_mean','correction_mean','removed_excess_ratio']}
    write_json(out/'summary.json',summary);write_json(out/'run_metadata.json',meta)
    lines=['# E2目标诊断结果','','默认复用保存的8位输出；启用raw-baseline时另做单张FP32前向。未训练、未生成BGC1融合图。',
           '梯度是各行输出处的局部导数（fp32行未量化），不是模型参数梯度；不包含重建项。',
           '缺少预先标记ROI时，不能判断修正是否保护真实SAR结构。','',
           '|场景|输出|强度项|原梯度项|新梯度项|原/新梯度余弦|冲突像素比例|',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cosine=r['old_new_gradient_cosine']
        lines.append(f"|{r['scene_id']}|{r['output']}|{r['intensity_loss']:.6f}|{r['old_gradient_loss']:.6f}|{r['new_gradient_loss']:.6f}|{cosine if cosine is not None else '未定义'}|{r['conflict_fraction']:.4%}|")
    lines += ['', '本表是候选输出在同一诊断目标下的代价，不是模型性能排名。',
              '决策：完成独有细线、点目标、角点和背景ROI核查前，不据此启动BGC1训练。']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register(out,meta,summary)
    print(json.dumps(dict(output=str(out),summary=summary),ensure_ascii=True))


if __name__=='__main__':main()
