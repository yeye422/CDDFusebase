"""User-run E2 LL-context training and explicit validation/test workflow."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from compare_lwpd_prl import METRICS, read_metrics, paired_record
from train_direct_fusion import sha256_file

ROOT=Path(__file__).resolve().parent
INIT=Path('models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth')
MANIFEST=Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
MODEL=Path('models/lwpd_e2_llctx_v1')
RESULT=Path('test_result/lwpd_e2_llctx_v1/val1000/seed42')
TEST_RESULT=Path('test_result/lwpd_e2_llctx_v1/test1000/seed42')
BASELINE=Path('test_result/lwpd_qxs_ablation_v1/val1000/e2_seed42')
GROUP='QXS-VAL1000-DIRECT-LWPD-v1'


def run(args):
    print('RUN:',subprocess.list2cmdline([str(x) for x in args]),flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def train_args(smoke=False):
    args=['train_lwpd.py','--variant','lwpd_e2_llctx_v1','--data-root','../datasets/QXSLAB_SAROPT',
          '--split-manifest',MANIFEST,'--init-checkpoint',INIT,'--gradient-weight','10',
          '--reconstruction-weight','.1','--structure-weight','0','--gradient-loss','baseline',
          '--train-limit','1' if smoke else '0','--val-limit','1' if smoke else '0',
          '--seed','42','--batch-size','1' if smoke else '2','--accumulation-steps','4',
          '--num-workers','0','--patch-size','256','--warmup-epochs','2','--local-epochs','3','--joint-epochs','7',
          '--warmup-lwpd-lr','1e-5','--local-lwpd-lr','2e-5','--joint-lwpd-lr','1e-5',
          '--warmup-fusion-lr','1e-5','--local-fusion-lr','2e-5','--joint-fusion-lr','1e-5',
          '--decoder-lr','1e-5','--encoder-tail-lr','5e-6','--encoder-early-lr','1e-6',
          '--warmup-ll-context-lr','1e-4','--local-ll-context-lr','1e-4','--joint-ll-context-lr','5e-5',
          '--output-dir','models/lwpd_e2_llctx_smoke_v1' if smoke else MODEL,'--run-name','seed42']
    if smoke:args+=['--dry-run','--no-amp']
    return [str(x) for x in args]


def lock_protocol():
    files=['lwpd_ll_context.py','direct_fusion.py','lwpd.py','net.py',
           'train_direct_fusion.py','train_lwpd.py','utils/loss.py','utils/dataset_os.py','test_OS.py',
           'utils/Evaluator.py','run_lwpd_e2_llctx.py']
    payload=dict(protocol='lwpd_e2_llctx_v1',variant='lwpd_e2_llctx_v1',
                 init_checkpoint_sha256=sha256_file(INIT),split_manifest_sha256=sha256_file(MANIFEST),
                 source_sha256={p:sha256_file(Path(p)) for p in files},train_args=train_args(),
                 evaluation=dict(split='val',offset=0,limit=1000,metric_profile='paper',output_policy='clamp'))
    MODEL.mkdir(parents=True,exist_ok=True)
    path=MODEL/'workflow_protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=payload:
            raise RuntimeError('Protocol changed; use a new version and output directories')
    else:write_json(path,payload)


def register(kind):
    smoke=kind=='SMOKE';validation=kind in ('VAL1000','TEST1000');formal=kind=='TEST1000'
    folder=Path('models/lwpd_e2_llctx_smoke_v1/seed42') if smoke else MODEL/'seed42'
    result=(TEST_RESULT if formal else RESULT) if validation else folder
    ck=INIT if smoke else folder/'epoch_012.pth'
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config['variant']!='lwpd_e2_llctx_v1' or config['init_checkpoint_sha256']!=sha256_file(INIT):
        raise RuntimeError('Wrong initialization/variant')
    entry=dict(id=f'LWPD-E2-LLCTX-V1-SEED42-{kind}',
        status='engineering_smoke' if smoke else ('formal_test_supplement' if formal else ('development_validation' if validation else 'training_complete')),
        task='engineering' if smoke else ('fusion' if validation else 'training'),
        method='E2 + LL-context; unchanged intensity + 10*gradient + 0.1*reconstruction',
        comparable_group=('QXS-TEST1000-PAPER-CLAMP-v1' if formal else GROUP) if validation else f'QXS-LWPD-E2-LLCTX-V1-{kind}',dataset='QXS-SAROPT',
        split_manifest=MANIFEST.as_posix(),split_manifest_sha256=sha256_file(MANIFEST),
        split='test' if formal else ('val' if validation or smoke else 'train'),offset=0,sample_count=1 if smoke else (1000 if validation else 8000),
        checkpoint=ck.as_posix(),checkpoint_sha256=sha256_file(ck),
        parent_checkpoint=INIT.as_posix(),parent_checkpoint_sha256=sha256_file(INIT),
        result_dir=result.as_posix(),metadata_path=(result/('run_metadata.json' if validation else 'config.json')).as_posix(),
        metric_profile='paper' if validation else 'training_loss',output_policy='clamp_then_uint8' if validation else 'float_loss',
        notes='单样本前向反传，无优化器更新，不是性能结果。' if smoke else
              'E2基础上新增LL-context并继续训练12轮；对原E2的比较包含额外训练收益，不单独证明模块因果贡献。未改损失；test仅由用户显式执行。')
    if smoke:
        path=folder/'dry_run_diagnostics.json';data=json.loads(path.read_text())
        if not data['gradients_finite'] or data['optimizer_updated']:raise RuntimeError('Invalid smoke result')
        entry['metrics']={'loss':data['loss']}
        entry['metric_source']={'type':'json_paths','path':path.as_posix(),'paths':{'loss':['loss']}}
    elif validation:
        _,mean=read_metrics(result/'metrics.csv')
        entry['metrics']={k:float(mean[k]) for k in METRICS}
        entry['metric_source']={'type':'csv_row','path':(result/'metrics.csv').as_posix(),
                                'match':{'image':'mean'},'columns':{k:k for k in METRICS}}
    else:
        with (folder/'history.csv').open(encoding='utf-8-sig',newline='') as f:
            row=next(r for r in csv.DictReader(f) if r['epoch']=='12')
        entry['metrics']={k:float(row[k]) for k in ['train_loss','val_loss']}
        entry['metric_source']={'type':'csv_row','path':(folder/'history.csv').as_posix(),
                                'match':{'epoch':'12'},'columns':{k:k for k in entry['metrics']}}
    path=Path('EXPERIMENT_REGISTRY.json');registry=json.loads(path.read_text(encoding='utf-8'))
    old=next((e for e in registry['experiments'] if e['id']==entry['id']),None)
    if old is not None and old!=entry:raise RuntimeError('Registration differs; do not overwrite')
    if old is None:
        registry['experiments'].append(entry)
        registry['updated_at']=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        write_json(path,registry)
    run([sys.executable,'maintain_experiment_index.py','--check','--render'])


def train():
    lock_protocol()
    folder=MODEL/'seed42';final=folder/'epoch_012.pth'
    if not final.exists():
        command=train_args()
        saved=sorted(folder.glob('epoch_*.pth'))
        if saved:command+=['--resume',str(saved[-1])]
        run([sys.executable,'-u',*command])
    else:print('SKIP completed candidate training:',final)
    register('TRAIN8000')


def validate(split='val'):
    lock_protocol()
    result=RESULT if split=='val' else TEST_RESULT
    if split=='test' and not (RESULT/'comparison_vs_e2/SUMMARY.md').exists():
        raise RuntimeError('Complete val1000 and inspect the comparison before explicit test')
    ck=MODEL/'seed42/epoch_012.pth'
    if not ck.exists():raise FileNotFoundError(ck)
    if not (result/'metrics.csv').exists():
        if result.exists():raise FileExistsError('Preserve partial result; use a new version')
        run([sys.executable,'-u','test_OS.py','--data-root','../datasets/QXSLAB_SAROPT',
             '--split-manifest',MANIFEST,'--split',split,'--variant','auto','--checkpoint',ck,
             '--output-dir',result,'--offset','0','--limit','1000','--metric-profile','paper',
             '--output-range-policy','clamp','--timing-warmup','10'])
    meta=json.loads((result/'run_metadata.json').read_text(encoding='utf-8'))
    expected=dict(checkpoint_sha256=sha256_file(ck),split_manifest_sha256=sha256_file(MANIFEST),
                  split=split,offset=0,evaluated_pairs=1000,metric_profile='paper',
                  output_range_policy='clamp',variant='lwpd_e2_llctx_v1',metrics_only=False)
    if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('Evaluation metadata mismatch')
    rows,_=read_metrics(result/'metrics.csv')
    if len(rows)!=1000 or set(rows)!=set(meta['selected_scene_ids']):raise RuntimeError('Expected identical 1000 IDs')
    if any(not math.isfinite(float(r[k])) for r in rows.values() for k in METRICS):raise RuntimeError('Non-finite metrics')
    for name in ['gray','rgb','preview']:
        if len(list((result/name).glob('*.png')))!=1000:raise RuntimeError('Incomplete saved '+name)
    register('VAL1000' if split=='val' else 'TEST1000');compare(split)
    if split=='test':formal_comparison()


def compare(split='val'):
    baseline=BASELINE if split=='val' else Path('test_result/lwpd_qxs_full_v1/e2_seed42_test1000')
    result=RESULT if split=='val' else TEST_RESULT
    group=GROUP if split=='val' else 'QXS-TEST1000-PAPER-CLAMP-v1'
    registry=json.loads(Path('EXPERIMENT_REGISTRY.json').read_text(encoding='utf-8'))
    entries=[next(e for e in registry['experiments'] if e['result_dir']==p.as_posix()) for p in [baseline,result]]
    if any(e['comparable_group']!=group for e in entries):raise RuntimeError('Different comparison groups')
    metas=[json.loads((p/'run_metadata.json').read_text(encoding='utf-8')) for p in [baseline,result]]
    for key in ['split_manifest_sha256','split','offset','evaluated_pairs','metric_profile','output_range_policy','selected_scene_ids']:
        if metas[0][key]!=metas[1][key]:raise RuntimeError('Comparison mismatch: '+key)
    b,bm=read_metrics(baseline/'metrics.csv');c,cm=read_metrics(result/'metrics.csv')
    rows=[paired_record(k,b,c,bm,cm) for k in METRICS]
    folder=result/'comparison_vs_e2'
    if folder.exists():
        print('SKIP existing comparison:',folder);return
    folder.mkdir()
    with (folder/'paired_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f'# E2与E2 + LL-context：固定{split}1000配对比较','',
           '候选额外训练12轮，不能将收益完全归因于LL-context模块。区间为逐图配对正态近似，不代表训练种子方差。','',
           '|指标|原E2|E2+LL-context|有利相对变化|有利绝对差95%区间|改善比例|',
           '|---|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['metric']}|{r['baseline']:.6f}|{r['candidate']:.6f}|{r['beneficial_relative_percent']:+.3f}%|[{r['paired_ci_low']:+.6f}, {r['paired_ci_high']:+.6f}]|{r['paired_win_percent']:.1f}%|")
    lines+=['','重点联合查看QAB/F、NAB/F、结构指标与SAR独有目标；不沿用PRL筛选阈值，不自动推进test。']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def formal_comparison():
    """Refresh required canonical five-method table and add this result separately."""
    import build_five_method_test1000_comparison as five
    registry=json.loads(Path('EXPERIMENT_REGISTRY.json').read_text(encoding='utf-8'))
    for source in five.SOURCES:
        folder=source['path'].parent.resolve()
        entries=[e for e in registry['experiments'] if (ROOT/e['result_dir']).resolve()==folder]
        if not entries or any(e['comparable_group']!='QXS-TEST1000-PAPER-CLAMP-v1' for e in entries):
            raise RuntimeError('Canonical five-method comparison protocol mismatch')
    five.main()
    records=[]
    for source in five.SOURCES:
        values=five.read_csv_source(source['path']) if source['type']=='csv' else five.read_summary_source(source['path'],source['key'])
        records.append((source['method'],values))
    for name,path in [('E2',Path('test_result/lwpd_qxs_full_v1/e2_seed42_test1000/metrics.csv')),
                      ('E2 + LL-context',TEST_RESULT/'metrics.csv')]:
        records.append((name,five.read_csv_source(path)))
    out=TEST_RESULT/'formal_comparison'
    if out.exists():print('SKIP existing formal table:',out);return
    out.mkdir()
    lines=['# 五方法、原E2与LL-context：正式test1000','',
           '|方法|'+'|'.join(METRICS)+'|','|---|'+'---:|'*len(METRICS)]
    flat=[]
    for name,values in records:
        lines.append('|'+name+'|'+'|'.join(f"{values[k]['mean']:.4f}" for k in METRICS)+'|')
        flat.append({'method':name,**{f'{k}_{stat}':values[k][stat] for k in METRICS for stat in ['mean','std']}})
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','smoke','train','val','train-val','test','compare'])
    args=parser.parse_args();os.chdir(ROOT)
    if args.action=='check':
        run([sys.executable,'-m','unittest','tests.test_lwpd_ll_context'])
    elif args.action=='smoke':
        folder=Path('models/lwpd_e2_llctx_smoke_v1/seed42')
        if folder.exists():raise FileExistsError('Smoke directory exists; preserve it')
        run([sys.executable,*train_args(smoke=True)]);register('SMOKE')
    elif args.action=='train':train()
    elif args.action=='val':validate()
    elif args.action=='train-val':train();validate()
    elif args.action=='test':validate('test')
    else:compare()


if __name__=='__main__':main()
