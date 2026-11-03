"""User-invoked matched E2 loss training/validation workflow. No held-out test."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from train_direct_fusion import sha256_file
from compare_lwpd_prl import METRICS, paired_record, read_metrics

ROOT = Path(__file__).resolve().parent
MODEL = Path('models/e2_general_loss_v1')
RESULT = Path('test_result/e2_general_loss_v1/val1000')
MANIFEST = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
INIT = Path('models/direct_fusion_qxs/qxs_direct_seed42_v1/r3/lowest_val_loss_diagnostic.pth')
ARMS = {'b': (10., 0.), 's': (10., .5), 'g': (5., 0.), 'gs': (5., .5)}


def run(arguments):
    print('RUN:', subprocess.list2cmdline([str(x) for x in arguments]), flush=True)
    subprocess.run([str(x) for x in arguments], cwd=ROOT, check=True)


def train_args(arm, smoke=False):
    gradient, structure = ARMS[arm]
    args = ['train_lwpd.py', '--variant', 'lwpd_e2', '--data-root', '../datasets/QXSLAB_SAROPT',
            '--split-manifest', MANIFEST, '--train-limit', 1 if smoke else 0,
            '--val-limit', 1 if smoke else 0, '--patch-size', 256, '--batch-size', 1 if smoke else 2,
            '--accumulation-steps', 4, '--num-workers', 0, '--invalid-threshold', 0,
            '--min-valid-fraction', .9, '--seed', 42, '--warmup-epochs', 2,
            '--local-epochs', 4, '--joint-epochs', 10,
            '--warmup-lwpd-lr', 1e-4, '--local-lwpd-lr', 1e-4, '--joint-lwpd-lr', 5e-5,
            '--warmup-fusion-lr', 5e-5, '--local-fusion-lr', 1e-4, '--joint-fusion-lr', 5e-5,
            '--decoder-lr', 2e-5, '--encoder-tail-lr', 1e-5, '--encoder-early-lr', 2e-6,
            '--weight-decay', 1e-4, '--clip-grad', 1, '--gradient-weight', gradient,
            '--structure-weight', structure, '--reconstruction-weight', .1,
            '--init-checkpoint', INIT, '--output-dir', MODEL if not smoke else Path('models/e2_general_loss_smoke_v1'),
            '--run-name', f'{arm}_seed42', '--device', 'cuda']
    if smoke:
        args += ['--dry-run']
    return [str(x) for x in args]


def lock_protocol():
    files = ['train_lwpd.py', 'train_direct_fusion.py', 'direct_fusion.py', 'lwpd.py',
             'utils/fusion_structure_loss.py', 'utils/Evaluator.py', 'test_OS.py',
             'run_e2_general_loss.py']
    payload = {'protocol': 'e2_general_loss_v1', 'init_sha256': sha256_file(INIT),
               'split_manifest_sha256': sha256_file(MANIFEST),
               'sources_sha256': {p: sha256_file(Path(p)) for p in files},
               'arm_commands': {arm: train_args(arm) for arm in ARMS},
               'metric_profile': 'paper', 'output_policy': 'clamp_then_uint8'}
    MODEL.mkdir(parents=True, exist_ok=True)
    lock = MODEL/'workflow_protocol.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8')) != payload:
            raise RuntimeError('Protocol/source changed. Use a NEW version and output directory; existing results cannot be reused.')
    else:
        lock.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def record(arm, validation=False, smoke=False):
    """Register completed existing work only, then validate/render the index."""
    run_dir = (Path('models/e2_general_loss_smoke_v1') if smoke else MODEL)/f'{arm}_seed42'
    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    kind = 'SMOKE' if smoke else ('VAL1000' if validation else 'TRAIN8000')
    eid = f'E2-GENERAL-LOSS-V1-{arm.upper()}-SEED42-{kind}'
    checkpoint = INIT if smoke else run_dir/'epoch_016.pth'
    result_dir = RESULT/f'{arm}_seed42' if validation else run_dir
    metadata = result_dir/'run_metadata.json' if validation else run_dir/'config.json'
    entry = dict(id=eid, status='engineering_smoke' if smoke else ('development_validation' if validation else 'training_complete'),
                 task='engineering' if smoke else ('fusion' if validation else 'training'), method=f'E2 loss {arm}: grad={ARMS[arm][0]}, structure={ARMS[arm][1]}',
                 comparable_group='QXS-VAL1000-E2-GENERAL-LOSS-v1' if validation else f'QXS-E2-GENERAL-LOSS-{kind}-{arm}-v1',
                 dataset='QXS-SAROPT', split_manifest=MANIFEST.as_posix(), split_manifest_sha256=sha256_file(MANIFEST),
                 split='val' if validation or smoke else 'train', offset=0, sample_count=1000 if validation else (1 if smoke else 8000),
                 checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                 parent_checkpoint=INIT.as_posix(), parent_checkpoint_sha256=sha256_file(INIT),
                 metric_profile='paper' if validation else 'training_loss', output_policy='clamp_then_uint8' if validation else 'float32_loss',
                 result_dir=result_dir.as_posix(), metadata_path=metadata.as_posix(),
                 loss_weights={'gradient': ARMS[arm][0], 'structure': ARMS[arm][1], 'reconstruction': .1},
                 notes='单样本前向反传工程检查，无优化器更新，无性能结论。' if smoke else '固定E2、R3初始化与16轮预算；不访问test。验证使用原paper指标，评价器未改变。')
    if validation:
        with (result_dir/'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            mean = next(r for r in csv.DictReader(f) if r['image']=='mean')
        entry['metrics'] = {k: float(mean[k]) for k in METRICS}
        entry['metric_source'] = {'type': 'csv_row', 'path': (result_dir/'metrics.csv').as_posix(),
                                  'match': {'image': 'mean'}, 'columns': {k:k for k in METRICS}}
    elif smoke:
        source = run_dir/'dry_run_diagnostics.json'
        diagnostics = json.loads(source.read_text(encoding='utf-8'))
        entry['metrics'] = {'loss': diagnostics['loss'], 'structure': diagnostics['components']['structure']}
        entry['metric_source'] = {'type': 'json_paths', 'path': source.as_posix(),
                                  'paths': {'loss': ['loss'], 'structure': ['components','structure']}}
    else:
        source = run_dir/'history.csv'
        with source.open(encoding='utf-8-sig') as f:
            last = next(r for r in csv.DictReader(f) if r['epoch']=='16')
        entry['metrics'] = {k: float(last[k]) for k in ['train_loss','val_loss']}
        entry['metric_source'] = {'type': 'csv_row', 'path': source.as_posix(),
                                  'match': {'epoch': '16'}, 'columns': {k:k for k in entry['metrics']}}
    registry_path = Path('EXPERIMENT_REGISTRY.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    previous = next((e for e in registry['experiments'] if e['id']==eid), None)
    if previous is not None and previous != entry:
        # Recover the initial smoke registration interrupted by index rendering;
        # never replace an existing raw-metric-backed evaluation.
        if 'metric_source' not in previous and all(previous.get(k)==entry.get(k) for k in
                ['checkpoint_sha256','result_dir','loss_weights','metadata_path']):
            previous.clear();previous.update(entry)
            registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        else:
            raise RuntimeError('Registered experiment differs: '+eid)
    if previous is None:
        registry['experiments'].append(entry)
        from datetime import datetime, timezone, timedelta
        registry['updated_at'] = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    run([sys.executable, 'maintain_experiment_index.py', '--check', '--render'])


def train(arm):
    folder = MODEL/f'{arm}_seed42'
    checkpoint = folder/'epoch_016.pth'
    if checkpoint.exists():
        print('SKIP completed training:', checkpoint)
    else:
        epochs = sorted(folder.glob('epoch_*.pth')) if folder.exists() else []
        args = train_args(arm)
        if epochs:
            args += ['--resume', str(epochs[-1])]
        run([sys.executable, '-u', *args])
    if not checkpoint.exists():
        raise RuntimeError('Final checkpoint missing: '+str(checkpoint))
    record(arm)


def validate(arm):
    checkpoint = MODEL/f'{arm}_seed42/epoch_016.pth'
    folder = RESULT/f'{arm}_seed42'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if (folder/'metrics.csv').exists():
        print('CHECK existing evaluation:', folder)
    else:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Partial validation directory exists; preserve it and use a new output version: '+str(folder))
        run([sys.executable, '-u', 'test_OS.py', '--data-root', '../datasets/QXSLAB_SAROPT',
             '--split-manifest', MANIFEST, '--split', 'val', '--variant', 'auto',
             '--checkpoint', checkpoint, '--output-dir', folder, '--offset', 0, '--limit', 1000,
             '--metric-profile', 'paper', '--output-range-policy', 'clamp', '--timing-warmup', 10])
    meta = json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
    expected = {'checkpoint_sha256': sha256_file(checkpoint), 'split_manifest_sha256': sha256_file(MANIFEST),
                'metric_profile': 'paper', 'output_range_policy': 'clamp', 'split': 'val',
                'offset': 0, 'evaluated_pairs': 1000, 'variant': 'lwpd_e2'}
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Validation metadata mismatch: '+str(folder))
    with (folder/'metrics.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['image'] not in ('mean','std')]
    if len(rows)!=1000 or len({r['image'] for r in rows})!=1000:
        raise RuntimeError('Expected exactly 1000 distinct validation images')
    if any(not math.isfinite(float(r[k])) for r in rows for k in METRICS):
        raise RuntimeError('Non-finite validation metric; investigate before comparing')
    record(arm, validation=True)


def compare(arm):
    if arm=='b': return
    paths = [RESULT/f'{a}_seed42/metrics.csv' for a in ['b', arm]]
    if not all(p.exists() for p in paths):
        print('Paired comparison pending matched b validation.');return
    baseline, candidate = [read_metrics(p) for p in paths]
    records = [paired_record(k, baseline[0], candidate[0], baseline[1], candidate[1]) for k in METRICS]
    folder = RESULT/f'{arm}_seed42/comparison_vs_b'
    if folder.exists():
        print('KEEP existing paired comparison:', folder);return
    folder.mkdir()
    with (folder/'paired_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    by={r['metric']:r for r in records}
    checks={'SSIM及MS-SSIM均不下降':all(by[k]['beneficial_change']>=0 for k in ['SSIM','MS_SSIM']),
            '信息及边缘指标下降不超过0.5%':all(by[k]['beneficial_relative_percent']>=-.5 for k in ['MI','FMI','VIF','Qabf','SCD']),
            'NAB/F不升高':by['Nabf']['beneficial_change']>=0}
    lines=['# E2通用损失与匹配基线的逐图对照','',
           '同一固定val1000，原paper指标。95%区间为配对均值差的正态近似；不是种子方差。','',
           '| 指标 | 基线B | 候选 | 有利相对变化 | 有利绝对差95%区间 | 改善比例 |',
           '|---|---:|---:|---:|---|---:|']
    for r in records:
        lines.append(f"| {r['metric']} | {r['baseline']:.6f} | {r['candidate']:.6f} | {r['beneficial_relative_percent']:+.3f}% | [{r['paired_ci_low']:+.6f}, {r['paired_ci_high']:+.6f}] | {r['paired_win_percent']:.1f}% |")
    lines+=['','## 均值工程筛选','']+[f"- {'通过' if v else '未通过'}：{k}" for k,v in checks.items()]
    lines+=['','最终结论仍需固定样本视觉检查及多种子复核；不授权test1000。']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    import os
    os.chdir(ROOT)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check','smoke','train','val','train-val'])
    parser.add_argument('arm', nargs='?', default='pair', choices=['b','s','g','gs','pair','all'])
    args=parser.parse_args()
    if args.mode=='check':
        run([sys.executable,'-m','unittest','tests.test_e2_structure_loss']);return
    if args.mode=='smoke':
        diagnostic=Path('models/e2_general_loss_smoke_v1/s_seed42/dry_run_diagnostics.json')
        if not diagnostic.exists():
            run([sys.executable,'-u',*train_args('s',smoke=True)])
        else:
            print('KEEP completed single-sample smoke:',diagnostic)
        record('s',smoke=True);return
    lock_protocol()
    arms=['b','s'] if args.arm=='pair' else list(ARMS) if args.arm=='all' else [args.arm]
    for arm in arms:
        if args.mode in ('train','train-val'):train(arm)
        if args.mode in ('val','train-val'):validate(arm)
    if args.mode in ('val','train-val'):
        for arm in arms:compare(arm)
    print('Completed. No held-out test split was evaluated.')


if __name__=='__main__':main()
