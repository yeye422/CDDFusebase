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
MODEL = Path('models/e2_g1_v1')
RESULT = Path('test_result/e2_g1_v1/val1000')
MANIFEST = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
INIT = Path('models/direct_fusion_qxs/qxs_direct_seed42_v1/r3/lowest_val_loss_diagnostic.pth')
ARMS = {'g1': (10., 0.)}


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
            '--gradient-loss', 'pa_directional', '--structure-weight', structure, '--reconstruction-weight', .1,
            '--init-checkpoint', INIT, '--output-dir', MODEL if not smoke else Path('models/e2_general_loss_smoke_v1'),
            '--run-name', f'{arm}_seed42', '--device', 'cuda']
    if smoke:
        args += ['--dry-run']
    return [str(x) for x in args]


def lock_protocol():
    baseline_config = Path('models/e2_general_loss_v1/b_seed42/config.json')
    if not baseline_config.exists():
        raise FileNotFoundError('Existing baseline config required: '+str(baseline_config))
    baseline = json.loads(baseline_config.read_text(encoding='utf-8'))
    command = train_args('g1')
    for i, token in enumerate(command[:-1]):
        if not token.startswith('--'):
            continue
        key = token[2:].replace('-', '_')
        if key in ('gradient_loss', 'output_dir', 'run_name'):
            continue
        value = command[i+1]
        previous = baseline.get(key)
        if key in ('data_root','split_manifest','init_checkpoint'):
            equal = Path(value).resolve() == Path(previous).resolve()
        elif isinstance(previous, (int, float)):
            equal = float(value) == float(previous)
        else:
            equal = value == previous
        if not equal:
            raise RuntimeError('Existing baseline training protocol differs: '+key)
    if baseline['init_checkpoint_sha256'] != sha256_file(INIT):
        raise RuntimeError('Baseline initialization hash mismatch')
    files = ['train_lwpd.py', 'train_direct_fusion.py', 'direct_fusion.py', 'lwpd.py',
             'utils/fusion_structure_loss.py', 'utils/Evaluator.py', 'test_OS.py',
             'run_e2_g1.py', 'utils/pa_directional_loss.py', 'phase_antisymmetry.py']
    payload = {'protocol': 'e2_g1_v1', 'init_sha256': sha256_file(INIT),
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
    eid = f'E2-G1-V1-{arm.upper()}-SEED42-{kind}'
    checkpoint = INIT if smoke else run_dir/'epoch_016.pth'
    result_dir = RESULT/f'{arm}_seed42' if validation else run_dir
    metadata = result_dir/'run_metadata.json' if validation else run_dir/'config.json'
    entry = dict(id=eid, status='engineering_smoke' if smoke else ('development_validation' if validation else 'training_complete'),
                 task='engineering' if smoke else ('fusion' if validation else 'training'), method='E2-G1 PA directional gradient (10), reconstruction (0.1)',
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



def compare():
    """User-invoked paired bootstrap of existing val1000; no inference."""
    import numpy as np
    baseline = Path('test_result/e2_general_loss_v1/val1000/b_seed42')
    candidate = RESULT/'g1_seed42'
    if not (baseline/'metrics.csv').exists():
        print('Existing E2-B val1000 missing; do not automatically retrain baseline.');return
    metas = [json.loads((p/'run_metadata.json').read_text(encoding='utf-8')) for p in (baseline,candidate)]
    for key in ('split_manifest_sha256','split','offset','evaluated_pairs','metric_profile','output_range_policy','variant'):
        if metas[0].get(key) != metas[1].get(key):
            raise RuntimeError('Incompatible validation protocol: '+key)
    registry = json.loads(Path('EXPERIMENT_REGISTRY.json').read_text(encoding='utf-8'))
    entries = [next(e for e in registry['experiments'] if e.get('result_dir') == p.as_posix()) for p in (baseline,candidate)]
    if entries[0]['comparable_group'] != entries[1]['comparable_group']:
        raise RuntimeError('Different comparable groups')
    def rows(p):
        with (p/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            return {r['image']:r for r in csv.DictReader(f) if r['image'] not in ('mean','std')}
    b,c = rows(baseline),rows(candidate)
    if set(b)!=set(c) or len(b)!=1000:
        raise RuntimeError('Expected identical 1000 validation image IDs')
    names=sorted(b)
    keys=['Nabf','Qabf','MI','VIF']
    bv=np.array([[float(b[n][k]) for k in keys] for n in names])
    cv=np.array([[float(c[n][k]) for k in keys] for n in names])
    delta=cv-bv
    rng=np.random.default_rng(42)
    samples=np.stack([delta[rng.integers(0,1000,1000)].mean(0) for _ in range(2000)])
    lo,hi=np.quantile(samples,[.025,.975],axis=0)
    folder=candidate/'comparison_vs_b_bootstrap_v1'
    if folder.exists():
        print('KEEP existing comparison:',folder);return
    folder.mkdir()
    lines=['# E2-G1 与已有 E2-B 配对验证','',
           '同一 val1000；2000 次配对 Bootstrap，随机种子42；95%百分位区间为候选减基线。',
           '逐图区间不代表训练种子方差，未校正场景相关性。','',
           '|指标|已有 B|G1|差值95%区间|','|---|---:|---:|---:|']
    for j,k in enumerate(keys):
        lines.append(f'|{k}|{bv[:,j].mean():.6f}|{cv[:,j].mean():.6f}|[{lo[j]:+.6f}, {hi[j]:+.6f}]|')
    checks = {'历史 E2-B NAB/F 上限0.067842':cv[:,0].mean()<=.067842,
              'QAB/F 不低于直接对照B':cv[:,1].mean()>=bv[:,1].mean(),
              'MI 改善':cv[:,2].mean()>bv[:,2].mean(),
              'VIF 改善':cv[:,3].mean()>bv[:,3].mean()}
    lines+=['']+[f"- {'通过' if v else '未通过'}：{k}" for k,v in checks.items()]
    lines+=['','粘贴审查中的历史 E2-B 与现有匹配 B 数值不同；本表使用现有匹配 B，历史 NAB/F 上限保持0.067842，不混用两者指标。']
    (folder/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    import os
    os.chdir(ROOT)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check','train','val','train-val','compare'])
    args=parser.parse_args()
    if args.mode=='check':
        run([sys.executable,'-m','unittest','tests.test_pa_directional_loss','tests.test_e2_structure_loss']);return
    if args.mode == "compare":
        compare();return
    lock_protocol()
    if args.mode in ('train','train-val'):train('g1')
    if args.mode in ('val','train-val'):
        validate('g1')
        compare()
    print('Completed. No held-out test split was evaluated.')


if __name__=='__main__':main()
