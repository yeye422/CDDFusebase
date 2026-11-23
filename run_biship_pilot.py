"""Frozen optical-Base E2 Detail-head trials, validation, and CUDA smoke check."""

import argparse
from contextlib import contextmanager
import csv
import math
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from biship_detail import MAIN_CONTROLS, METHODS, BiSHIPDetailFusion, build_detail, circular_product
from diagnose_e2_architecture_r0 import MS_WEIGHTS, scale_readouts
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, TOTAL_EPOCHS
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, write_json
from train_direct_fusion import DirectFusionLosses
from train_lwpd import set_seed
from utils.dataset_os import OpticalSARDataset, digest_splits, evenly_spaced, manifest_pairs, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
CONFIG = Path('configs/biship_v1.json')
DEFAULT_RUN = Path('test_result/biship_v1/f1a_seed42')
PARENT = ORIGINAL_LOSS_PILOT.final
EVALUATION_STATS = ('inference_ms', 'peak_allocated_mib', 'detail_rms', 'detail_absmax',
                    'optical_detail_drop_image_rms', 'sar_detail_drop_image_rms')


def device_from(name):
    torch.set_num_threads(2)
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(name)


def split_for(count):
    pilot, parent = map(read_json, (ORIGINAL_LOSS_PILOT.manifest, MANIFEST))
    validate_manifest_splits(pilot)
    validate_manifest_splits(parent)
    train, val = (pilot['splits'][key] for key in ('train', 'val'))
    if (len(train) != 1024 or len(val) != 64 or pilot['source_manifest_sha256'] != sha256_file(MANIFEST)
            or any(pilot[key] != parent[key] for key in ('optical_dir', 'sar_dir'))
            or not set(val).issubset(parent['splits']['val']) or not set(train).issubset(parent['splits']['train'])):
        raise ValueError('Unexpected original-loss QXS pilot manifest')
    if count == 200:
        used = set(val)
        val = evenly_spaced([scene for scene in parent['splits']['val'] if scene not in used], count)
    splits = dict(train=train, val=val, test=[])
    return dict(optical_dir=pilot['optical_dir'], sar_dir=pilot['sar_dir'], splits=splits,
                splits_sha256=digest_splits(splits), parent_manifest=MANIFEST.as_posix(),
                parent_manifest_sha256=pilot['source_manifest_sha256'],
                original_e2_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest))


def load_frozen(device):
    saved = torch.load(PARENT, map_location='cpu', weights_only=False)
    if (saved['variant'], saved['epoch'], saved['stage']) != (
            OpticalBaseAnchoredE2.VARIANT, TOTAL_EPOCHS - 1, 'fusion'):
        raise ValueError('Expected the original-loss optical-Base E2 pilot checkpoint')
    model = OpticalBaseAnchoredE2().eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device)


def source_hashes():
    names = ('biship_detail.py', 'train_direct_fusion.py', 'direct_fusion.py', 'e2_base_anchor.py', 'lwpd.py',
             'net.py', 'utils/dataset_os.py', 'run_biship_pilot.py')
    return {name: sha256_file(ROOT / name) for name in names}


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


@contextmanager
def new_result(destination):
    if destination.exists():
        raise FileExistsError(f'Preserve existing result: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f'.{destination.name}-{uuid4().hex}')
    work.mkdir()
    try:
        yield work
        work.rename(destination)
    finally:
        if work.exists():
            shutil.rmtree(work)


def fused_inputs(model, optical, sar):
    bo, do, bs, ds = model.encode_pair(optical, sar)
    return bo, do, bs, ds, bo + model.base_fusion(bo, bs)


def train_head(model, head, pairs, config, device):
    set_seed(config['seed'])
    dataset = OpticalSARDataset(pairs, patch_size=256, training=True,
                               invalid_threshold=0, min_valid_fraction=0.9)
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        generator=generator, pin_memory=device.type == 'cuda')
    accumulation = config['accumulation_steps']
    if len(dataset) % (config['batch_size'] * accumulation):
        raise ValueError('Batch size times accumulation must divide train1024')
    criterion = DirectFusionLosses().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history = []
    for epoch in range(1, config['epochs'] + 1):
        totals = np.zeros(5)
        gradient_max, steps, skipped = 0.0, 0, 0
        head.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (optical, sar, valid, _) in enumerate(loader, 1):
            optical, sar, valid = (value.to(device, non_blocking=True) for value in (optical, sar, valid))
            with torch.autocast(device_type=device.type, enabled=amp):
                with torch.no_grad():
                    bo, do, bs, ds, base = fused_inputs(model, optical, sar)
                    reconstruct = 0.5 * (criterion.reconstruction(optical, model.reconstruct(optical, bo, do))
                                         + criterion.reconstruction(sar, model.reconstruct(sar, bs, ds), valid))
                stage_stats = {} if step == 1 else None
                detail = (head(do, ds, base, diagnostics=stage_stats) if isinstance(head, BiSHIPDetailFusion)
                          else head(do, ds, base))
                fused = model.reconstruct(optical, base, detail)
                fusion, intensity, gradient = criterion.fusion(optical, sar, fused, valid, 10.0)
                total = fusion + 0.1 * reconstruct
            losses = torch.stack((total, fusion, intensity, gradient, reconstruct)).detach().float().cpu().numpy()
            if not np.isfinite(losses).all():
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            totals += len(optical) * losses
            scaler.scale(total / accumulation).backward()
            if step == 1:
                first_batch = {key: float(value) for key, value in (stage_stats or {}).items()}
                first_batch['detail_absmax'] = float(detail.detach().abs().max())
                first_batch['detail_rms'] = float(detail.detach().float().square().mean().sqrt())
            if step % accumulation == 0:
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), config['clip_grad']))
                if math.isfinite(norm):
                    scaler.step(optimizer)
                    gradient_max = max(gradient_max, norm)
                    steps += 1
                elif not amp:
                    raise FloatingPointError(f'Nonfinite gradient at epoch {epoch}, step {step}')
                else:
                    skipped += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if steps == 0:
            raise FloatingPointError(f'No finite optimizer steps in epoch {epoch}')
        averages = dict(zip(('loss', 'fusion', 'intensity', 'gradient', 'reconstruction'), (totals / len(dataset)).tolist()))
        history.append(dict(epoch=epoch, **averages, grad_norm_max=gradient_max, steps=steps,
                            skipped_steps=skipped, **first_batch))
        print(f'epoch={epoch}/{config["epochs"]} loss={averages["loss"]:.6f} grad_max={gradient_max:.4g} '
              f'skipped={skipped}', flush=True)
    return history


def register_training(folder):
    metadata = read_json(folder / 'run_metadata.json')
    summary = read_json(folder / 'training.json')
    checkpoint, manifest = folder / 'head.pth', folder / 'split_manifest.json'
    token = sha256_file(folder / 'run_metadata.json')[:16].upper()
    metrics = {key: summary['final'][key] for key in ('loss', 'fusion', 'grad_norm_max', 'skipped_steps')}
    entry = dict(id=f'BISHIP-F1A-TRAIN-{token}', comparable_group=f'BISHIP-F1A-TRAIN-{token}',
                 dataset='QXS-SAROPT', task='training', status='training_complete',
                 method=f'BiSHIP F1-A {metadata["method"]}', split='train', offset=0, sample_count=1024,
                 checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                 parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=metadata['parent_sha256'],
                 split_manifest=manifest.as_posix(), split_manifest_sha256=sha256_file(manifest),
                 result_dir=folder.as_posix(), metadata_path=(folder / 'run_metadata.json').as_posix(),
                 metric_profile='training_loss', output_policy='float_loss', metrics=metrics,
                 metric_source=dict(type='json_paths', path=(folder / 'training.json').as_posix(),
                                    paths={key: ['final', key] for key in metrics}))
    register_many([entry])


def train(run, methods, config, device):
    run.mkdir(parents=True, exist_ok=True)
    manifest = split_for(64)
    parent_sha = sha256_file(PARENT)
    hashes = source_hashes()
    for method in methods:
        folder = run / 'heads' / method
        settings = dict(method=method, config=config, device=str(device), torch=str(torch.__version__),
                        parent_sha256=parent_sha, source_sha256=hashes)
        if folder.exists():
            verify_head(folder, settings)
            register_training(folder)
            continue
        print(f'F1-A {method}', flush=True)
        set_seed(config['seed'])
        model = load_frozen(device)
        # All heads, including the old architecture, start from new weights.
        set_seed(config['seed'])
        head = build_detail(method).to(device)
        with new_result(folder) as work:
            split = work / 'split_manifest.json'
            write_json(split, manifest)
            pairs, _ = manifest_pairs(DATA, split)
            history = train_head(model, head, pairs['train'], config, device)
            checkpoint = work / 'head.pth'
            torch.save(head.state_dict(), checkpoint)
            write_json(work / 'training.json', dict(history=history, final=history[-1]))
            write_json(work / 'run_metadata.json', dict(**settings, parameters=parameter_count(head),
                       checkpoint_sha256=sha256_file(checkpoint), split_manifest_sha256=sha256_file(split)))
        register_training(folder)


def verify_head(folder, expected):
    metadata = read_json(folder / 'run_metadata.json')
    if (any(metadata[key] != value for key, value in expected.items())
            or metadata['checkpoint_sha256'] != sha256_file(folder / 'head.pth')
            or metadata['split_manifest_sha256'] != sha256_file(folder / 'split_manifest.json')
            or read_json(folder / 'split_manifest.json') != split_for(64)):
        raise ValueError(f'Checkpoint or protocol differs from the locked F1-A trial: {folder}')
    return metadata


def nonnegative_ms_ssim(fused, optical, sar):
    def score(source):
        stats, _ = scale_readouts(fused.astype(np.float64), source.astype(np.float64), 'x')
        terms = [stats[f'cs_x_{2**level}x'] for level in range(4)] + [stats['ssim_x_16x']]
        return float(np.prod(np.maximum(terms, 0) ** MS_WEIGHTS))

    return 0.5 * (score(optical) + score(sar))


@torch.inference_mode()
def render_scene(model, optical_u8, sar_u8, device, profile=False):
    optical, sar = (to_tensor(image, device) for image in (optical_u8, sar_u8))
    timed_cuda = profile and device.type == 'cuda'
    if timed_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    _, do, _, ds, base = fused_inputs(model, optical, sar)
    head = model.detail_fusion
    detail = head(do, ds, base)
    fused = model.reconstruct(optical, base, detail)
    if timed_cuda:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000 if profile else None
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if timed_cuda else None
    if not torch.isfinite(fused).all():
        raise FloatingPointError('Nonfinite fused image')
    fusion_u8 = fused[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    rms, absmax = float(detail.square().mean().sqrt()), float(detail.abs().max())
    drops = (None, None)
    if profile:
        drops = tuple(float((model.reconstruct(optical, base, head(a, b, base)).clamp(0, 1)
                             - fused.clamp(0, 1)).square().mean().sqrt())
                      for a, b in ((torch.zeros_like(do), ds), (do, torch.zeros_like(ds))))
    return fusion_u8, (elapsed, peak, rms, absmax, *drops)


def evaluate_scene(model, pair, device, image_path, *, profile=False, archive=None):
    optical, sar = read_pair(pair)
    if archive is None:
        fused, stats = render_scene(model, optical, sar, device, profile)
        if not cv2.imwrite(str(image_path), fused):
            raise OSError(f'Cannot save {image_path}')
        values = FusionEvaluator.metric_vector(fused, optical, sar)
    else:
        archive_image, values = archive
        fused = cv2.imread(str(archive_image), cv2.IMREAD_UNCHANGED)
        if fused is None or fused.shape != optical.shape or fused.dtype != np.uint8:
            raise ValueError(f'Invalid archived E2 image: {archive_image}')
        shutil.copyfile(archive_image, image_path)
        stats = render_scene(model, optical, sar, device, True)[1] if profile else (None,) * len(EVALUATION_STATS)
    return values, (nonnegative_ms_ssim(fused, optical, sar),), stats


def screening(path, config):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = {(row['control'], row['metric']): row for row in csv.DictReader(handle) if row['candidate'] == 'B7'}
    if not all((method, 'SSIM') in rows for method in MAIN_CONTROLS[:-1]):
        return dict(decision='incomplete_main_comparisons')

    def passes(control, metric, field, predicate):
        value = rows[control, metric][field]
        return bool(value) and predicate(float(value))

    safeguards = {metric: passes('B0', metric, 'candidate_minus_control', predicate) for metric, predicate in (
        ('Nabf', lambda x: x <= 0), ('FMI', lambda x: x >= 0), ('VIF', lambda x: x >= 0),
        ('Qabf', lambda x: x >= -config['qabf_tolerance']))}
    structure = {}
    for name, target in (('SSIM', config['ssim_gain']), ('MS_SSIM', config['ms_ssim_gain'])):
        structure[name] = (passes('B0', name, 'candidate_minus_control', lambda x: x >= target)
                           and all(passes(method, name, 'ci_low', lambda x: x > 0) for method in MAIN_CONTROLS[:-1]))
    return dict(decision='review_artifacts' if all(safeguards.values()) and any(structure.values()) else 'screen_not_passed',
                safeguards=safeguards, structure=structure)


def validate_models(output, manifest, checkpoints, device, config, load, *, stage, parent_sha=None, archive=None):
    ids = [str(scene) for scene in manifest['splits']['val']]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('Expected an unambiguous validation cohort')
    with new_result(output) as work:
        write_json(work / 'split_manifest.json', manifest)
        pairs, _ = manifest_pairs(DATA, work / 'split_manifest.json')
        checkpoints_hashes = {method: sha256_file(path) for method, path in checkpoints.items()}
        results, compute = {}, {}
        val_pairs = pairs['val']
        sample = set(evenly_spaced(ids, min(8, len(ids))))
        for method in checkpoints:
            model = load(method)
            gray = work / method / 'gray'
            gray.mkdir(parents=True)
            warmup_images = read_pair(val_pairs[0])
            for _ in range(3):
                render_scene(model, *warmup_images, device)
            rows = []
            for scene, pair in zip(ids, val_pairs):
                archived = None
                if method == 'B0' and archive is not None:
                    archived = archive['gray'] / f'{scene}.png', archive['values'][scene]
                rows.append(evaluate_scene(model, pair, device, gray / f'{scene}.png',
                                           profile=scene in sample, archive=archived))
            results[method] = rows
            cost = np.asarray([row[2][:2] for row in rows if row[2][0] is not None], dtype=np.float64)
            compute[method] = dict(detail_parameters=parameter_count(model.detail_fusion),
                                   median_inference_ms=float(np.median(cost[:, 0])),
                                   max_allocated_mib=float(cost[:, 1].max()) if device.type == 'cuda' else None)
            del model
            print(f'{stage} {method}: {len(ids)} scenes', flush=True)
        ms_col = METRICS.index('MS_SSIM')
        common_ms = np.isfinite([[row[0][ms_col] for row in rows] for rows in results.values()]).all(axis=0)
        results = {method: [(values, (aux[0], aux[0] if keep else None), stats)
                            for (values, aux, stats), keep in zip(rows, common_ms)]
                   for method, rows in results.items()}
        contrasts = [(method, 'B7') for method in checkpoints if method != 'B7']
        summary = summarize(work, ids, results, contrasts, diagnostic_fields=EVALUATION_STATS,
                            source_metrics=('MS_SSIM_NONNEG', 'MS_SSIM_NONNEG_COMMON'))
        decision = screening(work / 'paired_comparison.csv', config) if stage == 'F1A' else None
        write_json(work / 'summary.json', dict(metrics=summary, compute=compute, screening=decision))
        metadata = dict(stage=stage, config=config, scene_ids=ids, split='val', sample_count=len(ids),
                        split_manifest_sha256=sha256_file(work / 'split_manifest.json'), checkpoints_sha256=checkpoints_hashes,
                        checkpoints={key: path.as_posix() for key, path in checkpoints.items()}, parent_sha256=parent_sha,
                        metric_profile=DEFAULT_PROFILE, output_policy='clamp_then_uint8_gray', device=str(device),
                        timing='FP32 full model, 3 warmups, no I/O; peak is PyTorch allocated memory',
                        profile_scene_ids=[scene for scene in ids if scene in sample],
                        ms_ssim_aux='clamp each signed contrast/SSIM term to zero before fractional power',
                        script_sha256=sha256_file(Path(__file__)))
        if archive is not None:
            metadata['archived_B0'] = archive['provenance']
        write_json(work / 'run_metadata.json', metadata)
    token = sha256_file(output / 'run_metadata.json')[:16].upper()
    group = f'QXS-BISHIP-{stage}-VAL{len(ids)}-{token}'
    entries = []
    for method, metrics in summary.items():
        folder = output / method
        entry = dict(id=f'{group}-{method}', comparable_group=group, method=f'BiSHIP {stage} {method}',
                     status='development_validation', dataset='QXS-SAROPT', task='fusion', split='val', offset=0,
                     sample_count=len(ids), split_manifest=(output / 'split_manifest.json').as_posix(),
                     split_manifest_sha256=metadata['split_manifest_sha256'], checkpoint=checkpoints[method].as_posix(),
                     checkpoint_sha256=checkpoints_hashes[method], result_dir=folder.as_posix(),
                     metadata_path=(output / 'run_metadata.json').as_posix(), metric_profile=DEFAULT_PROFILE,
                     output_policy=metadata['output_policy'], metrics=metrics,
                     metric_source=dict(type='csv_row', path=(folder / 'metrics.csv').as_posix(),
                                        match={'image': 'mean'}, columns={key: key for key in metrics}))
        if parent_sha is not None:
            entry.update(parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=parent_sha)
        entries.append(entry)
    register_many(entries)
    print(f'Results: {output / "paired_comparison.csv"}', flush=True)


def evaluate(run, output, methods, count, config, device):
    parent_sha = sha256_file(PARENT)
    hashes = source_hashes()
    folders = {method: run / 'heads' / method for method in methods}
    for method, folder in folders.items():
        verify_head(folder, dict(config=config, method=method, parent_sha256=parent_sha, source_sha256=hashes))
    checkpoints = {method: folder / 'head.pth' for method, folder in folders.items()}

    def load(method):
        head = build_detail(method).to(device).eval()
        head.load_state_dict(torch.load(checkpoints[method], map_location='cpu', weights_only=True))
        model = load_frozen(device)
        model.detail_fusion = head
        return model

    validate_models(output, split_for(count), checkpoints, device, config, load, stage='F1A', parent_sha=parent_sha)


def smoke(device):
    torch.manual_seed(42)
    model = load_frozen(device)
    manifest = split_for(64)
    pair = locate_scene(DATA, manifest, manifest['splits']['val'][0])
    optical, sar = (to_tensor(img, device) for img in read_pair(pair))
    head = build_detail('B7').to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        with torch.no_grad():
            _, do, _, ds, base = fused_inputs(model, optical, sar)
        diagnostics = {}
        detail = head(do, ds, base, diagnostics=diagnostics)
        fused = model.reconstruct(optical, base, detail)
    fused.float().square().mean().backward()
    grads = [param.grad for param in head.parameters()]
    if (detail.shape != base.shape or fused.shape != optical.shape or not torch.isfinite(fused).all()
            or any(g is None or not torch.isfinite(g).all() for g in grads)
            or any(param.grad is not None for param in model.parameters())):
        raise RuntimeError('Invalid frozen-head forward/backward')
    result = dict(scene=pair[0].stem, parameters=parameter_count(head),
                  grad_norm=float(torch.stack([g.float().norm().square() for g in grads]).sum().sqrt()),
                  detail_rms=float(detail.detach().float().square().mean().sqrt()),
                  detail_absmax=float(detail.detach().abs().max()),
                  peak_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None)
    print(result, flush=True)
    odd = torch.randn(1, 64, 17, 19, device=device)
    for method in METHODS:
        test = build_detail(method).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            value = test(odd, odd.roll(1, -1), odd)
        value.float().square().mean().backward()
        assert value.shape == odd.shape and torch.isfinite(value).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in test.parameters()), method
        print(f'{method} {parameter_count(test)} parameters; odd shape and gradients OK')
    impulse = torch.zeros(1, 1, 11, 13, device=device)
    impulse[..., -1, -1] = 1
    wrap = circular_product(impulse, impulse)
    expected = torch.zeros_like(impulse)
    expected[..., -2, -2] = 1
    torch.testing.assert_close(wrap, expected, atol=1e-6, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'train', 'val64', 'val200'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=MAIN_CONTROLS)
    args = parser.parse_args()
    os.chdir(ROOT)
    device = device_from(args.device)
    if args.action == 'smoke':
        smoke(device)
        return
    config = read_json(CONFIG)
    run = args.run_dir.resolve().relative_to(ROOT)
    methods = tuple(dict.fromkeys(args.methods))
    if args.action == 'train':
        train(run, methods, config, device)
    else:
        output = (args.output_dir or run / args.action).resolve().relative_to(ROOT)
        evaluate(run, output, methods, int(args.action[3:]), config, device)


if __name__ == '__main__':
    main()
