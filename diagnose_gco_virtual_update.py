"""GCO-E2 one-step matched-displacement virtual-update experiment.

Compute direction ONLY from locked train1024 scenes; evaluate unchanged baseline
and temporary Detail-only steps on independent val64 scenes. No optimizer,
checkpoint mutation, training schedule, or test-set contact. Each direction
has the same total parameter L2 displacement. This is a local diagnostic,
NOT evidence of improved end-to-end training or image quality.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint, loss_partition, pair_grads
from gco_gradient_policy import (
    POLICIES, apply_delta, direction_metadata, dot,
    equal_displacement, vector_norm,
)
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, training_args
from run_qxs_reference_test1000 import DATA
from train_direct_fusion import DirectFusionLosses


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path('test_result/e2_gco_virtual_update_v1/train8_val8_seed42')


def parse_fractions(raw):
    try:
        numbers = [float(x.strip()) for x in raw.split(',')]
    except ValueError as exc:
        raise ValueError('Invalid comma-separated --fractions') from exc
    if (not numbers or len(set(numbers)) != len(numbers) or
            any(not math.isfinite(x) or not 0 < x <= 0.05 for x in numbers)):
        raise ValueError('Fractions must be distinct, finite, in (0,0.05]')
    return numbers


def cached_features(model, manifest, ids, device, crop, split):
    records = []
    for index, scene in enumerate(ids):
        o, s = (to_tensor(img, device) for img in read_pair(
            locate_scene(DATA, manifest, scene)))
        if crop < o.shape[-1]:
            y = (o.shape[-2] - crop) // 2
            x = (o.shape[-1] - crop) // 2
            o, s = (v[..., y:y + crop, x:x + crop] for v in (o, s))
        with torch.no_grad():
            bo, do, bs, ds = model.encode_pair(o, s)
            fused_base = bo + model.base_fusion(bo, bs)
        records.append(dict(
            scene=str(scene), split=split, optical=o, sar=s,
            valid=(s > 0).float(), detail_o=do, detail_s=ds, fused_base=fused_base))
        print(f'encode {split} {index + 1}/{len(ids)}: {scene}', flush=True)
    return records


def losses_for_sample(model, losses, record, weight):
    detail = model.detail_fusion(record['detail_o'], record['detail_s'], record['fused_base'])
    fused = model.reconstruct(record['optical'], record['fused_base'], detail)
    optical, sar, winner = loss_partition(
        record['optical'], record['sar'], fused, record['valid'], losses, weight)
    return fused, optical, sar, winner


def training_source_gradients(model, losses, records, weight):
    params = list(model.detail_fusion.parameters())
    g_opt = [torch.zeros_like(p) for p in params]
    g_sar = [torch.zeros_like(p) for p in params]
    records_used = []
    for j, record in enumerate(records):
        with torch.enable_grad():
            fused, lo, ls, winner = losses_for_sample(model, losses, record, weight)
            reference = losses.fusion(record['optical'], record['sar'], fused,
                                      record['valid'], weight)[0]
            if not torch.allclose(lo + ls, reference, atol=1e-5, rtol=1e-5):
                raise AssertionError('Source partition did not reproduce original E2 fusion loss')
            grad_opt, grad_sar = pair_grads(params, lo, ls)
        for k, (o, s) in enumerate(zip(grad_opt, grad_sar)):
            if o is not None:
                g_opt[k].add_(o.detach(), alpha=1.0 / len(records))
            if s is not None:
                g_sar[k].add_(s.detach(), alpha=1.0 / len(records))
        records_used.append(dict(
            scene=record['scene'], optical_loss=float(lo.detach()),
            sar_loss=float(ls.detach()), **winner))
        del fused, lo, ls, reference, grad_opt, grad_sar
        print(f'grad train {j + 1}/{len(records)}', flush=True)
    return g_opt, g_sar, records_used


@torch.no_grad()
def evaluate(model, losses, records, weight):
    rows = []
    for record in records:
        _, lo, ls, winner = losses_for_sample(model, losses, record, weight)
        values = dict(scene=record['scene'], split=record['split'],
                      optical_loss=float(lo), sar_loss=float(ls),
                      total_loss=float(lo + ls), **winner)
        if not all(math.isfinite(x) for x in (
                values['optical_loss'], values['sar_loss'], values['total_loss'])):
            raise FloatingPointError('Nonfinite source loss after virtual update')
        rows.append(values)
    return rows


def summary_effect(rows):
    records = [x for x in rows if x['split'] == 'val' and x['policy'] != 'baseline']
    result = {}
    for policy in POLICIES:
        for fraction in sorted({r['fraction'] for r in records}):
            selected = [r for r in records if r['policy'] == policy
                        and r['fraction'] == fraction]
            if not selected:
                continue
            result[f'{policy}:{fraction:g}'] = {
                'val_scenes': len(selected),
                'mean_delta_optical': sum(r['delta_optical'] for r in selected) / len(selected),
                'mean_delta_sar': sum(r['delta_sar'] for r in selected) / len(selected),
                'mean_delta_total': sum(r['delta_total'] for r in selected) / len(selected),
                'val_optical_improvement_count': sum(r['delta_optical'] < 0 for r in selected),
                'val_total_improvement_count': sum(r['delta_total'] < 0 for r in selected),
            }
    return result


def run(args):
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not 1 <= args.train_scenes <= 1024 or not 1 <= args.val_scenes <= 64:
        raise ValueError('train-scenes must be 1..1024; val-scenes 1..64')
    if args.crop < 32 or args.crop > 256 or args.crop % 8:
        raise ValueError('--crop must be a multiple of 8 from 32 to 256')
    fractions = parse_fractions(args.fractions)
    manifest = source_manifest()
    train_ids = [str(x) for x in manifest['splits']['train'][:args.train_scenes]]
    val_ids = [str(x) for x in manifest['splits']['val'][:args.val_scenes]]
    if set(train_ids) & set(val_ids):
        raise AssertionError('Train/val scene IDs overlap')
    output = args.output_dir or DEFAULT_OUTPUT
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {output}')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model, checkpoint = frozen_checkpoint(device)
    losses = DirectFusionLosses().to(device).eval()
    weight = float(training_args(ORIGINAL_LOSS_PILOT).gradient_weight)
    training = cached_features(model, manifest, train_ids, device, args.crop, 'train')
    holdout = cached_features(model, manifest, val_ids, device, args.crop, 'val')
    params = list(model.detail_fusion.parameters())
    originals = [p.detach().clone() for p in params]
    model.detail_fusion.requires_grad_(True)
    grad_o, grad_s, attribution = training_source_gradients(model, losses, training, weight)
    model.detail_fusion.requires_grad_(False)
    # Do not reinterpret this as a training loss proxy: every policy uses a
    # single averaged train gradient, frozen held-out features, and one step.
    baseline_rows = evaluate(model, losses, training + holdout, weight)
    baseline = {(r['split'], r['scene']): r for r in baseline_rows}
    observations = [dict(r, policy='baseline', fraction=0.0,
                         displacement_norm=0.0,
                         delta_optical=0.0, delta_sar=0.0, delta_total=0.0)
                    for r in baseline_rows]
    target_param_norm = vector_norm(params)
    first_fraction = fractions[0]
    _, directions = equal_displacement(grad_o, grad_s, params, first_fraction)
    metadata = direction_metadata(grad_o, grad_s, directions)
    for fraction in fractions:
        magnitude, _ = equal_displacement(grad_o, grad_s, params, fraction)
        for policy, direction in directions.items():
            try:
                apply_delta(params, direction, magnitude, originals)
                actual_displacement = vector_norm([
                    p.detach() - original for p, original in zip(params, originals)])
                if abs(actual_displacement / magnitude - 1) > 1e-3:
                    raise AssertionError('Parameter displacement not matched across policies')
                changed = evaluate(model, losses, training + holdout, weight)
            finally:
                with torch.no_grad():
                    for p, original in zip(params, originals):
                        p.copy_(original)
            for row in changed:
                reference = baseline[(row['split'], row['scene'])]
                observations.append(dict(
                    row, policy=policy, fraction=fraction,
                    displacement_norm=actual_displacement,
                    delta_optical=row['optical_loss'] - reference['optical_loss'],
                    delta_sar=row['sar_loss'] - reference['sar_loss'],
                    delta_total=row['total_loss'] - reference['total_loss']))
            print(f'virtual {policy} fraction={fraction:g} done', flush=True)
    if any(not torch.equal(p, initial) for p, initial in zip(params, originals)):
        raise AssertionError('Virtual experiment did not restore original E2 parameters')
    predictions = {
        method: {
            'predicted_train_optical_delta': -min(fractions) * target_param_norm * dot(grad_o, direction),
            'predicted_train_sar_delta': -min(fractions) * target_param_norm * dot(grad_s, direction),
        } for method, direction in directions.items()
    }
    summary = dict(
        status='virtual_detail_only_no_optimizer_no_checkpoint_write',
        setup='single mean train-source gradient applied to same E2 checkpoint; separate val-only evaluation',
        train_ids=train_ids, val_ids=val_ids, patch=args.crop,
        device=str(device), seed=42,
        checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
        code_sha256=sha256_file(Path(__file__)),
        source_split_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
        gradient_weight=weight, methods=list(POLICIES),
        fractions=fractions, detail_parameter_l2_norm=target_param_norm,
        displacement_definition='fraction * L2 norm of all detail_fusion parameters; same for all four policies',
        task_definition='original E2 max-intensity/max-Sobel L1 fusion loss split by hard-max winner; no new source objectives',
        projection_definition='two-task simultaneous original-grad symmetric PCGrad; SAR-only projection; norm balance; raw',
        gradient_metadata=metadata, train_attribution=attribution,
        local_first_order_predictions_at_min_fraction=predictions,
        holdout_summary=summary_effect(observations),
        interpretation='diagnoses local one-step behavior, not retraining performance or fusion-image metric gains',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gco-virtual-', dir=output.parent) as temp:
        pending = Path(temp) / 'output'
        pending.mkdir()
        with (pending / 'virtual_updates.csv').open('x', newline='', encoding='utf-8-sig') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
        (pending / 'summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding='utf-8')
        pending.rename(output)
    print(f'Saved: {output.resolve()}', flush=True)
    print(json.dumps(summary['holdout_summary'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-scenes', type=int, default=8)
    parser.add_argument('--val-scenes', type=int, default=8)
    parser.add_argument('--crop', type=int, default=128)
    parser.add_argument('--fractions', default='0.0005,0.002')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
