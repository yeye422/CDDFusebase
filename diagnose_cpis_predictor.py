"""Evaluate CPIS R1.1 prediction controls and residual structure on val200."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from cpis_r11 import (ROOT, SOURCES, features, frozen_e2, heads, load_images,
                      protocol, read_json, register, runtime, sha256_file, write_json)
from rct_predictive_transfer import mismatched_partners, paired_stats
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT
from utils.dataset_os import evenly_spaced


CONTROLS = ('p0', 'p2', 'p2_aligned', 'p1_shuffled')
REGIONS = ('shared', 'optical_only', 'sar_only_candidate', 'flat')
EXCLUSIVE = dict(zip(SOURCES, REGIONS[1:3]))


def source_structure(image):
    smooth = F.avg_pool2d(image, 5, stride=1, padding=2, count_include_pad=False)
    k = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    dx, dy = F.conv2d(smooth, torch.stack((k, k.T))[:, None], padding=1).unbind(dim=1)
    xx, xy, yy = F.avg_pool2d(torch.stack((dx.square(), dx * dy, dy.square()), dim=1),
                            8, stride=2, padding=3, count_include_pad=False)[0]
    strength = (xx + yy).sqrt()
    coherence = ((xx - yy).square() + 4 * xy.square()).sqrt() / (xx + yy).clamp_min(1e-12)
    return strength, coherence


def region_masks(images):
    def classify(image):
        strength, coherence = source_structure(image)
        core = strength[4:-4, 4:-4].contiguous().flatten()
        low, high = torch.quantile(core, strength.new_tensor([0.40, 0.75]))
        return (strength > high.clamp_min(1 / 255)) & (coherence >= 0.5), strength <= low

    (o_edge, o_low), (s_edge, s_low) = map(classify, images)
    masks = (o_edge & s_edge, o_edge & s_low, s_edge & o_low, o_low & s_low)
    valid = torch.zeros_like(o_edge)
    valid[4:-4, 4:-4] = True
    return {name: mask & valid for name, mask in zip(REGIONS, masks)}


def rough_energy_share(coefficients):
    # Within-block variance fraction of signed coefficients on the Haar cell grid.
    center = coefficients[..., 4:-4, 4:-4]
    total_energy = center.square().mean().item()
    coarse_energy = F.avg_pool2d(center, 2).square().mean().item()
    return min(1.0, max(0.0, 1 - coarse_energy / total_energy)) if total_energy > 0 else None


def residual_readout(row, source, target, residual, masks, min_cells):
    row[f'{source}_rough_energy_share'] = rough_energy_share(residual)
    row[f'{source}_source_rough_energy_share'] = rough_energy_share(target)
    local_mean = F.avg_pool2d(residual.abs(), 3, stride=1, padding=1, count_include_pad=False)
    envelope = local_mean.square().mean(1)[0]
    core = envelope[4:-4, 4:-4]
    shifted = envelope.clone()
    shifted[4:-4, 4:-4] = core.roll(tuple(n // 2 for n in core.shape), dims=(-2, -1))
    target_energy = target[0].square().mean(0)
    residual_energy = residual[0].square().mean(0)
    for region, mask in masks.items():
        if row[f'{region}_cells'] < min_cells:
            continue
        prefix = f'{source}_{region}'
        row[f'{prefix}_residual_energy'] = residual_energy[mask].mean().item()
        row[f'{prefix}_source_energy'] = target_energy[mask].mean().item()
        row[f'{prefix}_envelope'] = envelope[mask].mean().item()
        row[f'{prefix}_shifted_envelope'] = shifted[mask].mean().item()
    return envelope.sqrt()


def save_preview(path, images, envelopes, masks):
    def gray(image):
        return (image.clamp(0, 1) * 255).round().byte().cpu().numpy()

    sources = [gray(F.avg_pool2d(image, 2)[0, 0]) for image in images]
    scale = torch.quantile(torch.stack(envelopes).flatten(), 0.99).clamp_min(1e-12)
    maps = [gray(envelope / scale) for envelope in envelopes]
    mask_images = [gray(mask.float()) for mask in masks.values()]
    ok, png = cv2.imencode('.png', np.block([sources + maps, mask_images]))
    if not ok:
        raise OSError(f'Failed to encode {path}')
    path.write_bytes(png.tobytes())


def evaluate_scene(scene, partner, images, values, other, native, paired, shuffled, means, config):
    row = dict(scene=scene, shuffled_partner=partner, **native)
    masks = region_masks(images)
    row.update({f'{region}_cells': int(mask.sum()) for region, mask in masks.items()})
    envelopes = []
    for i, (source, target) in enumerate(zip(SOURCES, values)):
        aligned_input, shuffled_input = values[1 - i], other[1 - i]
        aligned = paired[i](aligned_input)
        predictions = dict(p0=means[i, :, None, None], p1=aligned, p2=shuffled[i](shuffled_input),
                           p2_aligned=shuffled[i](aligned_input), p1_shuffled=paired[i](shuffled_input))
        core = target[..., 4:-4, 4:-4]
        energy = core.square().mean().item()
        row[f'{source}_target_energy'] = energy
        # All five conditions share the same denominator and valid scene cohort.
        for method, pred in predictions.items():
            error = (pred - target)[..., 4:-4, 4:-4].square().mean().item()
            row[f'{source}_{method}_mse'] = error
            row[f'{source}_{method}_nmse'] = error / energy if energy > 1e-12 else None
        envelopes.append(residual_readout(row, source, target, target - aligned, masks, config['min_region_cells']))
    return row, envelopes, masks


def compare(rows, candidate, reference, config, lower=False):
    return paired_stats([row.get(candidate) for row in rows], [row.get(reference) for row in rows],
                        lower_is_better=lower, seed=config['seed'])


def summarize(rows, config):
    def passes(stats, min_scenes):
        return (stats['n'] >= min_scenes and stats['relative_gain'] >= config['min_relative_gain']
                and stats['ci_low'] > 0 and stats['median_gain'] > 0
                and stats['improved'] / stats['n'] >= config['min_scene_win_fraction'])

    prediction, residual, metrics = {}, {}, {}
    for source in SOURCES:
        prediction[source] = {}
        for control in CONTROLS:
            stats = compare(rows, f'{source}_p1_nmse', f'{source}_{control}_nmse', config, lower=True)
            stats['passes'] = passes(stats, len(rows))
            prediction[source][control] = stats
            if stats['n']:
                metrics[f'{source}_p1_vs_{control}_relative_gain'] = stats['relative_gain']
        prefix = f'{source}_{EXCLUSIVE[source]}'
        halves = []
        for parity in (0, 1):
            subset = rows[parity::2]
            comparisons = dict(
                versus_flat=compare(subset, f'{prefix}_envelope', f'{source}_flat_envelope', config),
                versus_shifted=compare(subset, f'{prefix}_envelope', f'{prefix}_shifted_envelope', config))
            roughness = [row[f'{source}_rough_energy_share'] for row in subset
                         if row[f'{source}_rough_energy_share'] is not None]
            rough_mean = sum(roughness) / len(roughness) if roughness else None
            enough_roughness = (len(roughness) == len(subset)
                                and rough_mean <= config['max_sar_rough_energy_share'])
            halves.append(dict(comparisons=comparisons, rough_energy_share_mean=rough_mean,
                               rough_energy_share_scenes=len(roughness),
                               passes=all(passes(stat, config['min_region_scenes_per_half'])
                                          for stat in comparisons.values())
                               and (source != 'sar' or enough_roughness)))
            if rough_mean is not None:
                metrics[f'{source}_half{parity + 1}_rough_energy_share'] = rough_mean
        residual[source] = halves
    prediction_pass = all(stats['passes'] for source in prediction.values() for stats in source.values())
    residual_pass = all(half['passes'] for source in residual.values() for half in source)
    if not prediction_pass:
        decision = 'stop_prediction_not_supported'
    elif not residual_pass:
        decision = 'stop_residual_evidence_insufficient'
    else:
        decision = 'review_previews_before_r12'
    return dict(decision=decision, prediction=prediction, residual=residual, registry_metrics=metrics)


@torch.inference_mode()
def evaluate(model, manifest, device, output, state):
    config = state['config']
    paired, shuffled = (heads(state['channels'], config['hidden_channels']).to(device).eval() for _ in range(2))
    paired.load_state_dict(state['paired'])
    shuffled.load_state_dict(state['shuffled'])
    means = state['p0_means'].to(device)
    scenes = [str(scene) for scene in manifest['splits']['val']]
    partners = mismatched_partners(scenes, config['seed'] + 1)
    previews = set(evenly_spaced(scenes, 8))
    (output / 'previews').mkdir()

    def prepare(scene):
        images = load_images(manifest, [scene], device)
        values, native = features(model, images, native=True)
        return images, values, native

    first = scene = scenes[0]
    first_data = current = prepare(first)
    rows = {}
    # mismatched_partners returns one cycle. Each scene is encoded only once.
    for _ in scenes:
        partner = partners[scene]
        following = first_data if partner == first else prepare(partner)
        images, values, native = current
        row, envelopes, masks = evaluate_scene(scene, partner, images, values, following[1], native,
                                               paired, shuffled, means, config)
        rows[scene] = row
        if scene in previews:
            save_preview(output / 'previews' / f'{scene}.png', images, envelopes, masks)
        scene, current = partner, following
    return [rows[scene] for scene in scenes]


@torch.inference_mode()
def smoke(model, manifest, device):
    scene = manifest['splits']['val'][0]
    images = load_images(manifest, [scene], device)
    values, native = features(model, images, native=True)
    for i in range(len(SOURCES)):
        modified = tuple(image.roll(37, dims=-1) if j != i else image for j, image in enumerate(images))
        torch.testing.assert_close(features(model, modified)[i], values[i], rtol=0, atol=0)
    predictor = heads(values[0].shape[1], 48).to(device).eval()
    row, _, _ = evaluate_scene(str(scene), None, images, values, values, native, predictor, predictor,
                               torch.zeros_like(torch.stack([x[0, :, 0, 0] for x in values])),
                               dict(min_region_cells=1))
    if any(not torch.isfinite(torch.tensor(val)) for val in row.values() if isinstance(val, float)):
        raise ValueError('Unexpected nonfinite smoke readout')
    print(f'Scene {scene}: frozen source isolation, head shapes and residual readouts verified; {native}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if not args.smoke and (args.train_dir is None or args.output_dir is None):
        parser.error('val200 requires --train-dir and --output-dir')
    os.chdir(ROOT)
    manifest = protocol()
    device = runtime(args.device)
    model = frozen_e2(device)
    if args.smoke:
        smoke(model, manifest, device)
        return
    output = args.output_dir.resolve().relative_to(ROOT)
    train_dir = args.train_dir.resolve().relative_to(ROOT)
    metadata = read_json(train_dir / 'run_metadata.json')
    checkpoint, train_split = train_dir / 'predictors.pth', train_dir / 'split_manifest.json'
    e2_checkpoint = ORIGINAL_LOSS_PILOT.final
    hashes = dict(checkpoint_sha256=sha256_file(checkpoint), split_manifest_sha256=sha256_file(train_split),
                  e2_checkpoint_sha256=sha256_file(e2_checkpoint), model_source_sha256=sha256_file(ROOT / 'cpis_r11.py'))
    if any(metadata[key] != value for key, value in hashes.items()) or read_json(train_split) != manifest:
        raise ValueError('Training artifacts do not match the locked R1.1 protocol')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state['config'] != metadata['config'] or state['e2_checkpoint_sha256'] != hashes['e2_checkpoint_sha256']
            or state['split_manifest_sha256'] != hashes['split_manifest_sha256']):
        raise ValueError('Predictor checkpoint provenance does not match its metadata')
    output.mkdir(parents=True, exist_ok=False)
    split_path = output / 'split_manifest.json'
    write_json(split_path, manifest)
    rows = evaluate(model, manifest, device, output, state)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, state['config'])
    write_json(output / 'summary.json', summary)
    write_json(output / 'run_metadata.json', dict(
        **hashes, config=state['config'], device=str(device), torch=str(torch.__version__), cuda=torch.version.cuda,
        checkpoint=checkpoint.as_posix(), e2_checkpoint=e2_checkpoint.as_posix(),
        split_manifest=split_path.as_posix(), train_split_manifest=train_split.as_posix(),
        train_metadata_sha256=sha256_file(train_dir / 'run_metadata.json'),
        per_image_sha256=sha256_file(output / 'per_image.csv'), script_sha256=sha256_file(Path(__file__))))
    print(summary['decision'])
    register(output, 'val', split_path, checkpoint, summary['registry_metrics'], e2_checkpoint)


if __name__ == '__main__':
    main()
