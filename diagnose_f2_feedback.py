"""Single-scene F2 probe: does a decoded draft improve one-query allocation?"""

import argparse
import csv
import os
from pathlib import Path

import torch

from diagnose_e2_boundary_supervision import BoundaryProbe, DATA, LABELS, ROOT, frozen_checkpoint, locate_scene, read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import MANIFEST, read_json, write_json
from utils.dataset_os import digest_splits
from utils.reference_metrics import ReferenceEvaluator


SCENE = '4880'  # Previously reviewed scene with one present and one missing boundary.
VISIBILITY = LABELS.with_name('e2_visibility.csv')
OUTPUT = Path('test_result/f2_feedback_v1/scene4880_nonwrap')
BUDGET = 0.245  # 8-bit gray levels of whole-image RMS, from the reachability audit.
SHIFT = 24  # All probed profiles remain inside the shifted field of view.
PROFILE = 'f2_feedback_optical_boundary_probe_v1'


def labels(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return {row['id']: row for row in csv.DictReader(handle) if row['scene'] == SCENE}


def rms(tensor):
    return float(tensor.square().mean().sqrt())


def readout(probe, image, polarity, indices):
    values = 255 * polarity * probe.contrast(image)[indices]
    return float(values.mean()), int((values >= 2).sum())


def dose_bound(render, baseline, direction, polarity):
    # Fixed number of trials for either sign. Keep the closest measured feasible dose.
    lower, upper = 0.0, 1.0
    best_image, best_dose = baseline, 0.0
    for _ in range(8):
        scale = (lower + upper) / 2
        candidate = render(scale * polarity * direction)
        dose = rms(255 * (candidate - baseline))
        if dose <= BUDGET:
            lower = scale
            if dose > best_dose:
                best_image, best_dose = candidate, dose
        else:
            upper = scale
    return best_image, best_dose


@torch.inference_mode()
def probe_scene(model, optical, sar, windows):
    coarse = slice(0, None, 4)
    query = slice(2, None, 4)
    audit = slice(1, None, 2)
    probes = {key: BoundaryProbe(optical, label) for key, label in windows.items()}
    encoded = model.encode_pair(optical, sar)
    base, detail = model.fuse_features(*encoded)

    def render(edit):
        return model.reconstruct(optical, base, detail + edit).clamp(0, 1)

    baseline = render(0)
    shifted = torch.zeros_like(baseline)
    shifted[..., SHIFT:] = baseline[..., :-SHIFT]
    if any(float(p.center[:, 0].min()) - 3 < SHIFT for p in probes.values()):
        raise ValueError('Shifted control would sample the padded region')
    strengths, polarities = {}, {}
    for key, probe in probes.items():
        source = 255 * probe.contrast(optical)[coarse]
        sign = source.median().sign()
        polarities[key] = sign
        strengths[key] = dict(source_fixed=float((sign * source).clamp_min(0).mean()),
                              feedback=float((sign * (source - 255 * probe.contrast(baseline)[coarse])).clamp_min(0).mean()),
                              shifted_feedback=float((sign * (source - 255 * probe.contrast(shifted)[coarse])).clamp_min(0).mean()))
    results, tested = {}, {}
    sources = [(image[0, 0] * 255).round().byte().cpu().numpy() for image in (optical, sar)]

    def nabf(image):
        return float(ReferenceEvaluator.Nabf((image[0, 0] * 255).round().byte().cpu().numpy(), *sources))

    for policy in ('source_fixed', 'feedback', 'shifted_feedback'):
        key = max(strengths, key=lambda name: strengths[name][policy])
        if key in tested:
            results[policy] = tested[key].copy()
            continue
        probe, sign = probes[key], polarities[key]
        query_mean, support = readout(probe, optical, sign, query)
        original, original_count = readout(probe, baseline, sign, audit)
        selected, dose, query_gain = baseline, 0.0, 0.0
        if support >= 3:
            mask = (1 - (probe.across / 7).square()).clamp_min(0) * probe.taper
            raw = (encoded[1] - detail) * mask
            query_before, _ = readout(probe, baseline, sign, query)
            for direction in (1, -1):
                candidate, candidate_dose = dose_bound(render, baseline, raw, direction)
                candidate_query, _ = readout(probe, candidate, sign, query)
                gain = candidate_query - query_before
                if candidate_dose >= 0.9 * BUDGET and gain > query_gain:
                    selected, dose, query_gain = candidate, candidate_dose, gain
        after, after_count = readout(probe, selected, sign, audit)
        change = 255 * (selected - baseline)
        protected_changes = {other: abs(readout(p, selected, polarities[other], audit)[0]
                                        - readout(p, baseline, polarities[other], audit)[0])
                             for other, p in probes.items() if other != key}
        tested[key] = dict(selected=key, queried_sites=1, source_confirmed_points=support,
                               source_query_mean=query_mean, query_gain=query_gain, output_rms=dose,
                               before_audit_contrast=original, after_audit_contrast=after,
                               audit_points_before=original_count, audit_points_after=after_count,
                               other_window_abs_mean_contrast_change=protected_changes,
                               outside_rms=rms(change[0, 0][(probe.across.abs() >= 7) | (probe.taper == 0)]),
                               candidate_selected=selected is not baseline, Nabf=nabf(selected))
        results[policy] = tested[key].copy()
    return strengths, results, nabf(baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(args.device)
    output = args.output_dir.resolve().relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f'Preserve existing output: {output}')
    parent = read_json(MANIFEST)
    if SCENE not in parent['splits']['val']:
        raise ValueError('The probe must use a validation scene')
    windows = labels(LABELS)
    pair = locate_scene(DATA, parent, SCENE)
    optical, sar = (to_tensor(image, device) for image in read_pair(pair))
    model, checkpoint = frozen_checkpoint(device)
    scores, results, baseline_nabf = probe_scene(model, optical, sar, windows)
    visibility = labels(VISIBILITY)
    for row in results.values():
        selected = row['selected']
        row['source_annotation'] = windows[selected]['label']
        row['archived_e2_visibility'] = visibility.get(selected, {}).get('e2_visibility')
    name = 'feedback_audit_contrast_change'
    feedback = results['feedback']
    metrics = {name: feedback['after_audit_contrast'] - feedback['before_audit_contrast']}
    output.mkdir(parents=True)
    split_path = output / 'split_manifest.json'
    splits = dict(train=[], val=[SCENE], test=[])
    write_json(split_path, dict(splits=splits, splits_sha256=digest_splits(splits),
                                optical_dir=parent['optical_dir'], sar_dir=parent['sar_dir'],
                                source_manifest=MANIFEST.as_posix(), source_manifest_sha256=sha256_file(MANIFEST)))
    checkpoint_hash = sha256_file(checkpoint)
    metadata_path = output / 'run_metadata.json'
    write_json(metadata_path, dict(scene=SCENE, checkpoint=checkpoint.as_posix(), checkpoint_sha256=checkpoint_hash,
                                  device=str(device), script_sha256=sha256_file(Path(__file__)),
                                  source_labels_sha256=sha256_file(LABELS), visibility_sha256=sha256_file(VISIBILITY),
                                  source_images_sha256=[sha256_file(path) for path in pair],
                                  whole_image_rms_limit_gray_levels=BUDGET,
                                  feedback_control_shift_px=SHIFT,
                                  scope='One previously reviewed scene, prelocalized windows, optical cue only'))
    summary_path = output / 'summary.json'
    write_json(summary_path, dict(ranking_scores=scores, policies=results, baseline_nabf=baseline_nabf,
                                  registry_metrics=metrics))
    identifier = f'QXS-F2-FEEDBACK-VAL1-{sha256_file(metadata_path)[:12].upper()}'
    register_many([dict(id=identifier, comparable_group=identifier, dataset='QXS-SAROPT', task='f2_feedback_probe',
                        status='diagnostic_fixed_subset', method='Frozen E2 F2 feedback probe', split='val', offset=0,
                        sample_count=1, checkpoint=checkpoint.as_posix(), checkpoint_sha256=checkpoint_hash,
                        split_manifest=split_path.as_posix(), split_manifest_sha256=sha256_file(split_path),
                        result_dir=output.as_posix(), metadata_path=metadata_path.as_posix(), metric_profile=PROFILE,
                        output_policy='frozen_decoder_source_detail_intervention', metrics=metrics,
                        metric_source=dict(type='json_paths', path=summary_path.as_posix(), paths={name: ['registry_metrics', name]}))])


if __name__ == '__main__':
    main()
