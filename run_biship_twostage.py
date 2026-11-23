"""From-scratch 4+12 BiSHIP training and matched validation after F1-A."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import torch

from biship_detail import MAIN_CONTROLS
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_biship import OpticalBaseAnchoredBiSHIP
from maintain_experiment_index import sha256_file
from run_biship_pilot import (CONFIG, DEFAULT_RUN, ROOT, device_from, screening, split_for,
                              validate_models)
from run_e2_rccr import EXPERIMENT as RCCR
from run_e2_source_interval_val import read_metric_rows
import run_e2_scratch_twostage as protocol
from run_qxs_reference_test1000 import read_json, write_json
from utils.dataset_os import digest_splits, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE


class BiSHIPExperiment(protocol.Experiment):
    save_continuation = True

    def build_model(self):
        return OpticalBaseAnchoredBiSHIP()


CANDIDATE = BiSHIPExperiment(
    name='e2_scratch_biship_v1', experiment_id='E2-SCRATCH-BISHIP-V1',
    variant=OpticalBaseAnchoredBiSHIP.VARIANT, evaluator='run_biship_twostage.py',
    method='光学 Base E2 + BiSHIP Detail', notes='随机初始化，4轮重建与12轮原损失融合。',
)


def pilot_evidence(root, config):
    evidence = {}
    for count in (64, 200):
        folder = root / f'val{count}'
        meta = read_json(folder / 'run_metadata.json')
        cohort = split_for(count)
        if (meta['stage'] != 'F1A' or meta['config'] != config or meta['metric_profile'] != DEFAULT_PROFILE
                or meta['scene_ids'] != [str(scene) for scene in cohort['splits']['val']]
                or set(meta['checkpoints']) != set(MAIN_CONTROLS)
                or meta['split_manifest_sha256'] != sha256_file(folder / 'split_manifest.json')
                or read_json(folder / 'split_manifest.json') != cohort):
            raise ValueError(f'Unexpected F1-A validation protocol: {folder}')
        paired = folder / 'paired_comparison.csv'
        if screening(paired, config)['decision'] != 'review_artifacts':
            raise ValueError(f'F1-A val{count} did not pass the architectural screen')
        evidence[f'val{count}'] = dict(metadata_sha256=sha256_file(folder / 'run_metadata.json'),
                                       paired_sha256=sha256_file(paired),
                                       checkpoints_sha256=meta['checkpoints_sha256'])
    if evidence['val64']['checkpoints_sha256'] != evidence['val200']['checkpoints_sha256']:
        raise ValueError('F1-A val64 and val200 must evaluate the same heads')
    return evidence


def verify_run(experiment, expected_args):
    run = experiment.run_dir
    meta = read_json(run / 'config.json')
    expected = dict(protocol.config_values(expected_args), total_epochs=protocol.TOTAL_EPOCHS,
                    stage_learning_rates=protocol.STAGE_LEARNING_RATES)
    mismatch = [key for key, value in expected.items() if meta.get(key) != value]
    if mismatch:
        raise ValueError(f'Two-stage training settings differ in {run}: {", ".join(mismatch)}')
    for name, hash_key in (('split_manifest.json', 'saved_split_manifest_sha256'),
                           ('pilot_subset.json', 'pilot_subset_sha256')):
        path = run / name
        if sha256_file(path) != meta[hash_key]:
            raise ValueError(f'Training manifest hash mismatch: {path}')
    if read_json(run / 'split_manifest.json')['splits'] != read_json(experiment.manifest)['splits']:
        raise ValueError(f'Two-stage training splits differ: {run}')


def baseline_archive(baseline):
    manifest = read_json(baseline.manifest)
    validate_manifest_splits(manifest)
    ids = [str(scene) for scene in manifest['splits']['val']]
    archive_meta = RCCR.result / 'run_metadata.json'
    archive_csv = RCCR.result / 'e2' / 'metrics.csv'
    image_meta = baseline.result / 'run_metadata.json'
    gray = baseline.result / 'gray'
    metrics, images = map(read_json, (archive_meta, image_meta))
    checkpoint_hash = sha256_file(baseline.final)
    manifest_hash = sha256_file(baseline.manifest)
    if (metrics['metric_profile'] != DEFAULT_PROFILE or metrics['checkpoint_hashes']['e2'] != checkpoint_hash
            or metrics['output_policy'] != 'clamp_then_uint8' or metrics['split_manifest_sha256'] != manifest_hash
            or metrics['scene_ids'] != ids or images['checkpoint_sha256'] != checkpoint_hash
            or images['split_manifest_sha256'] != manifest_hash or images['output_range_policy'] != 'clamp'
            or images['selected_scene_ids'] != ids or {p.stem for p in gray.glob('*.png')} != set(ids)):
        raise ValueError('Archived E2 baseline differs from the F1-B validation protocol')
    values = read_metric_rows(archive_csv, ids)
    provenance = {path.as_posix(): sha256_file(path) for path in (archive_meta, archive_csv, image_meta)}
    splits = dict(train=[], val=manifest['splits']['val'], test=[])
    val_manifest = dict(optical_dir=manifest['optical_dir'], sar_dir=manifest['sar_dir'], splits=splits,
                        splits_sha256=digest_splits(splits), parent_manifest=baseline.manifest.as_posix(),
                        parent_manifest_sha256=manifest_hash)
    return val_manifest, dict(gray=gray, values={scene: row[0] for scene, row in zip(ids, values)},
                              provenance=dict(files=provenance, gray=gray.as_posix()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'resume', 'val1000'))
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--pilot-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/biship_v1/f1b_seed42/val1000'))
    args = parser.parse_args()
    os.chdir(ROOT)
    device = device_from(args.device)
    config = read_json(CONFIG)
    evidence = pilot_evidence(args.pilot_dir, config)
    baseline = protocol.BASE_ANCHOR
    verify_run(baseline, protocol.training_args(baseline))
    candidate = CANDIDATE
    original_args = protocol.training_args
    candidate_args = original_args(candidate)
    candidate_args.biship_architecture_sha256 = sha256_file(ROOT / 'biship_detail.py')
    candidate_args.biship_pilot_evidence = evidence
    if args.action in ('train', 'resume'):
        if candidate.final.exists():
            verify_run(candidate, candidate_args)
        with patch.object(protocol, 'training_args', return_value=candidate_args):
            protocol.train(candidate, resume=args.action == 'resume')
        return

    verify_run(candidate, candidate_args)
    manifest, archive = baseline_archive(baseline)
    checkpoints = {'B0': baseline.final, 'B7': candidate.final}

    def load(method):
        model = OpticalBaseAnchoredE2() if method == 'B0' else candidate.build_model()
        saved = torch.load(checkpoints[method], map_location='cpu', weights_only=False)
        if (saved['variant'], saved['epoch'], saved['stage']) != (model.variant, protocol.TOTAL_EPOCHS - 1, 'fusion'):
            raise ValueError(f'Unexpected F1-B checkpoint: {checkpoints[method]}')
        model.load_state_dict(saved['model'], strict=True)
        return model.to(device).eval().requires_grad_(False)

    output = args.output_dir.resolve().relative_to(ROOT)
    validate_models(output, manifest, checkpoints, device, config, load, stage='F1B', archive=archive)
    write_json(output / 'f1a_evidence.json', evidence)


if __name__ == '__main__':
    main()
