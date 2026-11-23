"""Shared frozen features, prediction heads and provenance for CPIS R1.1."""

import json
from pathlib import Path

import torch
from torch import nn

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, TOTAL_EPOCHS
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json
from utils.dataset_os import digest_splits, evenly_spaced, validate_manifest_splits


ROOT = Path(__file__).resolve().parent
SOURCES = ('optical', 'sar')
CONFIG = Path('configs/cpis_r11.json')
PROFILE = 'cpis_r11_source_only_haar_detail_v1'


class Predictor(nn.Sequential):
    def __init__(self, channels, hidden):
        super().__init__(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )


def heads(channels, hidden):
    return nn.ModuleList(Predictor(channels, hidden) for _ in SOURCES)


def protocol():
    pilot = read_json(ORIGINAL_LOSS_PILOT.manifest)
    parent = read_json(MANIFEST)
    validate_manifest_splits(pilot)
    validate_manifest_splits(parent)
    parent_hash = sha256_file(MANIFEST)
    train = pilot['splits']['train']
    if (len(train) != 1024 or not set(train).issubset(parent['splits']['train'])
            or pilot['source_manifest_sha256'] != parent_hash
            or any(pilot[key] != parent[key] for key in ('optical_dir', 'sar_dir'))):
        raise ValueError('The original-loss pilot must use the fixed QXS train1024')
    pilot_val = set(pilot['splits']['val'])
    available = [scene for scene in parent['splits']['val'] if scene not in pilot_val]
    splits = dict(train=train, val=evenly_spaced(available, 200), test=[])
    return dict(dataset=pilot['dataset'], optical_dir=pilot['optical_dir'], sar_dir=pilot['sar_dir'],
                splits=splits, splits_sha256=digest_splits(splits),
                parent_manifest=MANIFEST.as_posix(), parent_manifest_sha256=parent_hash,
                e2_train_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
                e2_train_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                excluded_e2_pilot_val=pilot['splits']['val'])


def frozen_e2(device):
    path = ORIGINAL_LOSS_PILOT.final
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('variant'), saved.get('epoch'), saved.get('stage')) != (
            ORIGINAL_LOSS_PILOT.variant, TOTAL_EPOCHS - 1, 'fusion'):
        raise ValueError('Expected the final optical-Base E2 original-loss pilot')
    model = ORIGINAL_LOSS_PILOT.build_model().eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    if model.lwpd.mode != 'gated':
        raise ValueError('R1.1 requires E2 gated Haar LWPD')
    return model.to(device)


def load_images(manifest, scenes, device):
    pairs = [read_pair(locate_scene(DATA, manifest, scene)) for scene in scenes]
    return tuple(torch.cat([to_tensor(image, 'cpu') for image in source]).to(device)
                 for source in zip(*pairs))


def detail_coefficients(lwpd, detail):
    bands, _ = lwpd.haar(detail)
    return torch.cat(bands[1:], dim=1)


@torch.no_grad()
def features(model, images, native=False):
    optical, sar = (model.encoder.encoder_level1(model.encoder.patch_embed(x)) for x in images)
    # E2's cross gates see zero context for the withheld modality, including its LL.
    _, optical_detail, _, _ = model.lwpd(optical, torch.zeros_like(sar))
    _, _, _, sar_detail = model.lwpd(torch.zeros_like(optical), sar)
    values = tuple(detail_coefficients(model.lwpd, detail) for detail in (optical_detail, sar_detail))
    if not native:
        return values
    bo, do, bs, ds = model.lwpd(optical, sar)
    base = bo + model.base_fusion(bo, bs)
    native_rms = dict(fused_base_rms=float(base.square().mean().sqrt()))
    for source, value, detail in zip(SOURCES, values, (do, ds)):
        delta = detail_coefficients(model.lwpd, detail) - value
        native_rms[f'{source}_native_delta_rms'] = float(delta.square().mean().sqrt())
    return values, native_rms


def runtime(device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python interpreter')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(device)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def register(output, split, manifest, checkpoint, metrics, parent=None):
    from run_e2_matched_no_pac import register_many

    token = sha256_file(output / 'run_metadata.json')[:16].upper()
    identifier = f'QXS-CPIS-R11-{split.upper()}-{token}'
    entry = dict(id=identifier, comparable_group=identifier, dataset='QXS-SAROPT',
                 method=f'CPIS R1.1 frozen Detail {split}', task='cpis_predictor_' + split,
                 status='diagnostic_fixed_subset', split=split, offset=0,
                 sample_count=len(read_json(manifest)['splits'][split]),
                 checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                 split_manifest=manifest.as_posix(), split_manifest_sha256=sha256_file(manifest),
                 metric_profile=PROFILE, output_policy='feature_prediction_and_residual_audit',
                 result_dir=output.as_posix(), metadata_path=(output / 'run_metadata.json').as_posix(),
                 metrics=metrics, metric_source=dict(type='json_paths', path=(output / 'summary.json').as_posix(),
                                                    paths={key: ['registry_metrics', key] for key in metrics}))
    if parent is not None:
        entry.update(parent_checkpoint=parent.as_posix(), parent_checkpoint_sha256=sha256_file(parent))
    register_many([entry])
