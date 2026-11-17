"""Post-training S2 expert diagnostics on a fixed validation subset (no test data).

This measures actual region selection, router entropy and pairwise centered
expert-output cosine. These are mechanism diagnostics, NOT fusion quality metrics.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s2_expert_models import HeterogeneousExpertFusion, S2ModelFactory
from run_s2_expert_pilot import EVAL_ROOT, MANIFEST, DATA_ROOT, digest, matched_group, register_result, validated_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dense', 'top2'), default='top2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', '--val-limit', dest='val_limit', type=int, default=64)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from utils.dataset_os import manifest_pairs, OpticalSARDataset
    ck, manifest, config, scene_ids = validated_run(args, args.mode)
    target = EVAL_ROOT / f'{args.mode}_seed{args.seed}_routes_val{args.val_limit}'
    if target.exists():
        raise FileExistsError(f'Preserve existing routing report: {target}')
    state = torch.load(ck, map_location='cpu', weights_only=False)
    if state.get('mode') != args.mode or state.get('config') != config:
        raise ValueError('Checkpoint configuration does not match the pilot run')
    model = S2ModelFactory.build(args.mode).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    fusion = model.detail_fusion
    if not isinstance(fusion, HeterogeneousExpertFusion):
        raise RuntimeError('This analysis requires heterogeneous experts')
    for p in model.parameters():
        p.requires_grad_(False)

    splits, _ = manifest_pairs(args.data_root, manifest)
    ds = OpticalSARDataset(splits['val'], patch_size=256, training=False,
                          invalid_threshold=0, min_valid_fraction=.9, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for optical, sar, _, names in loader:
            optical = optical.to(args.device)
            sar = sar.to(args.device)
            _ = model(optical, sar)
            weights = fusion.last_route
            cosines = fusion.last_expert_cosine
            if weights is None or cosines is None:
                raise RuntimeError('Expert diagnostics missing')
            weights = weights.float()
            active = weights > 0.
            winner = weights.argmax(dim=1)
            row = {'scene_id': str(names[0]),
                   'route_entropy': float((-(weights * weights.clamp_min(1e-12).log()).sum(1)).mean()),
                   'cos_opt_sar': cosines[0],
                   'cos_opt_col': cosines[1],
                   'cos_sar_col': cosines[2]}
            for i, name in enumerate(('optical', 'sar', 'cooperative')):
                row[f'{name}_selected_fraction'] = float(active[:, i].float().mean())
                row[f'{name}_top1_fraction'] = float((winner == i).float().mean())
                row[f'{name}_weight_mean'] = float(weights[:, i].mean())
            if not all(math.isfinite(float(v)) for k, v in row.items() if k != 'scene_id'):
                raise RuntimeError('Nonfinite routing diagnostic')
            rows.append(row)
            print(f'route {args.mode}: {len(rows)}/{len(loader)}', flush=True)

    if [row['scene_id'] for row in rows] != scene_ids:
        raise ValueError('Routing scenes differ from the recorded pilot split')
    target.mkdir(parents=True, exist_ok=False)
    with (target / 'routes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {key: sum(float(row[key]) for row in rows) / len(rows)
             for key in rows[0] if key != 'scene_id'}
    report = {
        'protocol': 'S2_expert_routing_val_only_v2',
        'mode': args.mode, 'seed': args.seed, 'sample_count': len(rows),
        'checkpoint': str(ck), 'checkpoint_sha256': digest(ck),
        'manifest': manifest.as_posix(), 'manifest_sha256': digest(manifest),
        'scene_ids': scene_ids, 'device': args.device,
        'mean': means,
        'limitations': [
            'Selected experts and output cosines are not structure ground truth.',
            'Top2 implementation computes all three experts and is not sparse FLOPs.',
            'Different expert responses do not prove that a given expert is optimal.',
            'No test split or manually labeled structural annotations are used.'
        ],
    }
    summary = target / 'summary.json'
    summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    register_result(
        args.mode, ck, manifest, target, summary, 'val', len(rows), 'routing_diagnostics', 'float_features',
        group=f'{matched_group(config)}-ROUTES-VAL{len(rows)}-{args.mode.upper()}',
        task='engineering', status='development_validation', metrics=means,
        metric_source=dict(type='json_paths', path=summary.as_posix(), paths={key: ['mean', key] for key in means}),
    )
    print('Saved:', target)


if __name__ == '__main__':
    main()
