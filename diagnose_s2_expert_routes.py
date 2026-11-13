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
from run_s2_expert_pilot import ROOT, EVAL_ROOT, MANIFEST, DATA_ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dense', 'top2'), default='top2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from utils.dataset_os import manifest_pairs, OpticalSARDataset
    ck = ROOT / f'{args.mode}_seed{args.seed}' / 'final.pth'
    if not ck.is_file() or not args.manifest.is_file():
        raise FileNotFoundError('Run checkpoint or manifest is missing')
    state = torch.load(ck, map_location='cpu', weights_only=False)
    if state.get('mode') != args.mode or state.get('config', {}).get('manifest_sha256') != digest(args.manifest):
        raise ValueError('Run mode or source manifest does not match')
    model = S2ModelFactory.build(args.mode).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    fusion = model.detail_fusion
    if not isinstance(fusion, HeterogeneousExpertFusion):
        raise RuntimeError('This analysis requires heterogeneous experts')
    for p in model.parameters():
        p.requires_grad_(False)

    splits, _ = manifest_pairs(args.data_root, args.manifest)
    if not 0 < args.limit <= len(splits['val']):
        raise ValueError('Invalid validation limit')
    ds = OpticalSARDataset(splits['val'][:args.limit], patch_size=256, training=False,
                          invalid_threshold=0, min_valid_fraction=.9, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for optical, sar, valid, names in loader:
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

    target = EVAL_ROOT / f'{args.mode}_seed{args.seed}_routes_val{args.limit}'
    if target.exists():
        raise FileExistsError(f'Preserve existing routing report: {target}')
    target.mkdir(parents=True)
    with (target / 'routes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {key: sum(float(row[key]) for row in rows) / len(rows)
             for key in rows[0] if key != 'scene_id'}
    report = {
        'protocol': 'S2_expert_routing_val_only_v1',
        'mode': args.mode, 'seed': args.seed, 'sample_count': len(rows),
        'checkpoint': str(ck), 'checkpoint_sha256': digest(ck),
        'manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest),
        'scene_ids': [row['scene_id'] for row in rows],
        'mean': means,
        'limitations': [
            'Selected experts and output cosines are not structure ground truth.',
            'Top2 implementation computes all three experts and is not sparse FLOPs.',
            'Different expert responses do not prove that a given expert is optimal.',
            'No test split or manually labeled structural annotations are used.'
        ],
    }
    (target / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', target)


if __name__ == '__main__':
    main()
