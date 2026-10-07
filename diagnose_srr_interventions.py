"""Small paired inference interventions; never trains or changes checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from test_OS import load_variant_models, forward_variant, sha256_file
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator


CASES = (
    'original', 'gate_zero', 'gate_mean', 'gate_one',
    'gate_invert', 'gate_blur', 'gate_shuffle_42', 'gate_shuffle_43',
    'prior_zero', 'prior_shift', 'prior_shuffle_42', 'prior_shuffle_43',
    'cross_evidence_zero', 'cross_off', 'complementary_off', 'both_off',
    'sar_base_half', 'sar_base_zero', 'sar_base_optical',
)


def scalar_mean(value, mask):
    selected = value[mask]
    return float(selected.mean()) if selected.numel() else None


def rms(value):
    return float(value.square().mean().sqrt())


def save_panel(path, tiles):
    rows = []
    for start in range(0, len(tiles), 4):
        cells = []
        for title, data in tiles[start:start + 4]:
            u8 = np.clip(np.rint(data * 255), 0, 255).astype(np.uint8)
            cell = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            cell = cv2.copyMakeBorder(cell, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(cell, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            cells.append(cell)
        while len(cells) < 4:
            cells.append(np.zeros_like(cells[0]))
        rows.append(np.concatenate(cells, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'models/reliability_v5/srr_detail_v6_ranking_pilot/epoch_010.pth'))
    parser.add_argument('--output-dir', type=Path, default=Path(
        'test_result/reliability_v5/srr_v6_e010_interventions_6'))
    args = parser.parse_args()
    # Fixed six evenly spaced positions within the previously evaluated val200.
    positions = [0, 39, 79, 119, 159, 199]
    manifest = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
    pairs, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), manifest)
    selected = [pairs['val'][i] for i in positions]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models, variant = load_variant_models(args.checkpoint, device)
    router = models['detail_router']
    rows, feature_rows = [], []
    prior_metadata = Path('test_result/reliability_v5/srr_detail_v6_ranking_e010_validation/run_metadata.json')
    recorded = json.loads(prior_metadata.read_text(encoding='utf-8'))
    checkpoint_hash = sha256_file(args.checkpoint)
    assert recorded['checkpoint_sha256'] == checkpoint_hash
    recorded_metrics = {}
    with prior_metadata.with_name('metrics.csv').open(encoding='utf-8-sig') as handle:
        recorded_metrics = {row['image']: row for row in csv.DictReader(handle)}

    with torch.no_grad():
        for optical_path, sar_path in selected:
            image_id = optical_path.stem
            optical_u8 = cv2.cvtColor(cv2.imread(str(optical_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            optical = torch.from_numpy(optical_u8.astype(np.float32)/255.0)[None, None].to(device)
            sar = torch.from_numpy(sar_u8.astype(np.float32)/255.0)[None, None].to(device)
            valid = (sar > 0).float()
            mask = valid.bool()
            bo, do, _ = models['encoder'](optical)
            bs, ds, _ = models['encoder'](sar)
            prior = models['line_evidence'](sar, valid)['reliability']
            fused_base = models['base_fuse'](bo + bs)
            original, reference_maps = forward_variant(models, variant, optical, sar, return_maps=True)
            original = original.clamp(0, 1)
            baseline_gate = reference_maps['srr_reliability_map']
            cross = reference_maps['srr_cross_evidence_map']
            positive = (prior >= 0.5) & mask
            negative = (prior <= 0.05) & (cross <= torch.quantile(cross[mask], .25)) & mask
            # Masks stay fixed to the unmodified forward for every intervention.
            mean_gate = scalar_mean(baseline_gate, mask)
            dx = router.cross_candidate(do, ds)
            dm = router.complementary_candidate(ds)
            ax = reference_maps['srr_cross_attribute_map']
            am = reference_maps['srr_complementary_attribute_map']
            weights = router.route_fusion.weight
            c = do.shape[1]
            fx = F.conv2d(baseline_gate * ax * dx, weights[:, c:2*c])
            fm = F.conv2d(baseline_gate * am * dm, weights[:, 2*c:])
            feature_rows.append({
                'image': image_id,
                'candidate_cosine': float(F.cosine_similarity(dx.flatten(), dm.flatten(), dim=0)),
                'candidate_relative_difference': rms(dx-dm)/max(rms(dx), 1e-12),
                'cross_post_projection_rms': rms(fx),
                'complementary_post_projection_rms': rms(fm),
                'optical_post_projection_rms': rms(F.conv2d(do, weights[:, :c])),
                'positive_fraction': float(positive.sum()/mask.sum()),
                'negative_fraction': float(negative.sum()/mask.sum()),
            })
            tiles = [('Optical', optical_u8/255.), ('SAR', sar_u8/255.),
                     ('prior', prior.squeeze().cpu().numpy()),
                     ('gate original', baseline_gate.squeeze().cpu().numpy())]
            for case in CASES:
                changed_prior = prior
                if case == 'prior_zero':
                    changed_prior = torch.zeros_like(prior)
                elif case == 'prior_shift':
                    changed_prior = torch.roll(prior, (prior.shape[-2]//2, prior.shape[-1]//2), (-2, -1))
                elif case.startswith('prior_shuffle'):
                    ids = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
                    generator = torch.Generator().manual_seed(int(case.rsplit('_', 1)[1]))
                    order = torch.randperm(ids.numel(), generator=generator).to(device)
                    changed_prior = prior.clone().flatten()
                    changed_prior[ids] = prior.flatten()[ids[order]]
                    changed_prior = changed_prior.view_as(prior)
                force = {'gate_zero': 0., 'gate_one': 1., 'gate_mean': mean_gate}.get(case)
                if case == 'gate_invert':
                    force = 1.0 - baseline_gate
                elif case == 'gate_blur':
                    force = F.avg_pool2d(baseline_gate, 15, stride=1, padding=7)
                elif case.startswith('gate_shuffle'):
                    ids = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
                    generator = torch.Generator().manual_seed(int(case.rsplit('_', 1)[1]))
                    order = torch.randperm(ids.numel(), generator=generator).to(device)
                    force = baseline_gate.clone().flatten()
                    force[ids] = baseline_gate.flatten()[ids[order]]
                    force = force.view_as(baseline_gate)
                hooks = []
                captured = {}

                def route_input_hook(module, inputs):
                    data = inputs[0].clone()
                    if case in ('cross_off', 'both_off'):
                        data[:, c:2*c] = 0
                    if case in ('complementary_off', 'both_off'):
                        data[:, 2*c:] = 0
                    captured['route_input'] = data
                    return (data,)

                hooks.append(router.route_fusion.register_forward_pre_hook(route_input_hook))
                if case == 'cross_evidence_zero':
                    hooks.append(router.cross_evidence.register_forward_hook(
                        lambda module, inputs, output: tuple(torch.zeros_like(x) for x in output)))
                try:
                    df, diagnostics = router(do, ds, changed_prior, valid=valid,
                                             return_maps=True, force_reliability=force)
                finally:
                    for hook in hooks:
                        hook.remove()
                base = fused_base
                if case == 'sar_base_half':
                    base = models['base_fuse'](bo + .5*bs)
                elif case == 'sar_base_zero':
                    base = models['base_fuse'](bo)
                elif case == 'sar_base_optical':
                    base = models['base_fuse'](bo + bo)
                output = models['decoder'](optical, base, df)[0].clamp(0, 1)
                diff = (output-original).abs()
                if case == 'original':
                    assert float(diff.max()) < 1e-6, 'Intervention path must reproduce production forward'
                if case in ('both_off', 'gate_zero'):
                    assert float(captured['route_input'][:, c:].abs().max()) == 0
                gate = diagnostics['srr_reliability_map']
                row = {'image': image_id, 'case': case,
                       'output_mae_change': float(diff.mean()),
                       'output_max_change': float(diff.max()),
                       'positive_output_change': scalar_mean(diff, positive),
                       'negative_output_change': scalar_mean(diff, negative),
                       'optical_mae': float((output-optical).abs().mean()),
                       'sar_mae': scalar_mean((output-sar).abs(), mask),
                       'gate_change_mae_valid': scalar_mean((gate-baseline_gate).abs(), mask),
                       'gate_mean_valid': scalar_mean(gate, mask),
                       'gate_std_valid': float(gate[mask].std(unbiased=False)),
                       'positive_gate': scalar_mean(gate, positive),
                       'negative_gate': scalar_mean(gate, negative)}
                rendered = np.rint(output.squeeze().cpu().numpy()*255).clip(0,255)
                for key, method in [('FMI', 'FMI'), ('Qabf', 'Qabf'), ('Nabf', 'Nabf'), ('VIF', 'VIFF')]:
                    row[key] = float(getattr(PaperEvaluator, method)(
                        rendered.astype(np.float32), optical_u8.astype(np.float32),
                        sar_u8.astype(np.float32)))
                    if case == 'original':
                        assert abs(row[key]-float(recorded_metrics[image_id][key])) < 2e-6, (
                            image_id, key, row[key], recorded_metrics[image_id][key])
                rows.append(row)
                if case in ('original', 'gate_zero', 'gate_mean', 'gate_one', 'gate_invert',
                            'gate_blur', 'gate_shuffle_42', 'prior_shift',
                            'cross_off', 'complementary_off', 'sar_base_zero'):
                    tiles.append((case, output.squeeze().cpu().numpy()))
            save_panel(args.output_dir / (image_id+'_panel.png'), tiles)
            print('finished image '+image_id, flush=True)

    for filename, data in [('per_image.csv', rows), ('branch_features.csv', feature_rows)]:
        with (args.output_dir / filename).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {}
    for case in CASES:
        subset = [r for r in rows if r['case'] == case]
        summary[case] = {key: float(np.mean([r[key] for r in subset if r[key] is not None]))
                         for key in rows[0] if key not in ('image', 'case')}
    report = {'checkpoint_sha256': checkpoint_hash, 'selection': positions,
              'image_ids': [p[0].stem for p in selected], 'n_images': len(selected),
              'note': 'Small paired diagnostic only; original anchors frozen across interventions. Base/branch ablations may be out of distribution.',
              'summary': summary, 'branch_features': feature_rows}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
