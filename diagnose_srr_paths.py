"""Bounded paired path interventions on six already-used validation images."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diagnose_srr_interventions import rms, save_panel
from test_OS import forward_variant, load_variant_models, sha256_file
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator


def lowpass(sar, valid):
    coordinates = torch.arange(-3, 4, device=sar.device, dtype=sar.dtype)
    kernel = torch.exp(-coordinates.square()/2)
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None]*kernel[None, :])[None, None]
    def blur(x):
        return F.conv2d(F.pad(x, (3, 3, 3, 3), mode='reflect'), kernel)
    return blur(sar*valid)/blur(valid).clamp_min(1e-6)


def gradient_rms(x):
    return float(((x[..., 1:, :]-x[..., :-1, :]).square().mean()
                  +(x[..., :, 1:]-x[..., :, :-1]).square().mean()).sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(
        'test_result/reliability_v5/srr_path_diagnosis_6'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    positions = [0, 39, 79, 119, 159, 199]
    manifest = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
    splits, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), manifest)
    pairs = [splits['val'][i] for i in positions]
    checkpoints = {
        'C0_full': Path('models/reliability_v5/frequency_minimax_seed42/epoch_005.pth'),
        'SRR_e010': Path('models/reliability_v5/srr_detail_v6_ranking_pilot/epoch_010.pth'),
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows, metadata = [], {}
    for model_name, checkpoint in checkpoints.items():
        models, variant = load_variant_models(checkpoint, device)
        metadata[model_name] = {'checkpoint': str(checkpoint), 'sha256': sha256_file(checkpoint)}
        cases = ['original', 'base_0.9', 'base_1.1', 'detail_0.9', 'detail_1.1',
                 'base_source_lowpass', 'detail_source_lowpass']
        if model_name == 'SRR_e010':
            cases += ['comp_half', 'comp_from_cross_same_norm', 'cross_from_comp_same_norm']
        with torch.no_grad():
            for op, sp in pairs:
                ou8 = cv2.cvtColor(cv2.imread(str(op)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
                su8 = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
                optical = torch.from_numpy(ou8.astype(np.float32)/255)[None, None].to(device)
                sar = torch.from_numpy(su8.astype(np.float32)/255)[None, None].to(device)
                valid = (sar>0).float()
                bo, do, _ = models['encoder'](optical)
                bs, ds, _ = models['encoder'](sar)
                bl, dl, _ = models['encoder'](lowpass(sar, valid))
                baseline = forward_variant(models, variant, optical, sar)[0]
                prior = None
                if model_name == 'SRR_e010':
                    router = models['detail_router']
                    prior = models['line_evidence'](sar, valid)['reliability']
                    # Hold evidence and route heads fixed when perturbing detail content.
                    _, maps = router(do, ds, prior, valid=valid, return_maps=True)
                    gs, ax, am = [maps[name] for name in (
                        'srr_reliability_map', 'srr_cross_attribute_map',
                        'srr_complementary_attribute_map')]
                tiles = [('Optical', ou8/255.), ('SAR', su8/255.)]
                outputs = {}
                for case in cases:
                    sb = bs
                    if case.startswith('base_') and case != 'base_source_lowpass':
                        sb = bs * float(case.split('_')[1])
                    if case == 'base_source_lowpass':
                        sb = bl
                    detail_input = dl if case == 'detail_source_lowpass' else ds
                    scale = float(case.split('_')[1]) if case.startswith('detail_') and case != 'detail_source_lowpass' else 1.
                    if model_name == 'C0_full':
                        fused_detail = models['detail_fuse'](do+scale*detail_input)
                    else:
                        dx = router.cross_candidate(do, detail_input)
                        dm = router.complementary_candidate(detail_input)
                        if case == 'comp_from_cross_same_norm':
                            dm = dx * (dm.square().mean().sqrt()/dx.square().mean().sqrt().clamp_min(1e-8))
                        if case == 'cross_from_comp_same_norm':
                            dx = dm * (dx.square().mean().sqrt()/dm.square().mean().sqrt().clamp_min(1e-8))
                        rx = gs*ax*dx*scale
                        rm = gs*am*dm*scale*(.5 if case=='comp_half' else 1.)
                        fused_detail = router.detail_transform(router.route_fusion(torch.cat((do, rx, rm), dim=1)))
                    base = models['base_fuse'](bo+sb)
                    output = models['decoder'](optical, base, fused_detail)[0].clamp(0, 1)
                    outputs[case] = output
                    if case == 'original':
                        assert float((output-baseline).abs().max()) < 1e-6
                    change = output-baseline
                    row = {'model': model_name, 'image': op.stem, 'case': case,
                           'output_change_mae': float(change.abs().mean()),
                           'output_change_gradient_rms': gradient_rms(change),
                           'optical_mae': float((output-optical).abs().mean()),
                           'sar_mae': float(((output-sar).abs()*valid).sum()/valid.sum()),
                           'sar_base_lowpass_relative_change': rms(bl-bs)/max(rms(bs),1e-12),
                           'sar_detail_lowpass_relative_change': rms(dl-ds)/max(rms(ds),1e-12)}
                    rendered = np.rint(output.squeeze().cpu().numpy()*255).clip(0,255).astype(np.float32)
                    for key, method in [('FMI','FMI'), ('Qabf','Qabf'), ('Nabf','Nabf'), ('VIF','VIFF')]:
                        row[key] = float(getattr(PaperEvaluator, method)(rendered, ou8.astype(np.float32), su8.astype(np.float32)))
                    rows.append(row)
                    if case in ('original', 'base_0.9','detail_0.9','base_source_lowpass',
                                'detail_source_lowpass','comp_from_cross_same_norm'):
                        tiles.append((case, output.squeeze().cpu().numpy()))
                # Matched +/-10% intervention approximates local path sensitivity.
                for row in rows[-len(cases):]:
                    row['base_symmetric_sensitivity'] = float((outputs['base_1.1']-outputs['base_0.9']).abs().mean())/.2
                    row['detail_symmetric_sensitivity'] = float((outputs['detail_1.1']-outputs['detail_0.9']).abs().mean())/.2
                save_panel(args.output_dir/(model_name+'_'+op.stem+'.png'), tiles)
                print(model_name+' '+op.stem+' done', flush=True)
        del models
    with (args.output_dir/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for model_name in checkpoints:
        summary[model_name] = {}
        for case in dict.fromkeys(r['case'] for r in rows if r['model']==model_name):
            selected = [r for r in rows if r['model']==model_name and r['case']==case]
            summary[model_name][case] = {key: float(np.mean([r[key] for r in selected]))
                                       for key in rows[0] if key not in ('model','case','image')}
            original = {r['image']: r for r in rows if r['model']==model_name and r['case']=='original'}
            summary[model_name][case]['metric_improved_counts'] = {
                key: sum((r[key]-original[r['image']][key])*sign > 0 for r in selected)
                for key, sign in [('FMI',1),('Qabf',1),('Nabf',-1),('VIF',1)]}
    report = {'models': metadata, 'positions': positions, 'image_ids': [p[0].stem for p in pairs],
              'detail_controls': 'SRR evidence and gates held fixed at original values',
              'caution': 'Within-checkpoint interventions only. C0 and SRR training histories differ.',
              'summary': summary}
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
