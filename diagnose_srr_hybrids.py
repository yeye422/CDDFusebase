"""Hybrid C0/SRR path diagnosis on six fixed validation pairs."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from test_OS import load_variant_models
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator


POSITIONS = [0, 39, 79, 119, 159, 199]
C0_PATH = Path('models/reliability_v5/frequency_minimax_baseline_pilot/epoch_005.pth')
V6_PATH = Path('models/reliability_v5/srr_detail_v6_ranking_pilot/epoch_010.pth')
OUTPUT = Path('test_result/reliability_v5/srr_v6_e010_hybrid_diagnosis_6_final')


def rms(value):
    return float(value.float().square().mean().sqrt())


def relative_state_distance(left, right):
    numerator = 0.0
    denominator = 0.0
    for key, value in left.items():
        other = right[key]
        numerator += float((value.float() - other.float()).square().sum())
        denominator += float(other.float().square().sum())
    return numerator ** 0.5 / max(denominator ** 0.5, 1e-12)


def decode(models, optical, fused_base, fused_detail):
    decoder = models['decoder']
    output = decoder(optical, fused_base, fused_detail)[0].clamp(0, 1)
    weight = decoder.reduce_channel.weight
    channels = fused_base.shape[1]
    base_projected = F.conv2d(fused_base, weight[:, :channels])
    detail_projected = F.conv2d(fused_detail, weight[:, channels:])
    return output, rms(base_projected), rms(detail_projected)


def metrics(output, optical_u8, sar_u8):
    rendered = np.rint(output.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.float32)
    optical = optical_u8.astype(np.float32)
    sar = sar_u8.astype(np.float32)
    return {
        'FMI': float(PaperEvaluator.FMI(rendered, optical, sar)),
        'Qabf': float(PaperEvaluator.Qabf(rendered, optical, sar)),
        'Nabf': float(PaperEvaluator.Nabf(rendered, optical, sar)),
        'VIF': float(PaperEvaluator.VIFF(rendered, optical, sar)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c0, _ = load_variant_models(C0_PATH, device, 'frequency_minimax_baseline')
    v6, _ = load_variant_models(V6_PATH, device, 'srr_detail')
    manifest = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
    splits, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), manifest)
    selected = [splits['val'][position] for position in POSITIONS]
    rows = []

    with torch.no_grad():
        for optical_path, sar_path in selected:
            optical_u8 = cv2.cvtColor(cv2.imread(str(optical_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            optical = torch.from_numpy(optical_u8.astype(np.float32) / 255)[None, None].to(device)
            sar = torch.from_numpy(sar_u8.astype(np.float32) / 255)[None, None].to(device)
            valid = (sar > 0).float()
            bo, do, _ = v6['encoder'](optical)
            bs, ds, _ = v6['encoder'](sar)
            prior = v6['line_evidence'](sar, valid)['reliability']
            detail_v6, diagnostics = v6['detail_router'](
                do, ds, prior, valid=valid, return_maps=True)
            detail_c0 = c0['detail_fuse'](do + ds)
            base_c0 = c0['base_fuse'](bo + bs)
            base_v6 = v6['base_fuse'](bo + bs)
            combinations = {
                'c0_base+c0_detail': (base_c0, detail_c0),
                'c0_base+v6_detail': (base_c0, detail_v6),
                'v6_base+c0_detail': (base_v6, detail_c0),
                'v6_base+v6_detail': (base_v6, detail_v6),
            }
            outputs = {}
            for name, (base, detail) in combinations.items():
                output, base_rms, detail_rms = decode(v6, optical, base, detail)
                outputs[name] = output
                row = {'image': optical_path.stem, 'case': name,
                       'base_feature_rms': rms(base), 'detail_feature_rms': rms(detail),
                       'decoder_base_projection_rms': base_rms,
                       'decoder_detail_projection_rms': detail_rms,
                       'optical_mae': float((output - optical).abs().mean()),
                       'sar_mae': float(((output-sar).abs()*valid).sum()/valid.sum().clamp_min(1))}
                row.update(metrics(output, optical_u8, sar_u8))
                rows.append(row)
            reference_c0 = outputs['c0_base+c0_detail']
            reference_v6 = outputs['v6_base+v6_detail']
            for row in rows[-4:]:
                output = outputs[row['case']]
                row['mae_from_c0'] = float((output-reference_c0).abs().mean())
                row['mae_from_v6'] = float((output-reference_v6).abs().mean())
            print('finished image ' + optical_path.stem, flush=True)

    with (OUTPUT/'per_image.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for case in sorted({row['case'] for row in rows}):
        subset = [row for row in rows if row['case'] == case]
        summary[case] = {key: float(np.mean([row[key] for row in subset]))
                         for key in rows[0] if key not in ('image', 'case')}
    checkpoint_c0 = torch.load(C0_PATH, map_location='cpu')
    checkpoint_v6 = torch.load(V6_PATH, map_location='cpu')
    distances = {
        'base_fuse_v6_vs_c0': relative_state_distance(
            checkpoint_v6['BaseFuseLayer'], checkpoint_c0['BaseFuseLayer']),
        'v6_detail_transform_vs_c0_detail_fuse': relative_state_distance(
            {key[len('detail_transform.'):]: value
             for key, value in checkpoint_v6['SRRDetailFuseLayer'].items()
             if key.startswith('detail_transform.')},
            checkpoint_c0['DetailFuseLayer']),
    }
    report = {'positions': POSITIONS, 'image_ids': [p[0].stem for p in selected],
              'summary': summary, 'relative_state_distances': distances,
              'note': 'Six-image paired hybrid diagnostic. C0 and V6 have different training histories; hybrids are attribution interventions, not trained models.'}
    (OUTPUT/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
