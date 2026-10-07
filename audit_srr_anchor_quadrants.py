"""Visual audit of the four R0/Ex evidence quadrants on a small val subset."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from test_OS import forward_variant, load_variant_models
from utils.dataset_os import manifest_pairs


V6 = Path('models/reliability_v5/srr_detail_v6_ranking_pilot/epoch_010.pth')
C0 = Path('models/reliability_v5/frequency_minimax_baseline_pilot/epoch_005.pth')
OUTPUT = Path('test_result/reliability_v5/srr_anchor_quadrant_audit_24_center_checked')
POSITIONS = np.linspace(0, 199, 24, dtype=int).tolist()
PATCH = 64
CONTEXT = 17
SAMPLES_PER_QUADRANT = 6

QUADRANTS = {
    'Rhigh_Ehigh': 'R0 high / Ex high',
    'Rhigh_Elow': 'R0 high / Ex low',
    'Rlow_Ehigh': 'R0 low / Ex high',
    'Rlow_Elow': 'R0 low / Ex low',
}


def crop(array, y, x, size=PATCH):
    half = size // 2
    return array[y-half:y+half, x-half:x+half]


def color_heatmap(values, maximum=1.0):
    scaled = np.clip(values / max(maximum, 1e-8), 0, 1)
    return cv2.applyColorMap(np.rint(scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def gray_bgr(values):
    u8 = np.clip(np.rint(values * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def label_cell(cell, text):
    cell = cv2.copyMakeBorder(cell, 22, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(cell, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def best_center(mask, score):
    support = F.avg_pool2d(mask.float(), CONTEXT, stride=1, padding=CONTEXT//2)
    # The context window should contain the quadrant, and the selected center
    # must itself satisfy the quadrant definition.
    combined = support * score * mask.float()
    margin = PATCH // 2
    combined[..., :margin, :] = 0
    combined[..., -margin:, :] = 0
    combined[..., :, :margin] = 0
    combined[..., :, -margin:] = 0
    index = int(combined.flatten().argmax())
    width = combined.shape[-1]
    y = (index // width) % combined.shape[-2]
    x = index % width
    return y, x, float(support[0, 0, y, x]), float(combined[0, 0, y, x])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v6, _ = load_variant_models(V6, device, 'srr_detail')
    c0, _ = load_variant_models(C0, device, 'frequency_minimax_baseline')
    manifest = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
    splits, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), manifest)
    selected = [splits['val'][position] for position in POSITIONS]
    candidates = {name: [] for name in QUADRANTS}
    all_cross = []

    with torch.no_grad():
        for optical_path, sar_path in selected:
            optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
            optical_u8 = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            optical = torch.from_numpy(optical_u8.astype(np.float32)/255)[None, None].to(device)
            sar = torch.from_numpy(sar_u8.astype(np.float32)/255)[None, None].to(device)
            valid = (sar > 0).float()
            srr, diagnostics = forward_variant(v6, 'srr_detail', optical, sar, return_maps=True)
            c0_output, _ = forward_variant(c0, 'frequency_minimax_baseline', optical, sar)
            prior = diagnostics['srr_structure_prior_map']
            cross = diagnostics['srr_cross_evidence_map']
            gate = diagnostics['srr_reliability_map']
            valid_bool = valid > .5
            cross_values = cross[valid_bool]
            low = torch.quantile(cross_values.float(), .25).to(cross.dtype)
            high = torch.quantile(cross_values.float(), .75).to(cross.dtype)
            masks = {
                'Rhigh_Ehigh': (prior >= .5) & (cross >= high) & valid_bool,
                'Rhigh_Elow': (prior >= .5) & (cross <= low) & valid_bool,
                'Rlow_Ehigh': (prior <= .05) & (cross >= high) & valid_bool,
                'Rlow_Elow': (prior <= .05) & (cross <= low) & valid_bool,
            }
            confidence = {
                'Rhigh_Ehigh': prior * cross / high.clamp_min(1e-6),
                'Rhigh_Elow': prior * (1-cross/low.clamp_min(1e-6)).clamp_min(0),
                'Rlow_Ehigh': (1-prior) * cross / high.clamp_min(1e-6),
                'Rlow_Elow': (1-prior) * (1-cross/low.clamp_min(1e-6)).clamp_min(0),
            }
            arrays = {
                'optical': optical.squeeze().cpu().numpy(),
                'sar': sar.squeeze().cpu().numpy(),
                'prior': prior.squeeze().cpu().numpy(),
                'cross': cross.squeeze().cpu().numpy(),
                'gate': gate.squeeze().cpu().numpy(),
                'c0': c0_output.squeeze().clamp(0,1).cpu().numpy(),
                'srr': srr.squeeze().clamp(0,1).cpu().numpy(),
            }
            all_cross.append(arrays['cross'][valid_bool.squeeze().cpu().numpy()])
            for name in QUADRANTS:
                y, x, support, combined = best_center(masks[name], confidence[name])
                if support <= 0:
                    continue
                record = {'quadrant': name, 'image': optical_path.stem,
                          'x': x, 'y': y, 'support_fraction_17': support,
                          'selection_score': combined,
                          'Ex_q25_image': float(low), 'Ex_q75_image': float(high)}
                for key in ('prior', 'cross', 'gate'):
                    patch = crop(arrays[key], y, x)
                    record[key+'_mean'] = float(patch.mean())
                    record[key+'_center'] = float(arrays[key][y, x])
                record['c0_srr_patch_mae'] = float(np.mean(np.abs(
                    crop(arrays['c0'], y, x)-crop(arrays['srr'], y, x))))
                record['optical_sar_patch_mae'] = float(np.mean(np.abs(
                    crop(arrays['optical'], y, x)-crop(arrays['sar'], y, x))))
                record['arrays'] = arrays
                candidates[name].append(record)
            print('scanned ' + optical_path.stem, flush=True)

    ex_display_max = float(np.quantile(np.concatenate(all_cross), .99))
    rows = []
    for name, title in QUADRANTS.items():
        (OUTPUT/name).mkdir(parents=True, exist_ok=True)
        chosen = sorted(candidates[name], key=lambda item: item['selection_score'], reverse=True)[:SAMPLES_PER_QUADRANT]
        montage_rows = []
        for record in chosen:
            arrays = record.pop('arrays')
            y, x = record['y'], record['x']
            cells = [
                label_cell(gray_bgr(crop(arrays['optical'], y, x)), 'Optical'),
                label_cell(gray_bgr(crop(arrays['sar'], y, x)), 'SAR'),
                label_cell(color_heatmap(crop(arrays['prior'], y, x)), 'R0'),
                label_cell(color_heatmap(crop(arrays['cross'], y, x), ex_display_max), 'Ex'),
                label_cell(color_heatmap(crop(arrays['gate'], y, x)), 'g'),
                label_cell(gray_bgr(crop(arrays['c0'], y, x)), 'C0'),
                label_cell(gray_bgr(crop(arrays['srr'], y, x)), 'SRR'),
            ]
            line = np.concatenate(cells, axis=1)
            line = cv2.copyMakeBorder(line, 20, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(line, f"{record['image']} @ ({x},{y}) support={record['support_fraction_17']:.2f}",
                        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
            montage_rows.append(line)
            cv2.imwrite(
                str(OUTPUT/name/(record['image']+f"_{x}_{y}.png")),
                cv2.resize(line, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
            )
            rows.append(record)
        cv2.imwrite(str(OUTPUT/(name+'.png')), np.concatenate(montage_rows, axis=0))

    with (OUTPUT/'samples.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {'checkpoint': str(V6), 'c0_checkpoint': str(C0), 'positions': POSITIONS,
              'R0_high': .5, 'R0_low': .05,
              'Ex_low': 'per-image valid q25', 'Ex_high': 'per-image valid q75',
              'patch': PATCH, 'support_window': CONTEXT,
              'Ex_display_q99': ex_display_max,
              'note': 'Candidate crops are selected for visual semantic audit; selection is not a prevalence estimate.'}
    (OUTPUT/'audit_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({name: len([r for r in rows if r['quadrant']==name]) for name in QUADRANTS}, indent=2))


if __name__ == '__main__':
    main()
