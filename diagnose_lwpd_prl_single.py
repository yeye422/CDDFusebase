"""Single-validation-image lifting intervention; never trains or scans a split."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from direct_fusion import DirectFusionCDDFuse
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.Evaluator import PaperEvaluator


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edge_stats(image, optical, sar):
    gf, af = PaperEvaluator.Qabf_getArray(image.astype(np.float64))
    go, ao = PaperEvaluator.Qabf_getArray(optical.astype(np.float64))
    gs, ass = PaperEvaluator.Qabf_getArray(sar.astype(np.float64))
    target = np.maximum(go, gs)
    excess = np.maximum(gf - target, 0)
    # Direction is pi-periodic; weight by stronger-source gradient.
    source_angle = np.where(go >= gs, ao, ass)
    angle = np.abs(af - source_angle)
    angle = np.minimum(angle, np.pi - angle)
    return dict(overshoot_fraction=float(np.mean(gf > target)),
                overshoot_mean=float(excess.mean()),
                overshoot_p99=float(np.quantile(excess, .99)),
                weighted_angle_error=float(np.sum(angle * target) / max(target.sum(), 1e-12)),
                near_threshold_fraction=float(np.mean(np.abs(gf - target) < 1.0))), excess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene-id', default='4')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    manifest = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
    splits, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), manifest)
    pair = next((pair for pair in splits['val'] if pair[0].stem == args.scene_id), None)
    if pair is None:
        raise ValueError('The requested scene must belong to val; test is prohibited.')
    optical_bgr = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
    sar = cv2.imread(str(pair[1]), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar is None:
        raise ValueError('Cannot load scene')
    ycc = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
    optical = ycc[:, :, 0]
    assert optical.shape == sar.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    torch.set_num_threads(4)
    x = torch.from_numpy(optical.astype(np.float32) / 255)[None, None].to(device)
    s = torch.from_numpy(sar.astype(np.float32) / 255)[None, None].to(device)
    out.mkdir(parents=True)
    e2_path = Path('models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth')
    prl_path = Path('models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth')
    checkpoints = {k: torch.load(v, map_location='cpu', weights_only=False)
                   for k, v in [('e2', e2_path), ('prl', prl_path)]}
    modes = [('e2_fp32', 'e2', False, False),
             ('prl_fp32', 'prl', False, False),
             ('prl_haar_fp32', 'prl', True, False)]
    if device == 'cuda':
        modes += [('prl_amp', 'prl', False, True), ('prl_haar_amp', 'prl', True, True)]
    images, raw_outputs, records = {}, {}, []
    for name, source, zero_delta, amp in modes:
        folder = out / name
        folder.mkdir()
        ck = checkpoints[source]
        model = DirectFusionCDDFuse(ck['variant']).to(device).eval()
        model.load_state_dict(ck['model'], strict=True)
        with torch.no_grad():
            if zero_delta:
                for key, value in model.lwpd.haar.named_parameters():
                    if key.endswith('_delta'):
                        value.zero_()
            with torch.autocast(device_type=device, enabled=amp):
                fused, features = model(x, s, return_features=True)
            raw = fused.float().squeeze().cpu().numpy()
            assert np.isfinite(raw).all(), name
            # Test reconstruction on the actual encoded optical features in FP32.
            shared = model.encoder.encoder_level1(model.encoder.patch_embed(x))
            bands, size = model.lwpd.haar(shared)
            reconstruction = model.lwpd.haar.inverse(bands, size)
            pr_error = float((shared - reconstruction).abs().max())
            diagnostic = features['lwpd']
            gates = {k: float(diagnostic[k].float().mean()) for k in
                     ['optical_gates', 'sar_gates', 'sar_coefficients', 'wavelet_deviation']}
            feature_rms = {k + '_rms': float(v.float().square().mean().sqrt())
                           for k, v in features.items() if torch.is_tensor(v)}
        clamped = np.clip(raw, 0, 1)
        u8 = np.clip(np.rint(clamped * 255), 0, 255).astype(np.uint8)
        values = PaperEvaluator.metric_vector(u8, optical, sar)
        metric = dict(zip(PaperEvaluator.METRIC_NAMES, map(float, values)))
        write_csv(folder / 'metrics.csv', [dict(image=args.scene_id, **metric),
                  dict(image='mean', **metric),
                  dict(image='std', **{k: 0.0 for k in metric})])
        stats = dict(mode=name, **metric, **gates, **feature_rms,
                     reconstruction_max_error=pr_error, raw_min=float(raw.min()),
                     raw_max=float(raw.max()), clip_fraction=float(np.mean((raw < 0) | (raw > 1))))
        for stage, image in [('raw', raw * 255), ('clamp', clamped * 255), ('uint8', u8)]:
            edge, excess = edge_stats(image, optical, sar)
            stats.update({stage + '_' + k: v for k, v in edge.items()})
            stats[stage + '_Nabf'] = PaperEvaluator.Nabf(image, optical, sar)
            if stage == 'uint8':
                np.save(folder / 'gradient_excess.npy', excess)
        cv2.imwrite(str(folder / 'gray.png'), u8)
        rgb = ycc.copy()
        rgb[:, :, 0] = u8
        rgb = cv2.cvtColor(rgb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(str(folder / 'rgb.png'), rgb)
        cv2.imwrite(str(folder / 'preview.png'), np.concatenate([optical_bgr, cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR), rgb], axis=1))
        np.save(folder / 'raw_output.npy', raw)
        checkpoint_path = e2_path if source == 'e2' else prl_path
        meta = dict(scene_id=args.scene_id, selection='explicit single val scene; default is first scene 4; not selected by result',
                    split='val', offset=next(i for i, item in enumerate(splits['val']) if item[0].stem == args.scene_id), evaluated_pairs=1, variant=ck['variant'],
                    checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=sha256_file(checkpoint_path),
                    split_manifest=str(manifest.resolve()), split_manifest_sha256=sha256_file(manifest),
                    input_paths=[str(v.resolve()) for v in pair], input_sha256=[sha256_file(v) for v in pair],
                    metric_profile='paper', output_range_policy='clamp', precision='amp_fp16' if amp else 'fp32',
                    intervention='zero_all_lifting_deltas' if zero_delta else 'none', device=device,
                    torch_version=torch.__version__, diagnostics=stats,
                    note='Single-scene diagnosis only. Float NAB/F and overshoot are diagnostic, not formal paper metrics.')
        (folder / 'run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        records.append(stats)
        images[name], raw_outputs[name] = u8, raw
        print(name, json.dumps({k: stats[k] for k in ['Nabf', 'SSIM', 'MI', 'uint8_overshoot_fraction', 'clip_fraction', 'reconstruction_max_error']}), flush=True)
        del model, features, shared, bands, reconstruction
    write_csv(out / 'diagnostics.csv', records)
    differences = []
    for left, right in [('prl_fp32', 'prl_haar_fp32'), ('e2_fp32', 'prl_fp32'),
                        ('prl_fp32', 'prl_amp'), ('prl_haar_fp32', 'prl_haar_amp')]:
        if right not in images:
            continue
        diff = np.abs(raw_outputs[left] - raw_outputs[right])
        differences.append(dict(left=left, right=right, raw_mae=float(diff.mean()),
                                raw_max=float(diff.max()), changed_u8_fraction=float(np.mean(images[left] != images[right]))))
        cv2.imwrite(str(out / f'diff_{left}_vs_{right}_x20.png'), np.clip(diff * 255 * 20, 0, 255).astype(np.uint8))
    write_csv(out / 'output_differences.csv', differences)
    tiles = []
    for label, image in [('Optical', optical), ('SAR', sar), *images.items()]:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(tile, label, (5,19), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 1)
        tiles.append(tile)
    cv2.imwrite(str(out / 'comparison.png'), np.concatenate(tiles, axis=1))
    print('Saved single-scene diagnosis:', out.resolve())


if __name__ == '__main__':
    main()
