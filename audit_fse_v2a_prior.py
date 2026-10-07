"""Independent audit of FSE-v2a prior quality; no Router or fusion model involved."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fixed_line_evidence import FixedLineEvidence
from utils.dataset_os import manifest_pairs


CALIBRATION = Path('models/reliability_v5/fse_v2a_calibration.json')
OUTPUT = Path('test_result/reliability_v5/fse_v2a_prior_audit_24')
MANIFEST = Path('../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json')
POSITIONS = np.linspace(0, 199, 24, dtype=int).tolist()
STABILITY_POSITIONS = POSITIONS[::2]
THRESHOLDS = (0.3, 0.5, 0.7)
RANDOM_HIGH_SAMPLES = 200
RANDOM_SEED = 20261007
PATCH = 40


def prior_forward(module, sar_u8, device):
    sar = torch.from_numpy(sar_u8.astype(np.float32)/255)[None, None].to(device)
    valid = (sar > 0).float()
    with torch.no_grad():
        prior = module(sar, valid)['reliability']
    return prior.squeeze().cpu().numpy(), valid.squeeze().cpu().numpy().astype(bool)


def component_statistics(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    total = int(areas.sum())
    elongations = []
    component_meta = {}
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        area = len(xs)
        if area >= 2:
            coords = np.stack((xs, ys), axis=1).astype(np.float64)
            covariance = np.cov(coords, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance)
            elongation = float(np.sqrt((eigenvalues[-1]+0.25)/(eigenvalues[0]+0.25)))
        else:
            elongation = 1.0
        elongations.append(elongation)
        component_meta[label] = {'area': area, 'elongation': elongation}
    return labels, component_meta, {
        'component_count': int(len(areas)),
        'high_pixel_count': total,
        'mean_component_area': float(areas.mean()) if len(areas) else 0.0,
        'median_component_area': float(np.median(areas)) if len(areas) else 0.0,
        'isolated_component_fraction': float(np.mean(areas == 1)) if len(areas) else 0.0,
        'small_component_fraction_le5': float(np.mean(areas <= 5)) if len(areas) else 0.0,
        'small_pixel_fraction_le5': float(areas[areas <= 5].sum()/max(total, 1)),
        'largest_component_fraction': float(areas.max()/max(total, 1)) if len(areas) else 0.0,
        'mean_component_elongation': float(np.mean(elongations)) if elongations else 0.0,
        'elongated_component_fraction_ge3': float(np.mean(np.asarray(elongations) >= 3)) if elongations else 0.0,
    }


def structure_tensor_stats(sar_u8, valid, high):
    sar = sar_u8.astype(np.float32)/255
    smooth = cv2.GaussianBlur(sar, (0,0), .8)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx*gx, (0,0), 1.2)
    jyy = cv2.GaussianBlur(gy*gy, (0,0), 1.2)
    jxy = cv2.GaussianBlur(gx*gy, (0,0), 1.2)
    coherence = np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-8)
    magnitude = np.sqrt(gx*gx+gy*gy)
    return {
        'orientation_coherence_high': float(coherence[high].mean()) if high.any() else 0.0,
        'orientation_coherence_all': float(coherence[valid].mean()),
        'gradient_magnitude_high': float(magnitude[high].mean()) if high.any() else 0.0,
        'gradient_magnitude_all': float(magnitude[valid].mean()),
    }


def perturbations(sar_u8, seed):
    sar = sar_u8.astype(np.float32)/255
    rng = np.random.default_rng(seed)
    noisy = np.clip(sar*(1.0+rng.normal(0, .03, sar.shape)), 0, 1)
    shifted = np.roll(np.roll(sar, 1, axis=0), 1, axis=1)
    return {
        'intensity': np.clip(sar*1.02+.005, 0, 1),
        'blur_sigma0.6': cv2.GaussianBlur(sar, (0,0), .6),
        'weak_multiplicative_noise': noisy,
        'shift_1px': shifted,
    }


def stability_metrics(reference, changed, valid, name, threshold):
    if name == 'shift_1px':
        changed = np.roll(np.roll(changed, -1, axis=0), -1, axis=1)
        interior = np.zeros_like(valid)
        interior[3:-3, 3:-3] = True
        valid = valid & interior
    left = reference[valid]
    right = changed[valid]
    corr = float(np.corrcoef(left, right)[0,1]) if left.std() and right.std() else 0.0
    a = (reference >= threshold) & valid
    b = (changed >= threshold) & valid
    union = np.logical_or(a,b).sum()
    iou = float(np.logical_and(a,b).sum()/union) if union else 1.0
    k = max(int(a.sum()), 1)
    valid_indices = np.flatnonzero(valid)
    top_a = valid_indices[np.argpartition(reference.flat[valid_indices], -k)[-k:]]
    top_b = valid_indices[np.argpartition(changed.flat[valid_indices], -k)[-k:]]
    overlap = float(len(np.intersect1d(top_a, top_b))/k)
    return {'corr': corr, 'mae': float(np.mean(np.abs(left-right))),
            'iou': iou, 'topk_overlap': overlap}


def heatmap(values):
    return cv2.applyColorMap(np.rint(np.clip(values,0,1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay(sar_u8, high):
    base = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base); color[...,2] = 255
    alpha = (high.astype(np.float32)*.65)[...,None]
    return np.rint(base*(1-alpha)+color*alpha).astype(np.uint8)


def crop(array, y, x):
    half = PATCH//2
    return array[y-half:y+half, x-half:x+half]


def make_random_contact_sheets(samples):
    for page_start in range(0, len(samples), 20):
        page = samples[page_start:page_start+20]
        tiles = []
        for sample in page:
            o = cv2.cvtColor(crop(sample['optical'], sample['y'], sample['x']), cv2.COLOR_GRAY2BGR)
            s = cv2.cvtColor(crop(sample['sar'], sample['y'], sample['x']), cv2.COLOR_GRAY2BGR)
            r = heatmap(crop(sample['prior'], sample['y'], sample['x']))
            ov = overlay(crop(sample['sar'], sample['y'], sample['x']),
                         crop(sample['prior'], sample['y'], sample['x']) >= .5)
            tile = np.concatenate((o,s,r,ov), axis=1)
            tile = cv2.copyMakeBorder(tile, 18, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f"{sample['index']:03d} {sample['image']} ({sample['x']},{sample['y']}) {sample['morphology_class']}",
                        (3,13), cv2.FONT_HERSHEY_SIMPLEX, .32, (255,255,255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(tile, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
        rows = []
        for start in range(0, len(tiles), 4):
            row = tiles[start:start+4]
            while len(row)<4: row.append(np.zeros_like(tiles[0]))
            rows.append(np.concatenate(row, axis=1))
        cv2.imwrite(str(OUTPUT/f"random_high_{page_start//20+1:02d}.png"), np.concatenate(rows, axis=0))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = FixedLineEvidence.from_paths(CALIBRATION).to(device).eval()
    splits, _ = manifest_pairs(Path('../datasets/QXSLAB_SAROPT'), MANIFEST)
    selected = [splits['val'][position] for position in POSITIONS]
    spatial_rows, stability_rows, images = [], [], []

    for index, (optical_path, sar_path) in enumerate(selected):
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        optical_u8 = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        prior, valid = prior_forward(module, sar_u8, device)
        image_data = {'image': optical_path.stem, 'optical': optical_u8,
                      'sar': sar_u8, 'prior': prior, 'valid': valid}
        images.append(image_data)
        for threshold in THRESHOLDS:
            high = (prior >= threshold) & valid
            labels, meta, stats = component_statistics(high)
            row = {'image': optical_path.stem, 'threshold': threshold,
                   'high_pixel_fraction': float(high.sum()/max(valid.sum(),1))}
            row.update(stats); row.update(structure_tensor_stats(sar_u8, valid, high))
            spatial_rows.append(row)
            if threshold == .5:
                image_data['labels'] = labels; image_data['component_meta'] = meta
        if index < 8:
            panels = [cv2.cvtColor(optical_u8,cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(sar_u8,cv2.COLOR_GRAY2BGR), heatmap(prior),
                      overlay(sar_u8, prior>=.5)]
            canvas = np.concatenate(panels,axis=1)
            cv2.imwrite(str(OUTPUT/(optical_path.stem+'_full.png')),canvas)
        if POSITIONS[index] in STABILITY_POSITIONS:
            for name, changed_float in perturbations(sar_u8, RANDOM_SEED+index).items():
                changed_u8 = np.rint(changed_float*255).clip(0,255).astype(np.uint8)
                changed_prior, changed_valid = prior_forward(module, changed_u8, device)
                for threshold in THRESHOLDS:
                    result = stability_metrics(prior, changed_prior, valid & changed_valid, name, threshold)
                    result.update({'image': optical_path.stem, 'perturbation': name,
                                   'threshold': threshold})
                    stability_rows.append(result)
        print('audited '+optical_path.stem,flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    per_image = int(np.ceil(RANDOM_HIGH_SAMPLES/len(images)))
    for image in images:
        eligible = np.argwhere((image['prior']>=.5) & image['valid'])
        rng.shuffle(eligible)
        selected_points = []
        for y,x in eligible:
            if y<PATCH//2 or x<PATCH//2 or y>=image['prior'].shape[0]-PATCH//2 or x>=image['prior'].shape[1]-PATCH//2:
                continue
            if any((x-x0)**2+(y-y0)**2 < 14**2 for y0,x0 in selected_points):
                continue
            selected_points.append((int(y),int(x)))
            if len(selected_points)>=per_image: break
        for y,x in selected_points:
            label = int(image['labels'][y,x]); meta = image['component_meta'][label]
            area, elongation = meta['area'], meta['elongation']
            if area <= 5: morphology = 'tiny_fragment'
            elif area <= 20: morphology = 'small_component'
            elif elongation >= 3: morphology = 'elongated_component'
            else: morphology = 'large_cluster'
            samples.append({'image': image['image'],'x':x,'y':y,
                            'R0_center':float(image['prior'][y,x]),
                            'component_area':area,'component_elongation':elongation,
                            'morphology_class':morphology,
                            'optical':image['optical'],'sar':image['sar'],'prior':image['prior']})
    rng.shuffle(samples); samples=samples[:RANDOM_HIGH_SAMPLES]
    for i,sample in enumerate(samples): sample['index']=i+1
    make_random_contact_sheets(samples)

    def write_csv(path, rows, fields=None):
        with path.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv(OUTPUT/'spatial_stats.csv',spatial_rows)
    write_csv(OUTPUT/'stability.csv',stability_rows)
    sample_rows=[{k:v for k,v in s.items() if k not in ('optical','sar','prior')} for s in samples]
    write_csv(OUTPUT/'random_high_samples.csv',sample_rows)

    summary={}
    for threshold in THRESHOLDS:
        subset=[r for r in spatial_rows if r['threshold']==threshold]
        summary[str(threshold)]={k:float(np.mean([r[k] for r in subset]))
                                 for k in subset[0] if k not in ('image','threshold')}
    stability_summary={}
    for perturbation in sorted({r['perturbation'] for r in stability_rows}):
        stability_summary[perturbation]={}
        for threshold in THRESHOLDS:
            subset=[r for r in stability_rows if r['perturbation']==perturbation and r['threshold']==threshold]
            stability_summary[perturbation][str(threshold)]={k:float(np.mean([r[k] for r in subset]))
                                                             for k in ('corr','mae','iou','topk_overlap')}
    morph={name:sum(s['morphology_class']==name for s in samples) for name in
           ('tiny_fragment','small_component','elongated_component','large_cluster')}
    report={'positions':POSITIONS,'threshold_summary':summary,
            'stability_summary':stability_summary,
            'random_high_sample_count':len(samples),'random_high_morphology':morph,
            'note':'Morphology classes describe R0 connected components, not semantic ground truth.'}
    (OUTPUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
