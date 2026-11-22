"""Train-only projections and conservative evidence rules for the MaRS probe.

No backbone training or fusion-quality claims. Decision thresholds are fixed
before validation; bootstrap units are scenes, never sampled pixels.
"""
import hashlib

import cv2
import numpy as np


DECISION_RULES = {
    "min_train_scenes": 32,
    "min_unaudited_scenes": 16,
    "mse_relative_gain": 0.05,
    "high_edge_mae_relative_gain": 0.03,
    "scene_win_fraction": 0.60,
    "bootstrap_replicates": 2000,
    "min_missing_lines": 3,
    "min_disconnected_lines": 12,
    "missing_longest_run_gain": 0.05,
    "negative_response_tolerance": 0.02,
}


def resize_channels(features, size):
    if features.ndim != 3 or not np.isfinite(features).all():
        raise ValueError("Expected finite CHW features")
    if features.shape[1:] == (size, size):
        return features.astype(np.float32, copy=False)
    image = cv2.resize(features.transpose(1, 2, 0), (size, size),
                       interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = image[..., None]
    return image.transpose(2, 0, 1).astype(np.float32)


def source_structure_features(gray, size):
    """32 inexpensive optical features; never include the exact target.

    Magnitudes at sigma=0/1/2/4 intentionally form a strong low-cost control.
    The target uses sigma=1.25. Pixel reconstruction favors this family, so
    beating it on pixel MSE is not an admission rule; audited lines are compared.
    """
    x = gray.astype(np.float32) / 255.0
    if x.shape != (256, 256):
        raise ValueError("Expected a 256x256 optical source")
    maps = []
    for sigma in (0, 1, 2, 4):
        smooth = x if sigma == 0 else cv2.GaussianBlur(x, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        maps.extend((smooth, gx, gy, np.abs(gx), np.abs(gy),
                     np.sqrt(gx*gx + gy*gy)))
    dense = resize_channels(np.stack(maps), size)
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    high = np.stack(((-a-b+c+d)/2, (-a+b-c+d)/2, (a-b-c+d)/2))
    bands = np.concatenate((high, np.abs(high),
                            np.sqrt(np.sum(high**2, axis=0, keepdims=True))))
    low = cv2.GaussianBlur(x, (0, 0), 2)
    variance = np.maximum(cv2.GaussianBlur(x*x, (0, 0), 2)-low*low, 0)
    return np.concatenate((dense, resize_channels(bands, size),
                           resize_channels(np.sqrt(variance)[None], size)))


def fit_projection(rows, dimensions):
    """Standardized PCA fitted exclusively on sampled TRAIN features."""
    x = np.asarray(rows, np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Projection needs finite training rows")
    if not 1 <= dimensions <= x.shape[1] or x.shape[0] <= dimensions*2:
        raise ValueError("Insufficient training rows/channels for projection")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-4)
    z = (x-mean)/std
    covariance = z.T@z/len(z)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:dimensions]
    basis = vectors[:, order]
    # Pin eigenvector signs for reproducible saved heads.
    for column in range(dimensions):
        pivot = np.argmax(np.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return dict(mean=mean, std=std, basis=basis,
                explained_variance=float(np.maximum(values[order], 0).sum()
                                         / max(np.maximum(values, 0).sum(), 1e-12)),
                effective_rank=int(np.sum(values > max(float(values.max()), 1e-12)*1e-8)))


def project_rows(projection, rows):
    x = np.asarray(rows, np.float64)
    if x.ndim != 2 or x.shape[1] != len(projection["mean"]):
        raise ValueError("Projection channel mismatch")
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite projected features")
    return (((x-projection["mean"])/projection["std"])@projection["basis"]).astype(np.float32)


def fit_residualizers(base_rows, projected_branches):
    """Remove TRAIN-linear redundancy with E2 without looking at edge labels.

    Otherwise duplicating an existing E2 feature can reduce Ridge shrinkage
    and masquerade as new information. All branches use the same E2 subspace.
    A shared eigensolve avoids repeating a large least-squares decomposition.
    """
    x=np.asarray(base_rows,np.float64)
    mean=x.mean(0);std=np.maximum(x.std(0),1e-4)
    z=(x-mean)/std
    covariance=z.T@z/len(z)
    values,vectors=np.linalg.eigh(covariance)
    active=values>max(float(values.max()),1e-12)*1e-8
    names=sorted(projected_branches)
    branch_means={name:np.asarray(projected_branches[name],np.float64).mean(0) for name in names}
    centered=np.concatenate([projected_branches[name]-branch_means[name] for name in names],axis=1)
    cross=z.T@centered/len(z)
    coefficients=vectors[:,active]@((vectors[:,active].T@cross)/values[active,None])
    output={};offset=0
    for name in names:
        dimensions=projected_branches[name].shape[1]
        output[name]=dict(mean=mean,std=std,branch_mean=branch_means[name],
                         coef=coefficients[:,offset:offset+dimensions],
                         e2_effective_rank=int(active.sum()))
        offset+=dimensions
    return output


def coherent_path_metrics(corridor, threshold=0.20):
    """Smooth offset path, longest active run, and largest inactive gap.

    A normal offset can change by at most one pixel per along-line sample.
    A fixed 0.02 penalty discourages wandering to unrelated nearby edges.
    These are path proxies in an annotated corridor, not recovered topology.
    """
    values = np.asarray(corridor, np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite offsets-by-samples corridor")
    offsets, count = values.shape
    scores = values[:, 0].copy()
    parent = np.zeros((offsets, count), dtype=np.int64)
    for sample in range(1, count):
        next_scores = np.empty(offsets)
        for offset in range(offsets):
            previous = np.arange(max(0, offset-1), min(offsets, offset+2))
            candidates = scores[previous]-0.02*np.abs(previous-offset)
            choice = int(np.argmax(candidates))
            parent[offset, sample] = previous[choice]
            next_scores[offset] = candidates[choice]+values[offset, sample]
        scores = next_scores
    path = np.zeros(count, dtype=np.int64)
    path[-1] = np.argmax(scores)
    for sample in range(count-1, 0, -1):
        path[sample-1] = parent[path[sample], sample]
    response = values[path, np.arange(count)]
    active = response >= threshold
    def longest(mask):
        best = run = 0
        for value in mask:
            run = run+1 if value else 0
            best = max(best, run)
        return best/count
    return dict(path_mean=float(response.mean()),
                path_coverage=float(active.mean()),
                longest_run_fraction=float(longest(active)),
                max_gap_fraction=float(longest(~active)),
                path_offset_variation=float(np.abs(np.diff(path)).mean()))


def paired_evidence(rows, methods, candidate, references, metrics, bootstrap=2000):
    """All methods share the SAME finite scenes for each stage/cohort/metric.

    Positive gains always favor the candidate, regardless of metric direction.
    Lines are first averaged within scene, so multiple labels are not treated
    as independent scenes by the paired bootstrap.
    """
    if candidate not in methods or not set(references).issubset(methods):
        raise ValueError("Unknown method in paired comparison")
    out = []
    for stage in sorted({row["stage"] for row in rows}):
        for cohort in sorted({row["cohort"] for row in rows if row["stage"] == stage}):
            chosen = [row for row in rows if row["stage"] == stage and row["cohort"] == cohort]
            scenes = sorted({row["scene"] for row in chosen})
            for metric, higher_is_better in metrics.items():
                cells = {}
                for row in chosen:
                    key = (row["scene"], row["method"])
                    cells.setdefault(key, []).append(float(row[metric]))
                finite = [scene for scene in scenes if all(
                    (scene, method) in cells and np.isfinite(cells[(scene, method)]).all()
                    for method in methods)]
                for reference in references:
                    a = np.array([np.mean(cells[(scene, candidate)]) for scene in finite])
                    b = np.array([np.mean(cells[(scene, reference)]) for scene in finite])
                    direction = not higher_is_better if cohort == "disconnected" else higher_is_better
                    gain = (a-b) if direction else (b-a)
                    key = f"{stage}:{cohort}:{metric}:{candidate}:{reference}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    if len(gain):
                        boot = gain[rng.integers(0, len(gain), (bootstrap, len(gain)))].mean(axis=1)
                        ci = np.quantile(boot, [0.025, 0.975])
                    else:
                        ci = [None, None]
                    out.append(dict(stage=stage, cohort=cohort, metric=metric,
                        method=candidate, reference=reference, n_scenes=len(finite),
                        scene_ids=finite, n_items=sum(len(cells[(s, candidate)]) for s in finite),
                        candidate_mean=float(a.mean()) if len(a) else None,
                        reference_mean=float(b.mean()) if len(b) else None,
                        mean_gain=float(gain.mean()) if len(gain) else None,
                        median_gain=float(np.median(gain)) if len(gain) else None,
                        relative_gain=float(gain.mean()/max(abs(b.mean()), 1e-10)) if len(gain) else None,
                        wins=int(np.sum(gain > 0)),
                        ci95_low=float(ci[0]) if len(gain) else None,
                        ci95_high=float(ci[1]) if len(gain) else None))
    return out


def admission_decision(pixel_pairs, line_pairs, random_methods, train_count, input_mode):
    """Predeclared, stage-consistent screening; never certify fusion benefit."""
    rules = dict(DECISION_RULES)
    def lookup(rows, stage, cohort, reference, metric):
        return next((r for r in rows if r["stage"] == stage and r["cohort"] == cohort
                     and r["reference"] == reference and r["metric"] == metric), None)
    def reliable(row, relative=0):
        return bool(row and row["n_scenes"] >= rules["min_unaudited_scenes"]
                    and row["relative_gain"] >= relative
                    and row["median_gain"] > 0 and row["ci95_low"] > 0
                    and row["wins"]/row["n_scenes"] >= rules["scene_win_fraction"])
    stages = []
    for stage in (0, 1):
        mse = lookup(pixel_pairs, stage, "unaudited", "e2_rich", "mse")
        high = lookup(pixel_pairs, stage, "unaudited", "e2_rich", "high_edge_mae")
        enough = bool(mse and mse["n_scenes"] >= rules["min_unaudited_scenes"])
        pixel_checks = {
            "mse_increment": reliable(mse, rules["mse_relative_gain"]),
            "high_edge_increment": reliable(high, rules["high_edge_mae_relative_gain"]),
        }
        for reference in ["e2_plus_shifted"]+list(random_methods):
            pixel_checks["beats_"+reference] = reliable(
                lookup(pixel_pairs, stage, "unaudited", reference, "mse"))
        line_checks = {}
        for reference in ("e2_rich", "e2_plus_source"):
            missing = lookup(line_pairs, stage, "missing", reference, "longest_run_fraction")
            gap = lookup(line_pairs, stage, "missing", reference, "max_gap_fraction")
            negative = lookup(line_pairs, stage, "disconnected", reference, "longest_run_fraction")
            coverage = lookup(line_pairs, stage, "disconnected", reference, "path_coverage")
            line_checks["missing_vs_"+reference] = bool(
                missing and missing["n_items"] >= rules["min_missing_lines"]
                and missing["mean_gain"] >= rules["missing_longest_run_gain"]
                and missing["wins"] >= int(np.ceil(missing["n_scenes"]*2/3))
                and gap and gap["mean_gain"] >= 0)
            line_checks["negative_vs_"+reference] = bool(
                negative and negative["n_items"] >= rules["min_disconnected_lines"]
                and negative["mean_gain"] >= -rules["negative_response_tolerance"]
                and negative["ci95_low"] >= -rules["negative_response_tolerance"]
                and coverage and coverage["mean_gain"] >= -rules["negative_response_tolerance"]
                and coverage["ci95_low"] >= -rules["negative_response_tolerance"])
        stages.append(dict(stage=stage, enough_scenes=enough,
                           pixel_checks=pixel_checks, line_checks=line_checks,
                           pixel_pass=all(pixel_checks.values()),
                           line_pass=all(line_checks.values())))
    if train_count < rules["min_train_scenes"] or len(random_methods) < 3 or not all(s["enough_scenes"] for s in stages):
        status = "insufficient_data"
        message = "样本数或随机初始化对照不足；仅作流程检查，不决定是否接入。"
    elif all(s["pixel_pass"] and s["line_pass"] for s in stages):
        status = "candidate_for_small_adapter_ablation"
        message = "两个尺度均通过增量读出与人工线段筛查，可进入小规模受控Adapter对照；尚未证明融合收益。"
    elif any(s["pixel_pass"] for s in stages):
        status = "inconclusive_structure_or_stage"
        message = "部分尺度通过或结构安全性不足，当前证据不足以推进接入；禁止事后只选有利尺度。"
    else:
        status = "not_supported_for_current_probe_route"
        message = "未通过当前前两阶段、固定投影及读出路线的增量门槛；不等同于证明全部MaRS路线无效。"
    if input_mode != "gray" and status == "candidate_for_small_adapter_ablation":
        status = "exploratory_rgb_requires_gray_confirmation"
        message = "RGB引入了额外颜色信息；只能作为探索结果，需灰度主实验确认。"
    return dict(status=status, message=message, rules=rules, stages=stages,
                scope="冻结源结构探针；固定人工审计仅3条missing线，不能视为拓扑真值或融合质量证明。")
