"""RCT: train-only linear structural prediction and frozen image-transfer audit.

Commands fit/evaluate/all process datasets and must be run by the user. smoke
uses ONE source scene, no fitting, no optimizer and no fusion-quality claims.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rct_predictive_transfer as core
from diagnose_e2_information_path import image_u8, locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from diagnose_fcsr_decoder import gradients, signed_transfer, source_specific_mask
from maintain_experiment_index import sha256_file
from mars_probe_evidence import coherent_path_metrics
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT
from utils.dataset_os import evenly_spaced, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "datasets/QXSLAB_SAROPT"
FIT_DIR = ROOT / "models/rct_predictive_transfer_v1/train128_seed42"
RESULT_DIR = ROOT / "test_result/rct_predictive_transfer_v1/train128_val64_audit16_seed42"
LABEL_DIRS = (ROOT / "test_result/stg_no_train_v1/val8",
              ROOT / "test_result/e2_contour_evidence_v1/holdout_labels")
METRICS = tuple(FusionEvaluator.METRIC_NAMES)
TRANSFERS = ("optical_transfer", "sar_transfer")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def completion_manifest(folder, **extra):
    return dict(artifacts_sha256={p.relative_to(folder).as_posix(): sha256_file(p)
                for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "SUCCESS.json"}, **extra)


def verify_completion(folder):
    success = read_json(folder/"SUCCESS.json")
    if success["artifacts_sha256"] != completion_manifest(folder)["artifacts_sha256"]:
        raise ValueError("Completed artifacts changed; refusing to reuse results")
    return success


def write_csv(path, rows):
    if not rows:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True).encode()).hexdigest()


def source_hashes():
    names = (Path(__file__).name, "rct_predictive_transfer.py", "direct_fusion.py", "lwpd.py", "net.py",
             "e2_base_anchor.py", "diagnose_fcsr_decoder.py", "diagnose_e2_subspace_gradients.py",
             "diagnose_e2_information_path.py", "mars_probe_evidence.py", "utils/fusion_metrics.py",
             "utils/reference_metrics.py", "utils/Evaluator.py")
    return {name: sha256_file(ROOT/name) for name in names}


def scene_files(manifest, scene):
    return locate_scene(DATA, manifest, str(scene))


def inputs_hashes(manifest, scenes):
    return {str(scene): [sha256_file(Path(path)) for path in scene_files(manifest, scene)] for scene in scenes}


def manifest_and_annotations():
    manifest = source_manifest()
    validate_manifest_splits(manifest)
    labels, audited, hashes = [], [], {}
    for folder in LABEL_DIRS:
        sample = read_json(folder/"sample.json")
        if (sha256_file(ROOT/sample["source_manifest"]) != sample["source_manifest_sha256"]
                or sha256_file(ROOT/sample["archive_metadata"]) != sample["archive_metadata_sha256"]):
            raise ValueError("Source annotation archive has changed")
        archive = read_json(ROOT/sample["archive_metadata"])
        if (Path(archive["checkpoints"]["e2"]) != ORIGINAL_LOSS_PILOT.final
                or archive["checkpoint_hashes"]["e2"] != sha256_file(ROOT/ORIGINAL_LOSS_PILOT.final)):
            raise ValueError("The frozen checkpoint no longer matches the archived E2")
        if set(audited) & set(sample["scene_ids"]):
            raise ValueError("Overlapping manual cohorts")
        audited.extend(sample["scene_ids"])
        with (folder/"e2_visibility.csv").open(encoding="utf-8-sig", newline="") as handle:
            visibility = {(x["scene"], x["id"]): x["e2_visibility"] for x in csv.DictReader(handle)}
        with (folder/"source_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
            labels.extend(dict(row, visibility=visibility.get((row["scene"], row["id"]), "uncertain"))
                          for row in csv.DictReader(handle))
        for name in ("sample.json", "source_labels.csv", "e2_visibility.csv"):
            hashes[(folder/name).relative_to(ROOT).as_posix()] = sha256_file(folder/name)
    if not set(audited).issubset(manifest["splits"]["val"]) or len(audited) != 16:
        raise ValueError("Expected the two fixed, disjoint, annotated validation cohorts")
    return manifest, labels, audited, hashes


def describe_scene(manifest, scene):
    optical, sar = read_pair(scene_files(manifest, scene))
    result = {}
    for name, gray in zip(core.SOURCES, (optical, sar)):
        image = to_tensor(gray, "cpu")
        valid = image > 0 if name == "sar" else torch.ones_like(image, dtype=torch.bool)
        result[name] = core.structure_descriptor(image, valid)
    return result


def designs(current, partner, source):
    other = "sar" if source == "optical" else "optical"
    own, mag, own_valid = current[source]
    opposite, _, other_valid = current[other]
    wrong, _, wrong_valid = partner[other]
    features, target, mask = core.prediction_design(own, opposite, own_valid, other_valid, wrong, wrong_valid)
    return features, target, mask, mag


def head_key(source, method, parameter):
    return source+"__"+method+"__"+parameter


def save_heads(path, heads):
    payload = {head_key(source, method, key): value for source, family in heads.items()
               for method, head in family.items() for key, value in head.items()}
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def load_heads(path):
    with np.load(path, allow_pickle=False) as archive:
        return {source: {method: {key: archive[head_key(source, method, key)].copy()
                                  for key in ("x_mean", "x_scale", "y_mean", "y_scale", "coef",
                                              "train_rows", "effective_rank", "context_columns", "extra_residual_rank")}
                         for method in core.PREDICTORS} for source in core.SOURCES}


def register_summary(folder, metadata, summary, task, split, count, metrics):
    from run_e2_matched_no_pac import register_many
    token = digest(dict(metadata=metadata, result_dir=str(folder)))[:16].upper()
    register_many([dict(id="RCT-"+task.upper()+"-"+token, method="RCT "+task,
        task=task, status="diagnostic_fixed_subset", comparable_group="RCT-"+task.upper()+"-"+token,
        dataset="QXS-SAROPT", split=split, offset=0, sample_count=count,
        checkpoint=ORIGINAL_LOSS_PILOT.final.as_posix(), checkpoint_sha256=metadata["checkpoint_sha256"],
        split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
        split_manifest_sha256=metadata["split_manifest_sha256"],
        metric_profile="source_hog_prediction_v1" if task == "probe_fit" else DEFAULT_PROFILE,
        output_policy="train_only_ridge_fit" if task == "probe_fit" else "frozen_interpolation_clamp_round_uint8",
        result_dir=relative_path(folder),
        metadata_path=(folder/"run_metadata.json").as_posix(), metrics=metrics,
        metric_source=dict(type="json_paths", path=(folder/"summary.json").as_posix(),
                           paths={key: [key] for key in metrics}),
        notes="Frozen E2 and fixed HOG linear probes; no RCT fusion training or test-set access. "
              "Inference-time conditions use observed source structural targets; not blind denoising." )])


def fit(args, manifest):
    selected = evenly_spaced(manifest["splits"]["train"], args.train_scenes)
    calibration = selected[3::4]
    fitting = [scene for scene in selected if scene not in set(calibration)]
    if len(fitting) < 16 or len(calibration) < 4:
        raise ValueError("Need >=16 fit and >=4 calibration scenes")
    config = dict(train_scenes=selected, fit_scenes=fitting, calibration_scenes=calibration,
                  rows_per_scene=args.rows_per_scene, ridge=args.ridge, seed=args.seed,
                  descriptor="fixed_unsigned_HOG9_pool4_support12", context_offsets=core.CONTEXT_OFFSETS,
                  wide_offsets=core.WIDE_OFFSETS, extra_features_residualized_on_train_context=True,
                  rules=core.RULES)
    metadata = dict(config=clean_json(config), source_hashes=source_hashes(),
                    checkpoint_sha256=sha256_file(ROOT/ORIGINAL_LOSS_PILOT.final),
                    split_manifest_sha256=sha256_file(ROOT/ORIGINAL_LOSS_PILOT.manifest),
                    inputs_sha256=inputs_hashes(manifest, selected),
                    backbone_used_for_fitting=False, fitting="closed_form_ridge_train_only",
                    versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__))
    folder = args.fit_dir
    if folder.exists():
        if not (folder/"SUCCESS.json").is_file() or read_json(folder/"run_metadata.json") != metadata:
            raise FileExistsError("Preserve incomplete or different probe fit; use a new --fit-dir")
        if sha256_file(folder/"probe.npz") != verify_completion(folder)["probe_sha256"]:
            raise ValueError("Saved probe changed")
        print("Reusing verified train-only probe: "+str(folder), flush=True)
        summary = read_json(folder/"summary.json")
        register_summary(folder, metadata, summary, "probe_fit", "train", len(selected),
                         {k: summary[k] for k in ("fit_scene_count", "calibration_scene_count")})
        return metadata, summary
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"run_metadata.json", metadata)
    width = (len(core.CONTEXT_OFFSETS)+1)*core.DESCRIPTOR_CHANNELS
    moments = {s: {m: core.RidgeMoments(width, core.DESCRIPTOR_CHANNELS) for m in core.PREDICTORS}
               for s in core.SOURCES}
    partners = core.mismatched_partners(fitting, args.seed)
    counts = []
    with torch.inference_mode():
        for index, scene in enumerate(fitting):
            current, other = describe_scene(manifest, scene), describe_scene(manifest, partners[str(scene)])
            for source in core.SOURCES:
                features, target, mask, _ = designs(current, other, source)
                take = core.sample_rows(mask, args.rows_per_scene, core.scene_seed(scene, args.seed))
                y = target.reshape(-1, core.DESCRIPTOR_CHANNELS)[take]
                for method in core.PREDICTORS:
                    moments[source][method].add(features[method].reshape(-1, width)[take], y)
                counts.append(dict(scene=scene, source=source, sampled_rows=len(take), eligible=int(mask.sum())))
            print("fit %d/%d %s"%(index+1, len(fitting), scene), flush=True)
    heads = {s: {m: moments[s][m].finish(args.ridge, width-core.DESCRIPTOR_CHANNELS)
                 for m in core.PREDICTORS} for s in core.SOURCES}
    calibration_partners = core.mismatched_partners(calibration, args.seed+1)
    magnitudes, gains = {s: [] for s in core.SOURCES}, {s: [] for s in core.SOURCES}
    with torch.inference_mode():
        for scene in calibration:
            current = describe_scene(manifest, scene)
            partner = describe_scene(manifest, calibration_partners[str(scene)])
            for source in core.SOURCES:
                features, target, mask, mag = designs(current, partner, source)
                errors = core.prediction_errors(heads[source], features, target)
                magnitudes[source].extend(mag[mask].tolist())
                gains[source].extend(np.maximum(errors["self"]-errors["cross"], 0)[mask].tolist())
    if any(not magnitudes[source] for source in core.SOURCES):
        raise ValueError("No common valid positions in the train-only calibration fold")
    normalizers = {s: dict(magnitude_scale=max(float(np.quantile(magnitudes[s], .95)), 1e-5),
                          gain_scale=max(float(np.quantile(gains[s], .95)), 1e-5),
                          positive_gain_fraction=float(np.mean(np.asarray(gains[s]) > 0))) for s in core.SOURCES}
    if metadata["inputs_sha256"] != inputs_hashes(manifest, selected) or metadata["source_hashes"] != source_hashes():
        raise ValueError("Fit inputs or source code changed during fitting")
    save_heads(folder/"probe.npz", heads)
    summary = dict(fit_scene_count=len(fitting), calibration_scene_count=len(calibration),
                   normalizers=normalizers, ranks={s: {m: int(heads[s][m]["effective_rank"]) for m in core.PREDICTORS}
                                                  for s in core.SOURCES},
                   extra_residual_ranks={s: {m: int(heads[s][m]["extra_residual_rank"]) for m in core.PREDICTORS}
                                         for s in core.SOURCES},
                   scope="Fixed local HOG linear predictability, not fusion quality or independent speckle observations")
    write_csv(folder/"fit_sample_counts.csv", counts)
    write_json(folder/"summary.json", summary)
    write_json(folder/"SUCCESS.json", completion_manifest(folder, probe_sha256=sha256_file(folder/"probe.npz")))
    register_summary(folder, metadata, summary, "probe_fit", "train", len(selected),
                     {k: summary[k] for k in ("fit_scene_count", "calibration_scene_count")})
    return metadata, summary


def prediction_comparisons(rows, audited):
    result = []
    for source in core.SOURCES:
        for cohort in ("all", "audited", "unaudited"):
            selected = [r for r in rows if r["source"] == source and
                        (cohort == "all" or (r["scene"] in audited) == (cohort == "audited"))]
            for reference in ("constant", "self", "self_wide", "rolled", "swapped"):
                stats = core.paired_stats([r["cross"] for r in selected], [r[reference] for r in selected],
                                         True, core.scene_seed(source+cohort+reference))
                result.append(dict(source=source, cohort=cohort, reference=reference, **stats))
    return result


def prepare_lines(labels, scene, optical, sar):
    """Source-only corridor/polarity selection, before looking at fused images."""
    go, gs = gradients(to_tensor(optical, "cpu")), gradients(to_tensor(sar, "cpu"))
    source_gradients = {s: g[0].numpy()*255 for s, g in zip(core.SOURCES, (go, gs))}
    lines, records = [], []
    for row in labels:
        if row["scene"] != scene or row["label"] == "uncertain":
            continue
        a = np.array([float(row["x1"]), float(row["y1"])])
        b = np.array([float(row["x2"]), float(row["y2"])])
        normal = np.array([-(b-a)[1], (b-a)[0]])/np.linalg.norm(b-a)
        centers = a + np.linspace(.15, .85, 33)[:, None]*(b-a)
        corridor = (centers[None]+np.arange(-6, 7)[:, None, None]*normal).astype(np.float32)
        if row["label"] == "disconnected":
            lines.append(dict(row=row, cohort="disconnected", normal=normal, corridor=corridor))
            continue
        for source in core.SOURCES:
            g = source_gradients[source]
            projected = g[0]*normal[0]+g[1]*normal[1]
            samples = cv2.remap(projected, corridor[..., 0], corridor[..., 1], cv2.INTER_LINEAR)
            best = int(np.argmax(np.abs(samples.mean(1))))
            target = float(samples[best].mean())
            accepted = abs(target) >= 2.0
            records.append(dict(scene=scene, id=row["id"], source=source, accepted=accepted,
                                source_signed_response=target, offset=best-6, visibility=row["visibility"]))
            if accepted:
                cohort = source+"_"+row["visibility"]
                lines.append(dict(row=row, cohort=cohort, normal=normal, corridor=corridor[best:best+1], target=target))
    return lines, records


def line_readout(fused, lines, method):
    gradient = gradients(fused.detach().cpu())[0].numpy()*255
    rows = []
    for line in lines:
        normal, corridor = line["normal"], line["corridor"]
        projected = gradient[0]*normal[0]+gradient[1]*normal[1]
        response = cv2.remap(projected, corridor[..., 0], corridor[..., 1], cv2.INTER_LINEAR)
        if line["cohort"] == "disconnected":
            values = coherent_path_metrics(np.abs(response)/10, threshold=.2)
        else:
            signed = response*np.sign(line["target"])
            active = signed[0] >= .5*abs(line["target"])
            best = run = 0
            for value in active:
                run = run+1 if value else 0
                best = max(best, run)
            values = dict(signed_response=float(signed.mean()), continuity=float(active.mean()),
                          longest_run_fraction=best/len(active), polarity_reversal=float(signed.mean() < 0))
        rows.append(dict(scene=line["row"]["scene"], id=line["row"]["id"], cohort=line["cohort"],
                         method=method, **values))
    return rows


def image_readout(model, o, s, optical, sar, base, detail, fused, masks, method, out):
    gray = image_u8(fused)
    directory = out/"gray"/method
    directory.mkdir(parents=True, exist_ok=True)
    scores = dict(zip(METRICS, map(float, FusionEvaluator.metric_vector(gray, optical, sar))))
    g = gradients(fused)
    for source, image in (("optical", o), ("sar", s)):
        scores[source+"_transfer"] = signed_transfer(gradients(image), g, masks[source])
    return gray, scores


def save_preview(folder, scene, optical, sar, images):
    cells = [("Optical Y", optical), ("SAR", sar), ("E2", images["e2"])]
    for source in core.SOURCES:
        for condition in ("predictive", "native", "uniform"):
            key = source+"_"+condition+"_plus"
            cells.append((key, images.get(key, images["e2"])))
    canvas = np.zeros((3*(optical.shape[0]+24), 3*optical.shape[1]), np.uint8)
    for index, (title, gray) in enumerate(cells):
        row, column = divmod(index, 3)
        y, x = row*(optical.shape[0]+24), column*optical.shape[1]
        canvas[y+24:y+24+gray.shape[0], x:x+gray.shape[1]] = gray
        cv2.putText(canvas, title, (x+4, y+17), cv2.FONT_HERSHEY_SIMPLEX, .40, 255, 1, cv2.LINE_AA)
    directory = folder/"previews"
    directory.mkdir(exist_ok=True)
    if not cv2.imwrite(str(directory/(scene+".png")), canvas):
        raise OSError("Cannot save source / E2 / condition contact sheet")


def render_scene(model, manifest, scene, maps, labels, args, out):
    optical, sar = read_pair(scene_files(manifest, scene))
    o, s = (to_tensor(x, args.device) for x in (optical, sar))
    encoded = model.encode_pair(o, s)
    base, detail = model.fuse_features(*encoded)
    baseline = model.reconstruct(o, base, detail)
    src = dict(optical=encoded[1], sar=encoded[3])
    go, gs = gradients(o), gradients(s)
    masks = dict(optical=source_specific_mask(go, gs, s > 0), sar=source_specific_mask(gs, go, s > 0))
    lines, qualifications = prepare_lines(labels, scene, optical, sar)
    conditions, directions = {}, {}
    for source in core.SOURCES:
        for name in core.CONDITIONS:
            condition = torch.from_numpy(maps[source][name].astype(np.float32))[None, None].to(args.device)
            condition = F.interpolate(condition, size=o.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
            conditions[source+"_"+name] = condition
            direction = core.interpolation_direction(src[source], detail, condition)
            for sign in (-1, 1):
                method = source+"_"+name+("_plus" if sign > 0 else "_minus")
                directions[method] = direction*sign
    nominal = {name: float((model.reconstruct(o, base, detail+args.max_interpolation*d)-baseline).abs().mean()*255)
               for name, d in directions.items()}
    target = core.common_effect_target(nominal)
    out_rows, probe_rows, effects = [], [], []
    gray, scores = image_readout(model, o, s, optical, sar, base, detail, baseline, masks, "e2", out)
    if not cv2.imwrite(str(out/"gray/e2"/(scene+".png")), gray):
        raise OSError("Cannot write E2 reference")
    archive = ROOT/"test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/e2/gray"/(scene+".png")
    saved = cv2.imread(str(archive), cv2.IMREAD_GRAYSCALE)
    if saved is None or saved.shape != gray.shape or np.abs(saved.astype(int)-gray.astype(int)).max() > 1:
        raise ValueError("Frozen E2 does not reproduce the archived output within one uint8 level")
    out_rows.append(dict(scene=scene, method="e2", image_mae=0., detail_rms_delta=0., changed_fraction=0., **scores))
    probe_rows.extend(line_readout(baseline, lines, "e2"))
    if target is None:
        return out_rows, probe_rows, [dict(scene=scene, method=k, nominal_mae=v,
               status="insufficient_condition_effect") for k, v in nominal.items()], qualifications
    preview_images = {"e2": gray}
    for method, direction in directions.items():
        def measure(alpha):
            return float((model.reconstruct(o, base, detail+alpha*direction)-baseline).abs().mean()*255)
        alpha, effect, error = core.calibrate_effect(measure, target, upper=args.max_interpolation)
        modified = detail+alpha*direction
        fused = model.reconstruct(o, base, modified)
        gray, scores = image_readout(model, o, s, optical, sar, base, detail, fused, masks, method, out)
        if not cv2.imwrite(str(out/"gray"/method/(scene+".png")), gray):
            raise OSError("Cannot save diagnostic fusion image")
        changed = float((gray != image_u8(baseline)).mean())
        if method.endswith("_plus"):
            preview_images[method] = gray
        condition_key = method.rsplit("_", 1)[0]
        support = conditions[condition_key] > 1e-6
        displacement = (fused-baseline).abs()
        outside_share = float((displacement*(~support)).sum()/displacement.sum().clamp_min(1e-12))
        out_rows.append(dict(scene=scene, method=method, image_mae=effect,
                             detail_rms_delta=float((modified-detail).square().mean().sqrt()),
                             changed_fraction=changed, outside_condition_effect_fraction=outside_share, **scores))
        probe_rows.extend(line_readout(fused, lines, method))
        effects.append(dict(scene=scene, method=method, nominal_mae=nominal[method], target_mae=target,
                            achieved_mae=effect, alpha=alpha, relative_matching_error=error, status="matched"))
    maps_dir = out/"maps"
    maps_dir.mkdir(exist_ok=True)
    with (maps_dir/(scene+".npz")).open("xb") as handle:
        np.savez_compressed(handle, **{source+"_"+k: v for source, family in maps.items() for k, v in family.items()})
    save_preview(out, scene, optical, sar, preview_images)
    return out_rows, probe_rows, effects, qualifications


def comparison_rows(rows, metrics, cohort="image"):
    """Common finite SCENES across every method; lines averaged within scenes."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scene"], row["method"]), []).append(row)
    methods = sorted({r["method"] for r in rows})
    results = []
    for metric in metrics:
        cells = {}
        for key, items in grouped.items():
            vals = [r.get(metric) for r in items]
            if vals and all(v is not None and math.isfinite(float(v)) for v in vals):
                cells[key] = float(np.mean(vals))
        scenes = sorted(scene for scene in {r["scene"] for r in rows}
                        if all((scene, method) in cells for method in methods))
        for source in core.SOURCES:
            for sign in ("plus", "minus"):
                candidate = source+"_predictive_"+sign
                references = ["e2"]+[source+"_"+c+"_"+sign for c in core.CONDITIONS if c != "predictive"]
                if candidate not in methods:
                    continue
                for reference in references:
                    if reference not in methods:
                        continue
                    lower = metric in ("Nabf", "polarity_reversal") or (cohort == "disconnected" and metric in
                            ("path_mean", "path_coverage", "longest_run_fraction"))
                    stats = core.paired_stats([cells[(s, candidate)] for s in scenes],
                                             [cells[(s, reference)] for s in scenes], lower,
                                             core.scene_seed(cohort+candidate+reference+metric))
                    line_count = sum(r["scene"] in scenes and r["method"] == candidate for r in rows)
                    results.append(dict(cohort=cohort, source=source, sign=sign, candidate=candidate, line_count=line_count,
                                        reference=reference, metric=metric, scene_ids=" ".join(scenes), **stats))
    return results


def transfer_decision(prediction, comparisons, image_rows):
    evidence = {}
    expected_scenes = {r["scene"] for r in image_rows if r["method"] == "e2"}
    for source in core.SOURCES:
        own = [r for r in comparisons if r["cohort"] == "image" and r["source"] == source
               and r["sign"] == "plus" and r["metric"] == source+"_transfer" and r["reference"] != "e2"]
        local = len(own) == 5 and all(r["n"] >= core.RULES["min_transfer_scenes"] and r["ci_low"] > 0
                                     and r["improved"]/r["n"] >= .6 for r in own)
        guard = [r for r in comparisons if r["cohort"] == "image" and r["source"] == source
                 and r["sign"] == "plus" and r["reference"] == "e2"
                 and r["metric"] in ("Nabf", "Qabf", "FMI", "VIF", "SSIM", "MS_SSIM", "SCD")]
        quality = len(guard) == 7 and all(r["n"] >= 8 and r["favorable_gain"] >= 0 for r in guard)
        opposite = "sar" if source == "optical" else "optical"
        other = [r for r in comparisons if r["cohort"] == "image" and r["source"] == source
                 and r["sign"] == "plus" and r["reference"] == "e2" and r["metric"] == opposite+"_transfer"]
        retained = len(other) == 1 and other[0]["n"] >= 8 and other[0]["delta"] >= -core.RULES["source_transfer_tolerance"]
        manual = [r for r in comparisons if r["cohort"] == source+"_missing" and r["source"] == source
                  and r["sign"] == "plus" and r["reference"] in ("e2", source+"_native_plus", source+"_uniform_plus")
                  and r["metric"] in ("signed_response", "continuity")]
        enough_manual = len(manual) == 6 and all(r["n"] >= core.RULES["min_manual_scenes"] for r in manual)
        manual_gain = enough_manual and all(r["favorable_gain"] >= 0 for r in manual) and any(
            r["ci_low"] > 0 for r in manual if r["metric"] == "signed_response")
        negative = [r for r in comparisons if r["cohort"] == "disconnected" and r["source"] == source
                    and r["sign"] == "plus" and r["reference"] == "e2" and r["metric"] == "longest_run_fraction"]
        safe = (len(negative) == 1 and negative[0]["n"] >= 3 and negative[0]["line_count"] >= 12
                and negative[0]["delta"] <= core.RULES["negative_path_tolerance"])
        changes = [r["changed_fraction"] for r in image_rows if r["method"] == source+"_predictive_plus"]
        complete = len(changes) == len(expected_scenes) and all(r["n"] == len(expected_scenes) for r in own)
        excited = len(changes) >= 8 and float(np.median(changes)) >= core.RULES["min_changed_pixel_fraction"]
        supported = prediction[source]["status"] == "supported" and all((complete, local, quality, retained, manual_gain, safe, excited))
        evidence[source] = dict(prediction=prediction[source]["status"], beats_amplitude_and_relation_controls=local,
            image_quality_non_decreasing_means=quality, opposite_source_retained=retained,
            enough_manual_scenes=enough_manual, manual_signal=manual_gain, negative_path_guard=safe,
            all_requested_scenes_excited=complete,
            sufficient_quantized_effect=excited,
            status="local_frozen_evidence_requires_training" if supported else "not_established_for_this_frozen_operator")
    return dict(sources=evidence, scope="Not a trained RCT model; failure cannot rule out nonlinear probes or other operators")


def export_and_register_images(folder, metadata, rows, write=True, register=True):
    methods = sorted({r["method"] for r in rows})
    by = {(r["scene"], r["method"]): r for r in rows}
    scenes = sorted({r["scene"] for r in rows if r["method"] == "e2"})
    token = digest(dict(metadata=metadata, result_dir=str(folder)))[:16].upper()
    entries = []
    for method in methods:
        output = folder/"metrics"/method
        if write:
            output.mkdir(parents=True, exist_ok=False)
        raw = [dict(image=s, evaluated=(s, method) in by,
                    **{m: by.get((s, method), {}).get(m) for m in METRICS}) for s in scenes]
        means, stds, counts = {}, {}, {}
        for metric in METRICS:
            common = [s for s in scenes if all((s, m) in by and by[(s, m)].get(metric) is not None
                       and math.isfinite(by[(s, m)][metric]) for m in methods)]
            vals = [by[(s, method)][metric] for s in common]
            counts[metric] = dict(common_count=len(common), excluded_scenes=[s for s in scenes if s not in common])
            means[metric] = float(np.mean(vals)) if vals else None
            stds[metric] = float(np.std(vals)) if vals else None
        if write:
            write_csv(output/"metrics.csv", raw+[dict(image="mean", **means), dict(image="std", **stds)])
            write_json(output/"common_samples.json", counts)
        finite = {k: v for k, v in means.items() if v is not None}
        entries.append(dict(id="RCT-TRANSFER-"+token+"-"+method.upper(), method="Frozen E2 "+method,
            task="fusion", status="diagnostic_fixed_subset", comparable_group="RCT-TRANSFER-"+token,
            dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(scenes),
            checkpoint=ORIGINAL_LOSS_PILOT.final.as_posix(), checkpoint_sha256=metadata["checkpoint_sha256"],
            split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(), split_manifest_sha256=metadata["split_manifest_sha256"],
            probe_artifact_sha256=metadata["probe_sha256"], metric_profile=DEFAULT_PROFILE,
            output_policy="clamp_round_uint8", result_dir=folder.as_posix(),
            metadata_path=(folder/"run_metadata.json").as_posix(), metrics=finite,
            metric_source=dict(type="csv_row", path=(output/"metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={k: k for k in finite}),
            notes="All methods use per-metric common finite scenes. Same frozen E2; matched final-image MAE, both signs. "
                  "This is not a trained RCT model or test1000."))
    if register:
        from run_e2_matched_no_pac import register_many
        register_many(entries)
    return entries


def write_report(out, summary, pcomp, icomp):
    def number(value):
        return "—" if value is None else "%.6g"%value
    lines = ["# RCT结构预测增量与冻结成图验证", "",
             "本报告由既定协议自动生成。没有训练RCT融合网络；E2权重全程冻结。", "",
             "预测验证场景：%d；请求成图场景：%d；成功校准条件：%d。"%(
                 summary["prediction_scene_count"], summary["rendered_scene_count"], summary["calibrated_conditions"]), "",
             "预测主读出使用排除固定人工16图后的验证场景；仍来自已有pilot，不是地理独立测试集。", ""]
    for source, title in (("optical", "光学"), ("sar", "SAR")):
        lines.extend(["## "+title+"结构预测", "",
                      "自动准入：`"+summary["prediction"][source]["status"]+"`。", "",
                      "| 对照 | 共同场景 | 相对有利变化 | 改善场景 | 有利差95%区间 |",
                      "|---|---:|---:|---:|---|"])
        for row in pcomp:
            if row["source"] == source and row["cohort"] == "unaudited":
                lines.append("| %s | %d | %s | %d | [%s, %s] |"%(
                    row["reference"], row["n"], number(row["relative_gain"]), row["improved"],
                    number(row["ci_low"]), number(row["ci_high"])))
        lines.append("")
    if "sources" in summary["transfer"]:
        for source, title in (("optical", "光学"), ("sar", "SAR")):
            status = summary["transfer"]["sources"][source]
            lines.extend(["## "+title+"条件的成图传递", "", "自动判断：`"+status["status"]+"`。", "",
                          "只以预定正向插值作为主判断；负向条件完整报告，不事后挑有利符号。", "",
                          "| 检查 | 结果 |", "|---|---|"])
            for key, value in status.items():
                if key != "status":
                    lines.append("| %s | %s |"%(key, value))
            lines.extend(["", "| 参照 | 共同场景 | 本源候选传递差 | 有利差95%区间 |",
                          "|---|---:|---:|---|"])
            for row in icomp:
                if (row["source"] == source and row["cohort"] == "image" and row["sign"] == "plus"
                        and row["metric"] == source+"_transfer"):
                    lines.append("| %s | %d | %s | [%s, %s] |"%(
                        row["reference"], row["n"], number(row["delta"]), number(row["ci_low"]), number(row["ci_high"])))
            lines.append("")
    lines.extend(["## 解释与结果入口", "",
        "- 预测头为固定HOG的线性Ridge，额外输入已在训练集上对本源上下文做无标签残差化。",
        "- 成图算子为带条件的源Detail插值；正结果只是局部冻结可行性，负结果不否定所有非线性预测器或RCT算子。",
        "- 同一场景全部来源、条件和符号匹配最终浮点图像MAE；不使用质量指标挑强度。",
        "- 梯度候选、走廊连续性和人工小样本均不是完整结构真值。请复核previews与gray中的实际图像。",
        "- 14项逐图指标、均值/标准差及共同样本数在metrics；非有限值保留且按共同交集统计。",
        "- 完整配对统计见[prediction_comparisons.csv](prediction_comparisons.csv)和[transfer_comparisons.csv](transfer_comparisons.csv)。",
        "- 效应不足的场景："+(", ".join(summary["insufficient_effect_scenes"]) or "无")+"。这些场景不能解释为恒等安全回退。", ""])
    with (out/"REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate(args, manifest, labels, audited, annotation_hashes):
    fit_metadata = read_json(args.fit_dir/"run_metadata.json")
    success = verify_completion(args.fit_dir)
    if args.seed != fit_metadata["config"]["seed"]:
        raise ValueError("Evaluation seed must match the fixed fitting protocol")
    if (fit_metadata["source_hashes"] != source_hashes()
            or fit_metadata["split_manifest_sha256"] != sha256_file(ROOT/ORIGINAL_LOSS_PILOT.manifest)
            or fit_metadata["checkpoint_sha256"] != sha256_file(ROOT/ORIGINAL_LOSS_PILOT.final)
            or success["probe_sha256"] != sha256_file(args.fit_dir/"probe.npz")):
        raise ValueError("Fitted probe provenance differs; preserve it and use a new protocol directory")
    # Verify saved train/calibration inputs without refitting anything.
    if fit_metadata["inputs_sha256"] != inputs_hashes(manifest, fit_metadata["config"]["train_scenes"]):
        raise ValueError("Predictor fit inputs changed")
    scenes = manifest["splits"]["val"]
    render_scenes = [] if args.prediction_only else (scenes if args.render_all else [s for s in scenes if s in audited])
    metadata = dict(checkpoint_sha256=fit_metadata["checkpoint_sha256"],
                    split_manifest_sha256=fit_metadata["split_manifest_sha256"],
                    probe_sha256=success["probe_sha256"], fit_metadata_sha256=sha256_file(args.fit_dir/"run_metadata.json"),
                    source_hashes=source_hashes(), prediction_scenes=scenes, render_scenes=render_scenes,
                    audited_scenes=audited, annotations_sha256=annotation_hashes,
                    inputs_sha256=inputs_hashes(manifest, scenes), device=args.device,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8",
                    max_interpolation=args.max_interpolation, common_target_fraction=.9,
                    mismatch_seed=args.seed+2,
                    final_image_mae_tolerance=.02, rules=core.RULES,
                    operator="fused_Detail + alpha * condition * (source_Detail - fused_Detail)",
                    primary_sign="plus", both_signs_reported=True, weights_updated=False)
    out = args.output_dir
    if out.exists():
        if not (out/"SUCCESS.json").is_file() or read_json(out/"run_metadata.json") != clean_json(metadata):
            raise FileExistsError("Preserve incomplete/different results; use a new --output-dir")
        verify_completion(out)
        print("Reusing completed diagnostic: "+str(out), flush=True)
        summary = read_json(out/"summary.json")
        with (out/"per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
            cached_rows = list(csv.DictReader(handle))
        for row in cached_rows:
            for key in (*METRICS, *TRANSFERS, "changed_fraction"):
                row[key] = float(row[key]) if row.get(key) else None
        if cached_rows:
            export_and_register_images(out, metadata, cached_rows, write=False)
        register_summary(out, metadata, summary, "predictive_transfer_audit", "val", len(scenes),
                         {k: summary[k] for k in ("prediction_scene_count", "rendered_scene_count", "calibrated_conditions")})
        return summary
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"run_metadata.json", metadata)
    heads = load_heads(args.fit_dir/"probe.npz")
    normalizers = read_json(args.fit_dir/"summary.json")["normalizers"]
    partners = core.mismatched_partners(scenes, args.seed+2)
    prediction_rows, image_rows, line_rows, effects, qualifications = [], [], [], [], []
    model = frozen_checkpoint(args.device)[0] if render_scenes else None
    if model is not None and any(p.requires_grad for p in model.parameters()):
        raise AssertionError("All E2 parameters must remain frozen")
    with torch.inference_mode():
        for index, scene in enumerate(scenes):
            current = describe_scene(manifest, scene)
            partner = describe_scene(manifest, partners[str(scene)])
            maps = {}
            for source in core.SOURCES:
                features, target, mask, mag = designs(current, partner, source)
                errors = core.prediction_errors(heads[source], features, target)
                count = int(mask.sum())
                prediction_rows.append(dict(scene=scene, source=source, common_positions=count,
                    **{m: float(e[mask].mean()) if count else math.nan for m, e in errors.items()}))
                maps[source] = core.relationship_maps(errors, mag, mask, **{
                    k: normalizers[source][k] for k in ("magnitude_scale", "gain_scale")})
            if scene in render_scenes:
                a, b, c, d = render_scene(model, manifest, scene, maps, labels, args, out)
                image_rows.extend(a); line_rows.extend(b); effects.extend(c); qualifications.extend(d)
            print("validate %d/%d %s%s"%(index+1, len(scenes), scene,
                                       " + frozen transfer" if scene in render_scenes else ""), flush=True)
    pcomp = prediction_comparisons(prediction_rows, set(audited))
    admission = core.prediction_admission(pcomp)
    icomp = comparison_rows(image_rows, (*METRICS, *TRANSFERS)) if image_rows else []
    for cohort in sorted({r["cohort"] for r in line_rows}):
        selected = [r for r in line_rows if r["cohort"] == cohort]
        metrics = ("path_mean", "path_coverage", "longest_run_fraction") if cohort == "disconnected" else (
            "signed_response", "continuity", "longest_run_fraction", "polarity_reversal")
        icomp.extend(comparison_rows(selected, metrics, cohort))
    decision = transfer_decision(admission, icomp, image_rows) if image_rows else dict(status="prediction_only")
    if (metadata["inputs_sha256"] != inputs_hashes(manifest, scenes)
            or metadata["source_hashes"] != source_hashes()
            or metadata["checkpoint_sha256"] != sha256_file(ROOT/ORIGINAL_LOSS_PILOT.final)):
        raise ValueError("Data, code, or checkpoint changed during evaluation")
    summary = dict(prediction_scene_count=len(scenes), rendered_scene_count=len(render_scenes),
                   calibrated_conditions=sum(e["status"] == "matched" for e in effects),
                   insufficient_effect_scenes=sorted({e["scene"] for e in effects if e["status"] != "matched"}),
                   prediction=admission, transfer=decision,
                   caveats=["Fixed HOG linear probe, no nonlinear MR or trained RCT tested",
                            "Observed source targets contribute to inference-time relation maps; not clean-data supervision",
                            "Manual probes and gradient candidates are small-sample proxies, not dense structure truth",
                            "Validation shares the existing pilot; unaudited does not mean geographically independent",
                            "Negative-sign arms are diagnostic, never selected as a replacement primary after validation"])
    write_csv(out/"prediction_per_image.csv", prediction_rows)
    write_csv(out/"prediction_comparisons.csv", pcomp)
    write_csv(out/"per_image.csv", image_rows)
    write_csv(out/"per_probe.csv", line_rows)
    write_csv(out/"matched_effects.csv", effects)
    write_csv(out/"source_probe_qualification.csv", qualifications)
    write_csv(out/"transfer_comparisons.csv", icomp)
    write_json(out/"summary.json", summary)
    write_json(out/"decision.json", decision)
    write_report(out, summary, pcomp, icomp)
    # Persist the completed numerical artifacts before potentially slow global
    # index validation. A later rerun can register cached artifacts without
    # fitting or inference if unrelated historical registry checks failed.
    entries = export_and_register_images(out, metadata, image_rows, write=True, register=False) if image_rows else []
    write_json(out/"SUCCESS.json", completion_manifest(out, summary_sha256=sha256_file(out/"summary.json")))
    if entries:
        from run_e2_matched_no_pac import register_many
        register_many(entries)
    register_summary(out, metadata, summary, "predictive_transfer_audit", "val", len(scenes),
                     {k: summary[k] for k in ("prediction_scene_count", "rendered_scene_count", "calibrated_conditions")})
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def smoke(args, manifest):
    if args.scene_id not in manifest["splits"]["val"]:
        raise ValueError("Smoke scene must belong to locked val64")
    optical, sar = read_pair(scene_files(manifest, args.scene_id))
    o, s = (to_tensor(x, args.device)[..., :96, :96] for x in (optical, sar))
    model, _ = frozen_checkpoint(args.device)
    with torch.inference_mode():
        encoded = model.encode_pair(o, s)
        base, detail = model.fuse_features(*encoded)
        original = model.reconstruct(o, base, detail)
        torch.testing.assert_close(original, model(o, s), rtol=1e-5, atol=1e-6)
        zero = core.interpolation_direction(encoded[1], detail, torch.zeros_like(o))
        torch.testing.assert_close(model.reconstruct(o, base, detail+zero), original, rtol=0, atol=0)
        desc, _, valid = core.structure_descriptor(o.cpu())
        assert desc.shape == (24, 24, 9) and np.isfinite(desc).all() and valid.any()
    print("Single-scene 96x96 smoke passed: exact zero-condition identity and real E2 composition; no probe fitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate", "all", "smoke"))
    parser.add_argument("--fit-dir", type=Path, default=FIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--train-scenes", type=int, default=128)
    parser.add_argument("--rows-per-scene", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-interpolation", type=float, default=.04)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--scene-id", default="211")
    args = parser.parse_args()
    if not 24 <= args.train_scenes <= 1024 or args.train_scenes % 4 or args.rows_per_scene < 32:
        parser.error("--train-scenes must be a multiple of four in [24,1024]; rows >=32")
    if not 0 < args.max_interpolation <= .1 or not 0 < args.ridge <= 1:
        parser.error("Invalid fixed interpolation budget or ridge")
    if args.prediction_only and args.render_all:
        parser.error("Prediction-only cannot request rendering")
    args.fit_dir, args.output_dir = args.fit_dir.resolve(), args.output_dir.resolve()
    if (args.command in ("fit", "all") and args.fit_dir == FIT_DIR
            and (args.train_scenes, args.rows_per_scene, args.ridge, args.seed) != (128, 512, .01, 42)):
        parser.error("Changed fitting settings require a new --fit-dir")
    if (args.command in ("evaluate", "all") and args.output_dir == RESULT_DIR
            and (args.prediction_only or args.render_all or args.max_interpolation != .04)):
        parser.error("Changed evaluation settings require a new --output-dir")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, labels, audited, hashes = manifest_and_annotations()
    if args.command == "smoke":
        smoke(args, manifest)
        return
    if args.command in ("fit", "all"):
        fit(args, manifest)
    if args.command in ("evaluate", "all"):
        evaluate(args, manifest, labels, audited, hashes)


if __name__ == "__main__":
    main()
