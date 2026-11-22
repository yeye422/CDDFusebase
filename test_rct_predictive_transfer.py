"""Synthetic tests only: no dataset fitting, no checkpoint or batch inference."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import rct_predictive_transfer as core


class StructuralPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(11)

    def descriptors(self):
        values = [self.rng.normal(size=(32, 32, 9)) for _ in range(3)]
        mask = np.ones((32, 32), bool)
        return values, mask

    def test_fixed_hog_flat_finite_and_invalid_boundaries(self):
        x = torch.ones(1, 1, 96, 96)*.5
        descriptor, magnitude, valid = core.structure_descriptor(x)
        self.assertEqual(descriptor.shape, (24, 24, 9))
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertLess(float(np.abs(descriptor).max()), 1e-3)
        self.assertLess(float(magnitude.max()), 1e-7)
        self.assertFalse(valid[0, 0])
        self.assertTrue(valid[12, 12])
        self.assertEqual(core.RAW_SUPPORT_WIDTH, 12)

    def test_no_target_center_in_self_context(self):
        (own, other, wrong), mask = self.descriptors()
        first, target, _ = core.prediction_design(own, other, mask, mask, wrong, mask)
        changed = own.copy()
        changed[16, 16] += 100
        second, modified, _ = core.prediction_design(changed, other, mask, mask, wrong, mask)
        for method in core.PREDICTORS:
            np.testing.assert_array_equal(first[method][16, 16], second[method][16, 16])
        self.assertFalse(np.array_equal(target[16, 16], modified[16, 16]))

    def test_raw_center_does_not_leak_through_descriptor_context(self):
        raw = torch.from_numpy(self.rng.random((1, 1, 128, 128)).astype(np.float32))
        a, _, valid = core.structure_descriptor(raw)
        modified = raw.clone()
        modified[..., 64:68, 64:68] = 1-modified[..., 64:68, 64:68]
        b, _, _ = core.structure_descriptor(modified)
        x, _, _ = core.prediction_design(a, a, valid, valid, a, valid)
        y, _, _ = core.prediction_design(b, a, valid, valid, a, valid)
        for method in core.PREDICTORS:
            np.testing.assert_allclose(x[method][16, 16], y[method][16, 16], atol=1e-10)
        self.assertGreater(float(np.abs(a[16, 16]-b[16, 16]).max()), 1e-4)

    def test_shapes_equal_and_shared_validity_excludes_context(self):
        (own, other, wrong), mask = self.descriptors()
        invalid = mask.copy()
        invalid[16, 19] = False
        features, _, support = core.prediction_design(own, other, invalid, mask, wrong, mask)
        self.assertEqual({v.shape for v in features.values()}, {(32, 32, 81)})
        self.assertFalse(support[16, 16])
        self.assertFalse(support[0, 0])
        np.testing.assert_array_equal(features["self"][..., -9:], 0)

    def test_streamed_ridge_matches_whole_fit(self):
        x = self.rng.normal(size=(200, 7))
        y = x@self.rng.normal(size=(7, 3))+.05*self.rng.normal(size=(200, 3))
        whole, streamed = core.RidgeMoments(7, 3), core.RidgeMoments(7, 3)
        whole.add(x, y)
        for a, b in ((0, 41), (41, 122), (122, 200)):
            streamed.add(x[a:b], y[a:b])
        a, b = whole.finish(.01, 5), streamed.finish(.01, 5)
        np.testing.assert_allclose(core.predict(a, x), core.predict(b, x), atol=1e-12)

    def test_duplicate_context_is_not_new_information(self):
        c = self.rng.normal(size=(300, 5))
        y = c[:, :2]+.1*self.rng.normal(size=(300, 2))
        control = np.column_stack((c, np.zeros((300, 2))))
        duplicated = np.column_stack((c, c[:, :2]))
        a, b = core.RidgeMoments(7, 2), core.RidgeMoments(7, 2)
        a.add(control, y); b.add(duplicated, y)
        ha, hb = a.finish(.1, 5), b.finish(.1, 5)
        self.assertEqual(int(hb["extra_residual_rank"]), 0)
        np.testing.assert_allclose(core.predict(ha, control), core.predict(hb, duplicated), atol=1e-10)

    def test_added_independent_predictive_feature_can_help(self):
        x = self.rng.normal(size=(300, 6))
        y = x[:, 5:6]
        stats = core.RidgeMoments(6, 1)
        stats.add(x, y)
        head = stats.finish(.001, 5)
        self.assertLess(np.mean((core.predict(head, x)-y)**2), .001)
        self.assertEqual(int(head["extra_residual_rank"]), 1)

    def test_conditions_keep_uncertain_zero_without_losing_original_path(self):
        errors = {name: np.ones((8, 8)) for name in ("self", "cross", "rolled", "swapped")}
        maps = core.relationship_maps(errors, np.ones((8, 8)), np.ones((8, 8), bool), 1., 1.)
        self.assertEqual(float(maps["predictive"].sum()), 0.)
        self.assertTrue(np.all(maps["uniform"] == 1))
        source, fused = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        delta = core.interpolation_direction(source, fused, torch.zeros(1, 1, 8, 8))
        torch.testing.assert_close(fused+delta, fused, rtol=0, atol=0)

    def test_shuffled_condition_preserves_common_support_histogram(self):
        mask = np.ones((8, 8), bool); mask[0, :] = False
        errors = dict(self=np.ones((8, 8)), cross=self.rng.random((8, 8)),
                      rolled=np.ones((8, 8)), swapped=np.ones((8, 8)))
        maps = core.relationship_maps(errors, np.ones((8, 8)), mask, 1., 1.)
        np.testing.assert_array_equal(np.sort(maps["predictive"][mask]), np.sort(maps["shuffled"][mask]))
        self.assertTrue(np.all(maps["shuffled"][~mask] == 0))

    def test_mismatch_pairs_are_derangements_and_fold_local(self):
        scenes = [str(i) for i in range(12)]
        partners = core.mismatched_partners(scenes, 42)
        self.assertEqual(set(partners.values()), set(scenes))
        self.assertTrue(all(k != v for k, v in partners.items()))
        self.assertEqual(partners, core.mismatched_partners(scenes, 42))

    def test_effect_calibration_equalizes_different_gains(self):
        nominal = {"a": .04*2, "b": .04*10, "c": .04*4}
        target = core.common_effect_target(nominal)
        for gain in (2., 10., 4.):
            alpha, effect, error = core.calibrate_effect(lambda x: gain*x, target)
            self.assertLessEqual(alpha, .04)
            self.assertLessEqual(error, .02)
            self.assertAlmostEqual(effect, target, places=8)

    def test_nonlinear_calibration_and_degenerate_effect(self):
        alpha, effect, error = core.calibrate_effect(lambda x: x*x, .0004)
        self.assertLess(error, .02)
        self.assertLessEqual(alpha, .04)
        self.assertIsNone(core.common_effect_target({"a": 0., "b": .3}))
        with self.assertRaises(ValueError):
            core.calibrate_effect(lambda x: x, .1, upper=.04)

    def test_statistics_keep_finite_scene_pairs_and_metric_direction(self):
        s = core.paired_stats([1., np.nan, 2.], [2., 3., 4.], lower_is_better=True)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["improved"], 2)
        self.assertEqual(s["delta"], -1.5)
        self.assertGreater(s["ci_low"], 0)

    def test_admission_needs_both_context_and_pairing_controls(self):
        rows = [dict(source=s, cohort="unaudited", reference=m, n=20, relative_gain=.1,
                     median_gain=.1, ci_low=.01, improved=18)
                for s in core.SOURCES for m in ("self", "self_wide", "rolled", "swapped")]
        self.assertEqual(core.prediction_admission(rows)["optical"]["status"], "supported")
        rows[3]["ci_low"] = -.01
        self.assertEqual(core.prediction_admission(rows)["optical"]["status"], "not_established")
        self.assertEqual(core.prediction_admission(rows)["sar"]["status"], "supported")


class AuditIntegrationTests(unittest.TestCase):
    def test_real_rendering_path_with_synthetic_frozen_decoder(self):
        import diagnose_rct_predictive_transfer as runner
        rng = np.random.default_rng(19)
        optical, sar = (rng.integers(10, 240, (64, 64), dtype=np.uint8) for _ in range(2))

        class Frozen(torch.nn.Module):
            def encode_pair(self, o, s):
                return o*0, o.repeat(1, 2, 1, 1), s*0, s.repeat(1, 2, 1, 1)

            def fuse_features(self, bo, do, bs, ds):
                return bo, (do+ds)/2

            def reconstruct(self, o, base, detail):
                return (detail.mean(1, keepdim=True)*2).sigmoid()

        model = Frozen().eval().requires_grad_(False)
        o, s = (runner.to_tensor(x, "cpu") for x in (optical, sar))
        base, detail = model.fuse_features(*model.encode_pair(o, s))
        archive = runner.image_u8(model.reconstruct(o, base, detail))
        maps = {source: {c: np.ones((16, 16)) if c == "uniform" else rng.uniform(.2, 1, (16, 16))
                         for c in core.CONDITIONS} for source in core.SOURCES}

        def cheap_readout(model, o, s, optical, sar, base, detail, fused, masks, method, out):
            (out/"gray"/method).mkdir(parents=True, exist_ok=True)
            return runner.image_u8(fused), dict(**{m: .5 for m in runner.METRICS},
                                               optical_transfer=.5, sar_transfer=.5)

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner, "read_pair", return_value=(optical, sar)), \
                    patch.object(runner, "scene_files", return_value=("o", "s")), \
                    patch.object(runner, "prepare_lines", return_value=([], [])), \
                    patch.object(runner, "image_readout", side_effect=cheap_readout), \
                    patch.object(runner.cv2, "imread", return_value=archive), torch.inference_mode():
                rows, probes, effects, qualification = runner.render_scene(
                    model, {}, "synthetic", maps, [], SimpleNamespace(device="cpu", max_interpolation=.04), Path(temporary))
            self.assertEqual(len(rows), 25)
            self.assertEqual(len(effects), 24)
            self.assertTrue(all(r["relative_matching_error"] <= .02 and r["alpha"] <= .04 for r in effects))
            self.assertEqual(len({r["target_mae"] for r in effects}), 1)
            self.assertTrue((Path(temporary)/"previews/synthetic.png").is_file())

    def test_export_keeps_all_scenes_and_common_finite_means(self):
        import diagnose_rct_predictive_transfer as runner
        rows = [dict(scene=s, method=m, **{k: float(i+1) for k in runner.METRICS})
                for i, s in enumerate(("a", "b", "c")) for m in ("e2", "optical_predictive_plus")]
        rows = [r for r in rows if (r["scene"], r["method"]) != ("b", "optical_predictive_plus")]
        metadata = dict(checkpoint_sha256="checkpoint", split_manifest_sha256="manifest", probe_sha256="probe")
        with tempfile.TemporaryDirectory() as temporary:
            entries = runner.export_and_register_images(Path(temporary), metadata, rows, register=False)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len({e["comparable_group"] for e in entries}), 1)
            self.assertTrue(all(e["sample_count"] == 3 and e["metrics"]["FMI"] == 2. for e in entries))
            counts = json.loads((Path(temporary)/"metrics/e2/common_samples.json").read_text())
            self.assertEqual(counts["FMI"]["common_count"], 2)
            content = (Path(temporary)/"metrics/optical_predictive_plus/metrics.csv").read_text()
            self.assertIn("b,False", content)

    def test_artifact_reuse_rejects_tampering(self):
        from diagnose_rct_predictive_transfer import completion_manifest, verify_completion, write_json
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/"data.txt").write_text("original", encoding="utf-8")
            write_json(folder/"SUCCESS.json", completion_manifest(folder))
            verify_completion(folder)
            (folder/"data.txt").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_completion(folder)

    def test_common_samples_across_all_methods(self):
        from diagnose_rct_predictive_transfer import comparison_rows
        rows = []
        methods = ["e2"]+["optical_"+c+"_plus" for c in core.CONDITIONS]
        for scene in ("a", "b", "c"):
            for method in methods:
                value = np.nan if scene == "b" and method == "optical_uniform_plus" else .5
                rows.append(dict(scene=scene, method=method, FMI=value))
        result = comparison_rows(rows, ("FMI",))
        self.assertTrue(result)
        self.assertTrue(all(row["n"] == 2 and row["scene_ids"] == "a c" for row in result))

    def test_no_transfer_evidence_is_never_a_pass(self):
        from diagnose_rct_predictive_transfer import transfer_decision
        prediction = {s: dict(status="supported") for s in core.SOURCES}
        rows = [dict(scene=str(i), method="e2", changed_fraction=0.) for i in range(16)]
        result = transfer_decision(prediction, [], rows)
        self.assertTrue(all(v["status"] != "local_frozen_evidence_requires_training"
                            for v in result["sources"].values()))

    def test_probe_roundtrip_no_pickle(self):
        from diagnose_rct_predictive_transfer import save_heads, load_heads
        rng = np.random.default_rng(3)
        stats = core.RidgeMoments(5, 2)
        x = rng.normal(size=(50, 5)); y = rng.normal(size=(50, 2))
        stats.add(x, y)
        head = stats.finish(.01, 3)
        heads = {s: {m: head for m in core.PREDICTORS} for s in core.SOURCES}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"probe.npz"
            save_heads(path, heads)
            restored = load_heads(path)
            np.testing.assert_array_equal(core.predict(head, x), core.predict(restored["sar"]["cross"], x))


if __name__ == "__main__":
    unittest.main()
