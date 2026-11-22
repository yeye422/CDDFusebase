import argparse
import copy
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import e2_source_objective as objective
from e2_base_anchor import OpticalBaseAnchoredE2
from diagnose_e2_source_objective import bounded_image_step, project_detail
import run_e2_source_objective as runner
from test_OS import load_variant_models, forward_variant
from train_direct_fusion import DirectFusionLosses


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.loss = objective.SourceContrastLosses()

    def edge(self):
        image = torch.full((1, 1, 21, 23), 0.2)
        image[..., 11:] = 0.8
        return image

    def test_low_target_takes_max_before_pool_and_odd_padding(self):
        grid = (torch.arange(9)[:, None] + torch.arange(11)[None, :]) % 2
        optical = grid.float()[None, None]
        sar = 1 - optical
        fused = torch.full_like(optical, 0.5, requires_grad=True)
        self.loss.fusion(optical, sar, fused, torch.ones_like(sar), 10)
        self.assertAlmostEqual(float(self.loss.stats["low_intensity"]), 0.5)
        self.assertEqual(objective.lowpass(fused).shape, (1, 1, 5, 6))
        self.assertFalse(torch.equal(objective.lowpass(torch.maximum(optical, sar)),
                                     torch.maximum(objective.lowpass(optical), objective.lowpass(sar))))

    def test_optical_and_sar_private_edges_do_not_require_cross_match(self):
        edge, flat = self.edge(), torch.full_like(self.edge(), 0.2)
        for optical, sar, expected in ((edge, flat, "optical"), (flat, edge, "sar")):
            evidence = self.loss.evidence(optical, sar, torch.ones_like(sar))
            self.assertGreater(float(evidence[expected].sum()), 0)
            other = "sar" if expected == "optical" else "optical"
            self.assertLess(float(evidence[other].sum()), 1e-7)
            self.assertLess(float(evidence["common"].sum()), 1e-7)

    def test_opposite_polarity_is_not_two_incompatible_targets(self):
        optical = self.edge()
        evidence = self.loss.evidence(optical, 1 - optical, torch.ones_like(optical))
        self.assertGreater(float(evidence["ambiguous"].sum()), 0)
        self.assertLess(float(evidence["common"].sum()), 1e-7)
        self.assertLess(float(evidence["optical"].max()), 1e-6)
        self.assertLess(float(evidence["sar"].max()), 1e-6)

    def test_evidence_detached_and_invalid_sar_cannot_change_loss(self):
        optical = self.edge().requires_grad_(True)
        sar = self.edge().flip(-1).requires_grad_(True)
        valid = torch.ones_like(optical)
        valid[..., :8] = 0
        evidence = self.loss.evidence(optical, sar, valid)
        self.assertTrue(all(not value.requires_grad for value in evidence.values()))
        self.assertFalse(evidence["sar_valid"][..., :11].any())
        fused = torch.rand_like(optical, requires_grad=True)
        before = self.loss.fusion(optical, sar, fused, valid, 10)[0]
        corrupted = sar.detach().clone()
        corrupted[valid == 0] = float("nan")
        after = self.loss.fusion(optical, corrupted, fused, valid, 10)[0]
        torch.testing.assert_close(before, after)
        after.backward()
        self.assertIsNone(optical.grad)
        self.assertIsNone(sar.grad)
        self.assertTrue(torch.isfinite(fused.grad).all())

    def test_group_reduction_is_per_image_and_empty_groups_are_finite(self):
        optical, sar, fused = (torch.rand(2, 1, 17, 19) for _ in range(3))
        valid = torch.ones_like(sar)
        valid[1] = 0
        together = self.loss.fusion(optical, sar, fused, valid, 10)[0]
        separate = torch.stack([self.loss.fusion(optical[i:i+1], sar[i:i+1], fused[i:i+1], valid[i:i+1], 10)[0]
                                for i in range(2)]).mean()
        torch.testing.assert_close(together, separate)
        flat = torch.full((1, 1, 9, 9), 0.4, requires_grad=True)
        value = self.loss.fusion(flat.detach(), flat.detach(), flat, torch.zeros_like(flat), 10)[0]
        self.assertEqual(float(value), 0)
        value.backward()
        self.assertTrue(torch.isfinite(flat.grad).all())

    def test_new_total_matches_explicit_weights_not_legacy_gradient_weight(self):
        optical, sar, fused = (torch.rand(1, 1, 15, 17) for _ in range(3))
        total = self.loss.fusion(optical, sar, fused, torch.ones_like(sar), 10)[0]
        s = self.loss.stats
        expected = (objective.SPEC["low_weight"] * s["low_intensity"]
                    + objective.SPEC["preserve_weight"] * s["preserve"]
                    + objective.SPEC["uncertain_weight"] * s["uncertain"])
        torch.testing.assert_close(total.detach(), expected)
        self.assertEqual(set(s), set(objective.LOSS_STATS))

    def test_low_only_has_no_response_to_aligned_zero_mean_pixel_detail(self):
        image = torch.zeros(1, 1, 10, 12)
        image[..., 0::2, :] = 1
        image[..., 1::2, :] = -1
        self.assertEqual(float(objective.lowpass(image).abs().max()), 0)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_objective_only_keeps_identical_architecture_and_seed(self):
        torch.manual_seed(42)
        original = OpticalBaseAnchoredE2().eval()
        torch.manual_seed(42)
        candidate = objective.SourceObjectiveE2().eval()
        self.assertEqual(original.state_dict().keys(), candidate.state_dict().keys())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
        optical, sar = torch.rand(1, 1, 17, 19), torch.rand(1, 1, 17, 19)
        with torch.no_grad():
            torch.testing.assert_close(original(optical, sar), candidate(optical, sar), rtol=0, atol=0)

    def test_readout_keeps_backbone_initialization_and_post_construction_rng(self):
        torch.manual_seed(42)
        original = OpticalBaseAnchoredE2()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = objective.SourceReadoutE2()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name in ("encoder", "lwpd", "base_fusion"):
            for key, value in getattr(original, name).state_dict().items():
                torch.testing.assert_close(value, getattr(model, name).state_dict()[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_amp_source_readout_and_objective_gradients(self):
        torch.manual_seed(42)
        model = objective.SourceReadoutE2().cuda()
        losses = objective.SourceContrastLosses().cuda()
        args = argparse.Namespace(gradient_weight=10, reconstruction_weight=0.1)
        o, s = torch.rand(1, 1, 17, 19, device="cuda"), torch.rand(1, 1, 17, 19, device="cuda")
        with torch.autocast("cuda"):
            value, _ = objective.forward_losses(model, losses, o, s, torch.ones_like(s), args)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_readout_reconstruction_and_fusion_have_finite_source_head_gradients(self):
        for original_loss in (False, True):
            torch.manual_seed(42)
            model = objective.SourceReadoutE2()
            losses = DirectFusionLosses() if original_loss else objective.SourceContrastLosses()
            args = argparse.Namespace(gradient_weight=10, reconstruction_weight=0.1)
            optical, sar = torch.rand(1, 1, 17, 19), torch.rand(1, 1, 17, 19)
            for forward in (objective.reconstruction_forward, objective.forward_losses):
                model.zero_grad(set_to_none=True)
                value, _ = forward(model, losses, optical, sar, torch.ones_like(sar), args)
                value.backward()
                self.assertTrue(torch.isfinite(value))
                for head in (model.decoder.optical_head, model.decoder.sar_head):
                    self.assertTrue(torch.isfinite(head[-1].weight.grad).all())
                    self.assertGreater(float(head[-1].weight.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            with torch.no_grad():
                image = model(optical, sar)
            self.assertEqual(image.shape, optical.shape)
            self.assertTrue(((image >= 0) & (image <= 1)).all())
            self.assertIsInstance(model.detail_fusion, nn.Identity)
            self.assertEqual(set(model.decoder.stats), set(objective.READOUT_STATS))

    def test_checkpoint_uses_normal_inference_path(self):
        for model_type in (objective.SourceObjectiveE2, objective.SourceReadoutE2):
            model = model_type().eval()
            optical, sar = torch.rand(1, 1, 17, 19), torch.rand(1, 1, 17, 19)
            with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
                path = Path(folder) / "candidate.pth"
                torch.save({"args": {"variant": model.variant}, "model": model.state_dict()}, path)
                restored, variant = load_variant_models(path, "cpu")
                with torch.no_grad():
                    actual, _ = forward_variant(restored, variant, optical, sar)
                    torch.testing.assert_close(actual, model(optical, sar), rtol=0, atol=0)

    def test_readout_refuses_ambiguous_single_detail_input(self):
        model = objective.SourceReadoutE2()
        detail = torch.rand(1, 64, 7, 9)
        with self.assertRaises(ValueError):
            model.decoder(None, detail, detail)

    def test_optimizer_and_stage_schedule_cover_both_readout_heads(self):
        model = objective.SourceReadoutE2()
        args = argparse.Namespace(decoder_lr=1e-4, warmup_fusion_lr=1e-4, warmup_lwpd_lr=1e-4,
                                  warmup_wavelet_lr=1e-5, weight_decay=1e-4,
                                  encoder_early_lr=1e-4, encoder_tail_lr=1e-4,
                                  joint_fusion_lr=1e-4, joint_lwpd_lr=1e-4, joint_wavelet_lr=1e-5)
        optimizer = runner.protocol.trainer.make_optimizer(model, args)
        grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        for head in (model.decoder.optical_head, model.decoder.sar_head):
            self.assertTrue(all(id(p) in grouped for p in head.parameters()))
        runner.protocol.configure(model, optimizer, args, reconstruction=True)
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertEqual(next(g for g in optimizer.param_groups if g["group_name"] == "base_fusion")["lr"], 0)
        runner.protocol.configure(model, optimizer, args, reconstruction=False)
        self.assertTrue(all(p.requires_grad for p in model.base_fusion.parameters()))


class ProtocolTests(unittest.TestCase):
    def test_missing_cuda_fails_before_any_training_lock_or_data_access(self):
        with patch.object(runner.torch.cuda, "is_available", return_value=False), \
                patch.object(runner.torch.version, "cuda", None), \
                patch.object(runner, "source_manifest") as manifest, \
                patch.object(runner, "lock_training") as lock, \
                patch.object(runner.protocol, "train") as train:
            with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                runner.train("objective")
        manifest.assert_not_called()
        lock.assert_not_called()
        train.assert_not_called()

    def test_only_unstarted_locks_can_be_refreshed_and_previous_record_is_kept(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            exp = argparse.Namespace(run_dir=Path(folder) / "seed42")
            path = Path(folder) / "workflow_protocol.json"
            before, after = {"sources": "old"}, {"sources": "new"}
            runner.write_json(path, before)
            with patch.object(runner, "signature", return_value=after):
                with self.assertRaises(ValueError):
                    runner.lock_training(exp, create=False)
                runner.lock_training(exp, create=True)
                self.assertEqual(runner.read_json(path), after)
                archives = list(Path(folder).glob("workflow_protocol.unstarted-*.json"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(runner.read_json(archives[0]), before)
                exp.run_dir.mkdir()
                runner.write_json(path, before)
                with self.assertRaises(ValueError):
                    runner.lock_training(exp, create=True)
                self.assertEqual(runner.read_json(path), before)

    def test_relative_cli_file_path_can_build_and_lock_signature_after_chdir(self):
        """Reproduce Python 3.8 script loading without starting real training."""
        cwd = Path.cwd()
        try:
            for directory, relative_script in (
                (runner.ROOT, Path("run_e2_source_objective.py")),
                (runner.ROOT.parent, Path(runner.ROOT.name) / "run_e2_source_objective.py"),
            ):
                os.chdir(directory)
                namespace = runpy.run_path(str(relative_script), run_name="source_objective_path_regression")
                self.assertFalse(Path(namespace["__file__"]).is_absolute())
                self.assertEqual(namespace["SCRIPT_PATH"], runner.SCRIPT_PATH)
                self.assertEqual(namespace["MODULE_PATH"], runner.MODULE_PATH)
                # main() changes cwd before lock_training(), as in the report.
                os.chdir(runner.ROOT)
                globals_ = namespace["signature"].__globals__
                with patch.dict(globals_, {
                    "training_args": lambda exp: argparse.Namespace(variant="synthetic"),
                    "sha256_file": lambda path: str(path),
                }), tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
                    exp = argparse.Namespace(run_dir=Path(folder) / "seed42")
                    signature = namespace["signature"](exp)
                    self.assertIn("run_e2_source_objective.py", signature["sources"])
                    self.assertIn("e2_source_objective.py", signature["sources"])
                    self.assertTrue(all(Path(path).is_absolute() for path in signature["sources"].values()))
                    lock = namespace["lock_training"](exp, create=True)
                    self.assertEqual(runner.read_json(lock), signature)
                    self.assertEqual(namespace["lock_training"](exp), lock)
                    self.assertFalse(exp.run_dir.exists())
        finally:
            os.chdir(cwd)

    def test_saved_artifacts_reject_changed_raw_metric_rows(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            root = Path(folder)
            metrics = root / "metrics.csv"
            metrics.write_text("image,Qabf\n211,0.8\n", encoding="utf-8")
            runner.write_artifacts(root)
            runner.verify_artifacts(root)
            metrics.write_text("image,Qabf\n211,0.9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_artifacts(root)

    def test_common_finite_mask_preserves_raw_values_and_reports_count(self):
        import csv
        import numpy as np
        from run_e2_aspr_pilot import summarize
        n = len(runner.METRICS)
        column = runner.METRICS.index("MS_SSIM")
        before = [np.ones(n), np.ones(n)]
        after = [np.ones(n) * 2, np.ones(n) * 2]
        before[1][column] = np.nan
        rows = {"e2": [(v, [], []) for v in before], "candidate": [(v, [], []) for v in after]}
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            root = Path(folder)
            (root / "e2").mkdir()
            (root / "candidate").mkdir()
            summary = summarize(root, ["211", "546"], rows, (("e2", "candidate"),),
                                diagnostic_fields=(), source_metrics=())
            self.assertEqual(summary["candidate"]["MS_SSIM"], 2)
            with (root / "paired_comparison.csv").open(encoding="utf-8", newline="") as handle:
                record = next(v for v in csv.DictReader(handle) if v["metric"] == "MS_SSIM")
            self.assertEqual(record["common_count"], "1")
            with (root / "e2/metrics.csv").open(encoding="utf-8", newline="") as handle:
                raw = list(csv.DictReader(handle))
            self.assertEqual(raw[1]["MS_SSIM"], "nan")

    def test_default_cli_only_dispatches_one_scene_probe(self):
        with patch.object(sys, "argv", ["run_e2_source_objective.py"]), \
                patch("diagnose_e2_source_objective.probe") as probe, patch.object(runner, "train") as train:
            runner.main()
        probe.assert_called_once_with("211", None, project=False)
        train.assert_not_called()

    def test_four_cells_are_explicit_and_baseline_has_no_training_arm(self):
        self.assertEqual(runner.ARMS, ("objective", "readout", "joint"))
        self.assertEqual(runner.experiment("objective").variant, objective.SourceObjectiveE2.VARIANT)
        for arm in ("readout", "joint"):
            self.assertEqual(runner.experiment(arm).variant, objective.SourceReadoutE2.VARIANT)
        self.assertEqual(len({runner.experiment(arm).run_dir for arm in runner.ARMS}), 3)
        with self.assertRaises(ValueError):
            runner.experiment("e2")

    def test_training_hook_dispatch_and_restoration_without_training(self):
        manifest = {"splits": {"val": ["211"]}}
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as folder:
            lock = Path(folder) / "lock.json"
            lock.write_text('{"settings": {}}', encoding="utf-8")
            original = runner.protocol.trainer.DirectFusionLosses
            observed = []

            def fake_train(exp):
                observed.append(runner.protocol.trainer.DirectFusionLosses)
                self.assertIs(runner.protocol.trainer.forward_losses, objective.forward_losses)
                self.assertIs(runner.protocol.reconstruction_forward, objective.reconstruction_forward)

            with patch.object(runner, "source_manifest", return_value=manifest), \
                    patch.object(runner, "ensure_training_cuda"), \
                    patch.object(runner, "archived_baseline"), patch.object(runner, "lock_training", return_value=lock), \
                    patch.object(runner.protocol, "train", side_effect=fake_train):
                runner.train("objective")
                runner.train("readout")
            self.assertEqual(observed, [objective.SourceContrastLosses, DirectFusionLosses])
            self.assertIs(runner.protocol.trainer.DirectFusionLosses, original)

    def test_training_args_serialize_identically_to_checkpoint_json(self):
        fake = argparse.Namespace()
        with patch.object(runner, "BASE_TRAINING_ARGS", side_effect=lambda exp: copy.deepcopy(fake)):
            args = runner.training_args(runner.experiment("joint"))
        import json
        self.assertEqual(vars(args), json.loads(json.dumps(vars(args))))
        self.assertIsNot(args.source_objective, objective.SPEC)

    def test_matched_pixel_step_obeys_output_budget(self):
        torch.manual_seed(42)
        image = torch.rand(1, 1, 9, 11)
        gradient = torch.randn_like(image)
        target = 0.1 / 255
        result = bounded_image_step(image, gradient, target)
        self.assertAlmostEqual(float((result - image).abs().mean()), target, places=7)
        self.assertLessEqual(float((result - image).abs().max()), 4 / 255 + 1e-7)
        with self.assertRaises(ValueError):
            bounded_image_step(image, torch.zeros_like(image), target)

    def test_latent_fit_changes_no_model_weights_and_respects_detail_bound(self):
        class ToyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

            def forward(self, image, base, detail):
                return torch.sigmoid(self.weight * detail), detail

        model = argparse.Namespace(decoder=ToyDecoder())
        detail = torch.ones(1, 1, 7, 9)
        target = torch.sigmoid(detail + 0.02)
        before = model.decoder.weight.clone()
        image, stats = project_detail(model, torch.zeros_like(detail), detail, target)
        self.assertLess(stats["fitted_target_mse"], stats["initial_target_mse"])
        self.assertLessEqual(stats["detail_delta_rms"], stats["detail_rms_bound"] + 1e-7)
        torch.testing.assert_close(before, model.decoder.weight)
        self.assertIsNone(model.decoder.weight.grad)
        self.assertTrue(torch.isfinite(image).all())


if __name__ == "__main__":
    unittest.main()
