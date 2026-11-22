"""Small synthetic checks: signal preservation, controls, gradients, and provenance."""

import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import cv2
import numpy as np

from direct_fusion import MultiScaleDetailFusion
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cross_scale import (CrossScaleConditioner, CrossScaleDetailFusion, CrossScaleE2,
                            SPEC, cross_scale_summaries)
import run_e2_cross_scale as workflow


class CrossScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(19)

    def test_relations_measure_cross_scale_agreement(self):
        x = torch.tensor([1., 0.]).reshape(1, 2, 1, 1)
        y = torch.tensor([0., 1.]).reshape_as(x)
        result = cross_scale_summaries([x, y, x], "relation")
        torch.testing.assert_close(result.flatten(), torch.tensor([0., 1., 0.]))
        ordinary = cross_scale_summaries([x, y, x], "wide")
        torch.testing.assert_close(ordinary.flatten(), torch.full((3,), .5))

    def test_haar_descriptor_does_not_cancel_signed_content(self):
        module = CrossScaleConditioner()
        zero = torch.zeros(1, 64, 5, 7)
        detail = module.haar.inverse((zero, torch.ones_like(zero), zero, zero))
        self.assertFalse(torch.nn.functional.avg_pool2d(detail, 2).any())
        descriptor = module.describe(detail)
        self.assertEqual(tuple(descriptor.shape), (1, 8, 5, 7))
        self.assertGreater(float(descriptor.abs().sum()), 0)

    def test_zero_initialization_exactly_preserves_original_head(self):
        for mode in ("relation", "wide", "shuffled"):
            keep = MultiScaleDetailFusion()
            module = CrossScaleDetailFusion(keep, mode)
            tensors = [torch.randn(2, 64, 15, 17) for _ in range(3)]
            expected = keep(*tensors)
            for condition in (None, "aligned", "shifted", "disabled"):
                torch.testing.assert_close(module(*tensors, condition=condition), expected, rtol=0, atol=0)

    def test_shift_preserves_condition_distribution_and_bound(self):
        module = CrossScaleConditioner()
        torch.nn.init.normal_(module.gate[-1].weight, std=.5)
        detail = torch.randn(1, 64, 15, 17)
        aligned = module(detail, (15, 17), "aligned")
        shifted = module(detail, (15, 17), "shifted")
        torch.testing.assert_close(shifted, aligned.roll((7, 8), (-2, -1)), rtol=0, atol=0)
        torch.testing.assert_close(shifted.flatten().sort().values, aligned.flatten().sort().values)
        self.assertGreater(float((aligned - shifted).abs().sum()), 0)
        self.assertLessEqual(float(aligned.abs().max()), SPEC["max_modulation"] + 1e-7)
        self.assertFalse(module(detail, (15, 17), "disabled").any())

    def test_original_values_remain_in_the_aggregation_slots(self):
        keep = MultiScaleDetailFusion()
        module = CrossScaleDetailFusion(keep)
        optical, sar, base = [torch.randn(1, 64, 9, 11) for _ in range(3)]
        captured = []
        hook = keep.output.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            with patch.object(module.conditioner, "forward",
                              side_effect=lambda detail, size, alignment: detail.new_full((1, 32, *size), .05)):
                module(optical, sar, base)
        finally:
            hook.remove()
        torch.testing.assert_close(captured[0][:, :32], keep.optical_projection(optical) * 1.05)
        torch.testing.assert_close(captured[0][:, 32:64], keep.sar_projection(sar) * 1.05)

    def test_controls_have_identical_parameter_shapes_and_no_extra_experts(self):
        models = [CrossScaleDetailFusion(MultiScaleDetailFusion(), mode)
                  for mode in ("relation", "wide", "shuffled")]
        shapes = [{k: tuple(v.shape) for k, v in m.state_dict().items()} for m in models]
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[0], shapes[2])
        for model in models:
            self.assertEqual(sum(p.numel() for p in model.conditioner.parameters()), 2536)
            self.assertEqual(len(model.keep_fusion.scale_branches), 3)

    def test_condition_branch_receives_gradients_after_zero_start(self):
        for mode in ("relation", "wide"):
            module = CrossScaleDetailFusion(MultiScaleDetailFusion(), mode)
            tensors = [torch.randn(1, 64, 13, 15, requires_grad=True) for _ in range(3)]
            optimizer = torch.optim.SGD(module.parameters(), lr=.1)
            module(*tensors).square().mean().backward()
            last = module.conditioner.gate[-1].weight.grad
            self.assertTrue(torch.isfinite(last).all() and last.abs().sum() > 0)
            optimizer.step()
            optimizer.zero_grad()
            module(*tensors).square().mean().backward()
            for parameter in (module.conditioner.descriptor.weight,
                              module.conditioner.gate[0].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0)
            for tensor in tensors[:2]:
                self.assertTrue(torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0)

    def test_tiny_odd_and_zero_inputs_remain_finite(self):
        module = CrossScaleConditioner()
        for shape in ((1, 1), (1, 3), (3, 1), (7, 9)):
            delta = module(torch.zeros(1, 64, *shape), shape)
            self.assertEqual(tuple(delta.shape), (1, 32, *shape))
            self.assertTrue(torch.isfinite(delta).all())

    def test_full_model_initialization_reconstruction_and_reload(self):
        torch.manual_seed(42)
        baseline = OpticalBaseAnchoredE2().eval()
        torch.manual_seed(42)
        model = CrossScaleE2().eval()
        state = model.state_dict()
        for key, value in baseline.state_dict().items():
            target = key.replace("detail_fusion.", "detail_fusion.keep_fusion.", 1) if key.startswith("detail_fusion.") else key
            torch.testing.assert_close(value, state[target], rtol=0, atol=0)
        optical, sar = [torch.rand(1, 1, 17, 19) for _ in range(2)]
        torch.testing.assert_close(model(optical, sar), baseline(optical, sar), rtol=0, atol=0)
        encoded = model.encode_pair(optical, sar)
        base, detail = model.fuse_features(*encoded)
        loss = (model.reconstruct(optical, base, detail) - optical).square().mean()
        loss.backward()
        gradient = model.detail_fusion.conditioner.gate[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
        restored = CrossScaleE2().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(optical, sar), model(optical, sar))

    def test_actual_two_stage_loss_and_optimizer_include_conditioner(self):
        import run_e2_scratch_twostage as stages
        import train_lwpd as trainer
        pilot = workflow.experiment()
        args = stages.training_args(pilot)
        model = pilot.build_model()
        losses = trainer.DirectFusionLosses()
        optimizer = trainer.make_optimizer(model, args)
        detail_group = next(g for g in optimizer.param_groups if g["group_name"] == "detail_fusion")
        parameter_ids = [id(p) for p in detail_group["params"]]
        for p in model.detail_fusion.conditioner.parameters():
            self.assertEqual(parameter_ids.count(id(p)), 1)
        optical, sar = [torch.rand(1, 1, 21, 23) for _ in range(2)]
        valid = torch.ones_like(sar)
        stages.configure(model, optimizer, args, reconstruction=True)
        self.assertFalse(any(p.requires_grad for p in model.detail_fusion.parameters()))
        reconstruction, _ = stages.reconstruction_forward(model, losses, optical, sar, valid, args)
        self.assertTrue(torch.isfinite(reconstruction))
        reconstruction.backward()
        self.assertIsNone(model.detail_fusion.conditioner.gate[-1].weight.grad)
        optimizer.zero_grad()
        stages.configure(model, optimizer, args, reconstruction=False)
        self.assertTrue(all(p.requires_grad for p in model.detail_fusion.parameters()))
        total, _ = trainer.forward_losses(model, losses, optical, sar, valid, args)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        grad = model.detail_fusion.conditioner.gate[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)


class WorkflowTests(unittest.TestCase):
    def test_cuda_preflight_rejects_cpu_build_and_unavailable_driver(self):
        with patch.object(torch.version, "cuda", None), patch.object(torch, "__version__", "2.4.1+cpu"):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                workflow.require_cuda_runtime()
        with patch.object(torch.version, "cuda", "12.1"), patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no usable CUDA device"):
                workflow.require_cuda_runtime()
        with patch.object(torch.version, "cuda", "12.1"), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="test GPU"):
            self.assertEqual(workflow.require_cuda_runtime()["device"], "test GPU")

    def test_only_prestart_locks_can_be_refreshed_and_old_snapshot_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = SimpleNamespace(run_dir=Path(temp) / "seed42")
            path = Path(temp) / "workflow_protocol.json"
            workflow.write_json(path, dict(snapshot="before_cuda_fix"))
            old = path.read_bytes()
            workflow.training_lock(pilot, dict(snapshot="fixed"), create=True)
            backups = list(Path(temp).glob("workflow_protocol.prestart_*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), old)
            pilot.run_dir.mkdir()
            with self.assertRaises(ValueError):
                workflow.training_lock(pilot, dict(snapshot="another"), create=True)
            self.assertEqual(workflow.read_json(path), dict(snapshot="fixed"))

    def test_same_run_checkpoint_and_random_state_can_resume(self):
        import run_e2_scratch_twostage as stages
        args = stages.training_args(workflow.experiment())
        loader = SimpleNamespace(generator=torch.Generator().manual_seed(42))
        device = torch.device("cpu")
        saved_rng = stages.continuation_state(loader, device, provenance=None)
        expected_draws = (random.random(), float(np.random.rand()), float(torch.rand(())),
                          float(torch.rand((), generator=loader.generator)))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for name in ("split_manifest.json", "pilot_subset.json"):
                (run / name).write_text("{}\n", encoding="utf-8")
            config = dict(stages.config_values(args), total_epochs=stages.TOTAL_EPOCHS,
                          stage_learning_rates=stages.STAGE_LEARNING_RATES,
                          saved_split_manifest_sha256=workflow.sha256_file(run / "split_manifest.json"),
                          pilot_subset_sha256=workflow.sha256_file(run / "pilot_subset.json"))
            workflow.write_json(run / "config.json", config)
            torch.save(dict(epoch=0, stage="reconstruction", variant=args.variant,
                            args=vars(args), continuation=saved_rng), run / "epoch_001.pth")
            path, state, _ = stages.load_continuation(run, args)
            self.assertEqual(path.name, "epoch_001.pth")
            stages.restore_random_state(state["continuation"], loader, device, args.seed, 1)
            actual_draws = (random.random(), float(np.random.rand()), float(torch.rand(())),
                            float(torch.rand((), generator=loader.generator)))
            self.assertEqual(actual_draws, expected_draws)
            workflow.write_json(run / "config.json", dict(config, seed=43))
            with self.assertRaises(ValueError):
                stages.load_continuation(run, args)

    def test_unsafe_modes_and_paths_are_rejected(self):
        for tag in ("../escape", "D:/other", "", "a/b"):
            with self.assertRaises(ValueError):
                workflow.experiment(tag=tag)
        self.assertTrue(workflow.experiment().save_continuation)

    def test_budget_changes_and_weight_initialization_are_rejected(self):
        original = dict(variant="lwpd_e2_base_anchor_v1", output_dir="old", split_manifest="old.json",
                        init_checkpoint=None, initialization="random", gradient_weight=10., total_epochs=16)
        candidate = dict(original, variant="new", output_dir="new", split_manifest="new.json")
        workflow.check_training_settings(candidate, original)
        with self.assertRaises(ValueError):
            workflow.check_training_settings(dict(candidate, gradient_weight=20.), original)
        with self.assertRaises(ValueError):
            workflow.check_training_settings(dict(candidate, init_checkpoint="old.pth"),
                                             dict(original, init_checkpoint="old.pth"))

    def test_read_rows_rejects_duplicates_and_preserves_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "metrics.csv"
            p.write_text("image,x\na,nan\nb,2\nmean,1\n", encoding="utf-8")
            rows = workflow.read_rows(p, ("x",), ["a", "b"])
            self.assertTrue(math.isnan(rows[0][0]))
            with self.assertRaises(ValueError):
                workflow.read_rows(p, ("x",), ["b", "a"])
            p.write_text("image,x\na,1\na,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.read_rows(p, ("x",), ["a", "b"])

    def test_sealed_results_cannot_be_silently_overwritten_or_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            metadata = dict(scene_ids=["a"], protocol="fixed")
            workflow.write_json(folder / "run_metadata.json", metadata)
            (folder / "metrics.csv").write_text("image,x\na,1\n", encoding="utf-8")
            workflow.seal_result(folder)
            workflow.verify_result(folder, metadata)
            with self.assertRaises(ValueError):
                workflow.verify_result(folder, dict(metadata, protocol="changed"))
            (folder / "metrics.csv").write_text("image,x\na,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.verify_result(folder, metadata)

    def test_summary_uses_common_finite_scenes_for_each_metric(self):
        ids = ["a", "b", "c"]
        results = {}
        for key, missing in (("e2", 0), ("relation", 1), ("wide", 1)):
            rows = []
            for i in range(3):
                metrics = [float(i + 1)] * len(workflow.METRICS)
                metrics[workflow.METRICS.index("MS_SSIM")] = math.nan if i == missing else float(i + 1)
                rows.append((metrics, [math.nan] * len(workflow.ALL_SOURCE_METRICS),
                             [None] * len(workflow.ALL_DIAGNOSTICS)))
            results[key] = rows
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for key in results:
                (folder / key).mkdir()
            summaries = workflow.summarize(folder, ids, results, [("wide", "relation")],
                                           diagnostic_fields=workflow.ALL_DIAGNOSTICS,
                                           source_metrics=workflow.ALL_SOURCE_METRICS)
            self.assertEqual(summaries["relation"]["MS_SSIM"], 3.)
            excluded = json.loads((folder / "excluded_scene_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(excluded["MS_SSIM"], ["a", "b"])
            with (folder / "paired_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(r for r in csv.DictReader(handle) if r["metric"] == "MS_SSIM")
            self.assertEqual(row["common_count"], "1")

    def test_compare_reuses_outputs_and_registers_baseline_in_same_group(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                ids = ["a", "b"]
                for mode in ("relation", "wide"):
                    pilot = workflow.experiment(mode)
                    root = pilot.result
                    root.mkdir(parents=True)
                    methods = {}
                    for key in ("e2", *workflow.CONDITIONS):
                        folder = root / key
                        folder.mkdir()
                        with (folder / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(("image", *workflow.METRICS))
                            writer.writerows((scene, *([1.] * len(workflow.METRICS))) for scene in ids)
                        fields = (*workflow.ALL_SOURCE_METRICS, *workflow.ALL_DIAGNOSTICS)
                        with (folder / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(("image", *fields))
                            writer.writerows((scene, *([1.] * len(fields))) for scene in ids)
                        for name in ("gray", "rgb", "preview"):
                            (folder / name).mkdir()
                            for scene in ids:
                                self.assertTrue(cv2.imwrite(str(folder / name / f"{scene}.png"),
                                                            np.zeros((2, 2), dtype=np.uint8)))
                        methods[key] = dict(method=key, checkpoint=f"{mode}.pth", checkpoint_sha256=mode)
                    metadata = dict(scene_ids=ids, split_manifest_sha256="split", metric_profile=workflow.DEFAULT_PROFILE,
                                    output_policy=workflow.POLICY, archive={"checkpoint": "e2"}, spec=SPEC,
                                    implementation_sources={"model": "same"}, training_settings={"epochs": 16},
                                    runtime={"torch": "test"},
                                    device="cpu", input_hashes={"a": ["x", "y"], "b": ["z", "w"]},
                                    methods=methods, primary_condition="aligned")
                    workflow.write_json(root / "run_metadata.json", metadata)
                    workflow.seal_result(root)
                with patch.object(workflow, "register_many") as register:
                    workflow.compare("v1")
                    entries = register.call_args[0][0]
                    self.assertEqual(len(entries), 3)
                    self.assertEqual(len({e["comparable_group"] for e in entries}), 1)
                    self.assertTrue(any(e["id"].endswith("-e2") for e in entries))
                    # Reusing a complete result does not load or run any model.
                    with patch.object(CrossScaleE2, "__init__", side_effect=AssertionError("No inference")):
                        workflow.compare("v1")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
