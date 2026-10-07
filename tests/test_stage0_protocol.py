import unittest
from types import SimpleNamespace

import torch

from test_OS import normalize_fused_tensor, resolve_variant
from reliability_modules import (
    DemandDetailInjection,
    OpticalDetailRefinement,
    ResidualDetailInjection,
)
from train_OS import (
    OpticalSARLosses,
    build_models,
    configure_trainability,
    forward_loss,
    normalized_pcgrad_backward,
    reference_normalized_smooth_minimax,
)
from utils.dataset_os import digest_splits, validate_manifest_splits


class Stage0LossTests(unittest.TestCase):
    def setUp(self):
        self.losses = OpticalSARLosses()

    def test_stage2_gradient_weight_is_effective(self):
        optical = torch.zeros(1, 1, 16, 16)
        sar = torch.zeros_like(optical)
        generated = torch.zeros_like(optical)
        generated[:, :, 8, 8] = 1.0
        valid = torch.ones_like(optical)

        total_zero, intensity, gradient = self.losses.fusion_old(
            optical, sar, generated, valid, 0.0
        )
        total_two, _, _ = self.losses.fusion_old(
            optical, sar, generated, valid, 2.0
        )
        self.assertTrue(torch.allclose(total_zero, intensity))
        self.assertTrue(torch.allclose(total_two, intensity + 2.0 * gradient))

    def test_common_loss_ignores_invalid_sar_value(self):
        optical = torch.zeros(1, 1, 16, 16)
        generated = torch.zeros_like(optical)
        valid = torch.ones_like(optical)
        valid[:, :, 8, 8] = 0.0
        sar_clean = torch.zeros_like(optical)
        sar_invalid_spike = sar_clean.clone()
        sar_invalid_spike[:, :, 8, 8] = 1.0

        clean = self.losses.fusion_common(
            optical, sar_clean, generated, valid, 1.0, 1.0, 99.0
        )[0]
        spiked = self.losses.fusion_common(
            optical, sar_invalid_spike, generated, valid, 1.0, 1.0, 99.0
        )[0]
        self.assertTrue(torch.allclose(clean, spiked, atol=1e-7))

    def test_common_structure_uses_one_gradient_scale(self):
        ramp = torch.linspace(0.0, 1.0, 16).view(1, 1, 1, 16)
        optical = ramp.expand(1, 1, 16, 16).contiguous()
        sar = torch.zeros_like(optical)
        valid = torch.ones_like(optical)
        result = self.losses.fusion_common(
            optical, sar, optical, valid, 1.0, 1.0, 99.0
        )
        structure_loss = result[-1]
        self.assertLess(float(structure_loss), 1e-7)

    def test_common_loss_has_finite_backward(self):
        torch.manual_seed(7)
        optical = torch.rand(2, 1, 16, 16)
        sar = torch.rand_like(optical)
        generated = torch.rand_like(optical, requires_grad=True)
        valid = torch.ones_like(optical)
        total = self.losses.fusion_common(
            optical, sar, generated, valid, 1.0, 1.0, 99.0
        )[0]
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.all(torch.isfinite(generated.grad)))

    def test_percentile_clipping_caps_valid_values(self):
        values = torch.tensor([[[[0.0, 1.0, 2.0, 3.0, 100.0]]]])
        valid = torch.ones_like(values)
        clipped = self.losses.clip_valid_percentile(values, valid, 50.0)
        self.assertLessEqual(float(clipped.max()), 2.0)

    def test_v2_structure_loss_has_gradient_at_optical_identity(self):
        optical = torch.zeros(1, 1, 32, 32)
        sar = torch.zeros_like(optical)
        sar[:, :, :, 16:] = 1.0
        generated = optical.clone().requires_grad_(True)
        valid = torch.ones_like(optical)
        result = self.losses.fusion_common_v2(
            optical,
            sar,
            generated,
            valid,
            optical_weight=1.0,
            structure_weight=1.0,
            structure_sigma=1.0,
            structure_percentile=99.0,
            optical_lowpass_sigma=2.0,
            sar_edge_percentile=85.0,
            sar_edge_margin=0.001,
        )
        total, _, _, _, structure, edge_fraction, _ = result
        total.backward()
        self.assertGreater(float(structure), 0.0)
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)
        self.assertGreater(float(edge_fraction), 0.0)
        self.assertLessEqual(float(edge_fraction), 0.16)

    def test_v3_optical_constraint_stops_inside_tolerance(self):
        torch.manual_seed(11)
        optical = torch.rand(1, 1, 32, 32)
        sar = torch.zeros_like(optical)
        sar[:, :, :, 16:] = 1.0
        generated = optical.clone().requires_grad_(True)
        valid = torch.ones_like(optical)
        result = self.losses.fusion_common_v3(
            optical,
            sar,
            generated,
            valid,
            optical_weight=0.2,
            optical_gradient_weight=0.25,
            structure_weight=1.0,
            structure_sigma=1.0,
            structure_percentile=99.0,
            optical_lowpass_sigma=2.0,
            optical_l1_tolerance=0.02,
            optical_dssim_tolerance=0.03,
            sar_edge_percentile=85.0,
            sar_edge_margin=0.001,
        )
        (
            total,
            _,
            _,
            _,
            optical_constraint,
            optical_gradient,
            sar_gradient,
            structure,
            edge_fraction,
            _,
            _,
            _,
        ) = result
        self.assertEqual(float(optical_constraint), 0.0)
        self.assertLess(float(optical_gradient), 1e-7)
        self.assertGreater(float(sar_gradient), 0.0)
        self.assertTrue(
            torch.allclose(
                total,
                0.2 * optical_constraint
                + 0.25 * optical_gradient
                + sar_gradient,
            )
        )
        self.assertTrue(torch.allclose(structure, 0.25 * optical_gradient + sar_gradient))
        self.assertGreater(float(edge_fraction), 0.0)
        total.backward()
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)

    def test_v4_is_exactly_two_term_common_loss(self):
        torch.manual_seed(19)
        optical = torch.rand(2, 1, 32, 32)
        sar = torch.rand_like(optical)
        generated = torch.rand_like(optical, requires_grad=True)
        valid = torch.ones_like(optical)
        result = self.losses.fusion_common_v4(
            optical,
            sar,
            generated,
            valid,
            structure_weight=3.0,
            structure_sigma=1.0,
            structure_percentile=99.0,
            optical_lowpass_sigma=2.0,
            charbonnier_epsilon=1e-3,
        )
        total, optical_loss, optical_l1, optical_ssim, structure, _ = result
        self.assertTrue(torch.allclose(optical_loss, optical_l1 + optical_ssim))
        self.assertTrue(torch.allclose(total, optical_loss + 3.0 * structure))
        total.backward()
        self.assertTrue(torch.all(torch.isfinite(generated.grad)))

    def test_v4_zero_difference_has_zero_charbonnier_loss(self):
        image = torch.zeros(1, 1, 32, 32)
        valid = torch.ones_like(image)
        result = self.losses.fusion_common_v4(
            image,
            image,
            image,
            valid,
            structure_weight=1.0,
            structure_sigma=1.0,
            structure_percentile=99.0,
            optical_lowpass_sigma=2.0,
            charbonnier_epsilon=1e-3,
        )
        _, optical_loss, _, _, structure, _ = result
        self.assertLess(float(optical_loss), 1e-7)
        self.assertLess(float(structure), 1e-7)

    def test_fusion_only_freezes_identity_path(self):
        models = build_models(torch.device("cpu"))
        configure_trainability(models, "fusion_only")
        self.assertFalse(any(p.requires_grad for p in models["encoder"].parameters()))
        self.assertFalse(any(p.requires_grad for p in models["decoder"].parameters()))
        self.assertTrue(all(p.requires_grad for p in models["base_fuse"].parameters()))
        self.assertTrue(all(p.requires_grad for p in models["detail_fuse"].parameters()))

    def test_a1_injector_is_identity_at_initialization(self):
        torch.manual_seed(23)
        module = DemandDetailInjection(channels=64, gamma=0.5)
        optical = torch.rand(2, 64, 16, 16)
        sar = torch.rand_like(optical)
        fused, diagnostics = module(optical, sar)
        self.assertTrue(torch.allclose(fused, optical, atol=1e-7))
        self.assertAlmostEqual(float(diagnostics["demand_mean"]), 0.5, places=7)
        self.assertEqual(float(diagnostics["demand_std"]), 0.0)
        self.assertEqual(float(diagnostics["delta_norm"]), 0.0)
        self.assertEqual(float(diagnostics["injection_norm"]), 0.0)

    def test_a1_injector_supports_fixed_gate_ablation(self):
        module = DemandDetailInjection(channels=64, gamma=0.5)
        optical = torch.rand(1, 64, 8, 8)
        sar = torch.rand_like(optical)
        _, diagnostics = module(
            optical, sar, force_demand=0.2, return_maps=True
        )
        self.assertAlmostEqual(float(diagnostics["demand_mean"]), 0.2, places=6)
        self.assertAlmostEqual(
            float(diagnostics["learned_demand_mean"]), 0.5, places=6
        )
        self.assertEqual(tuple(diagnostics["demand_map"].shape), (1, 1, 8, 8))

    def test_a1_fusion_only_trains_base_and_injector(self):
        models = build_models(torch.device("cpu"), variant="demand_only")
        configure_trainability(models, "fusion_only")
        self.assertFalse(any(p.requires_grad for p in models["encoder"].parameters()))
        self.assertFalse(any(p.requires_grad for p in models["decoder"].parameters()))
        self.assertTrue(all(p.requires_grad for p in models["base_fuse"].parameters()))
        self.assertTrue(
            all(p.requires_grad for p in models["detail_injector"].parameters())
        )

    def test_a1prime_residual_only_has_no_gate_and_gamma_quarter(self):
        module = ResidualDetailInjection(channels=64, gamma=0.25)
        optical = torch.rand(2, 64, 16, 16)
        sar = torch.rand_like(optical)
        fused, diagnostics = module(optical, sar)
        self.assertFalse(hasattr(module, "demand_gate"))
        self.assertAlmostEqual(float(module.gamma), 0.25, places=7)
        self.assertTrue(torch.allclose(fused, optical, atol=1e-7))
        self.assertEqual(float(diagnostics["delta_norm"]), 0.0)
        self.assertEqual(float(diagnostics["injection_norm"]), 0.0)
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()), 32256
        )

    def test_a1prime_supports_inference_scale_ablation(self):
        module = ResidualDetailInjection(channels=64, gamma=0.25)
        with torch.no_grad():
            module.residual_candidate.output.bias.fill_(0.5)
        optical = torch.rand(1, 64, 8, 8)
        sar = torch.rand_like(optical)

        fused_zero, diagnostics_zero = module(
            optical, sar, force_scale=0.0
        )
        fused_half, diagnostics_half = module(
            optical, sar, force_scale=0.5
        )

        self.assertTrue(torch.allclose(fused_zero, optical, atol=1e-7))
        self.assertEqual(float(diagnostics_zero["injection_norm"]), 0.0)
        self.assertGreater(float(diagnostics_half["injection_norm"]), 0.0)
        expected = 0.5 * torch.tanh(torch.tensor(0.5))
        self.assertTrue(
            torch.allclose(fused_half - optical, expected, atol=1e-6)
        )

    def test_a1prime_fusion_only_trains_base_and_residual_injector(self):
        models = build_models(torch.device("cpu"), variant="residual_only")
        configure_trainability(models, "fusion_only")
        self.assertFalse(any(p.requires_grad for p in models["encoder"].parameters()))
        self.assertFalse(any(p.requires_grad for p in models["decoder"].parameters()))
        self.assertTrue(all(p.requires_grad for p in models["base_fuse"].parameters()))
        self.assertTrue(
            all(p.requires_grad for p in models["detail_injector"].parameters())
        )

    def test_frequency_loss_is_zero_for_identical_constant_sources(self):
        image = torch.full((1, 1, 32, 32), 0.25)
        valid = torch.ones_like(image)
        result = self.losses.fusion_frequency_decoupled(
            image,
            image,
            image,
            valid,
            optical_lowpass_sigma=2.0,
            highpass_sigma=1.0,
            midpass_sigma=2.0,
            optical_high_weight=1.0,
            structure_percentile=99.0,
            charbonnier_epsilon=1e-3,
        )
        total, optical, _, _, optical_high, sar_mid, _, _ = result
        self.assertLess(float(total), 1e-6)
        self.assertLess(float(optical), 1e-6)
        self.assertLess(float(optical_high), 1e-6)
        self.assertLess(float(sar_mid), 1e-6)

    def test_frequency_loss_separates_optical_high_and_sar_mid_terms(self):
        optical = torch.zeros(1, 1, 32, 32)
        sar = torch.zeros_like(optical)
        sar[:, :, :, 16:] = 1.0
        generated = optical.clone().requires_grad_(True)
        valid = torch.ones_like(optical)
        result = self.losses.fusion_frequency_decoupled(
            optical,
            sar,
            generated,
            valid,
            optical_lowpass_sigma=2.0,
            highpass_sigma=1.0,
            midpass_sigma=2.0,
            optical_high_weight=1.0,
            structure_percentile=99.0,
            charbonnier_epsilon=1e-3,
        )
        total, _, _, _, optical_high, sar_mid, _, _ = result
        self.assertLess(float(optical_high), 1e-7)
        self.assertGreater(float(sar_mid), 0.0)
        total.backward()
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)

    def test_selective_line_loss_is_zero_when_reliability_is_below_threshold(self):
        optical = torch.zeros(1, 1, 32, 32)
        sar = torch.zeros_like(optical)
        sar[:, :, :, 16:] = 1.0
        generated = optical.clone().requires_grad_(True)
        valid = torch.ones_like(optical)
        reliability = torch.zeros_like(optical)
        result = self.losses.fusion_frequency_selective_line(
            optical,
            sar,
            generated,
            valid,
            reliability,
            optical_lowpass_sigma=2.0,
            highpass_sigma=1.0,
            midpass_sigma=2.0,
            optical_high_weight=1.0,
            line_structure_weight=1.0,
            line_reliability_threshold=0.5,
            structure_percentile=99.0,
            charbonnier_epsilon=1e-3,
        )
        self.assertLess(float(result[6]), 1e-7)
        self.assertEqual(float(result[7]), 0.0)

    def test_selective_line_loss_uses_high_reliability_sar_edge(self):
        optical = torch.zeros(1, 1, 32, 32)
        sar = torch.zeros_like(optical)
        sar[:, :, :, 16:] = 1.0
        generated = optical.clone().requires_grad_(True)
        valid = torch.ones_like(optical)
        reliability = torch.ones_like(optical)
        result = self.losses.fusion_frequency_selective_line(
            optical,
            sar,
            generated,
            valid,
            reliability,
            optical_lowpass_sigma=2.0,
            highpass_sigma=1.0,
            midpass_sigma=2.0,
            optical_high_weight=1.0,
            line_structure_weight=1.0,
            line_reliability_threshold=0.5,
            structure_percentile=99.0,
            charbonnier_epsilon=1e-3,
        )
        total, _, _, _, _, _, line_loss = result[:7]
        self.assertGreater(float(line_loss), 0.0)
        total.backward()
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)

    def test_c0_c1_c2_have_expected_detail_paths(self):
        c0 = build_models(torch.device("cpu"), variant="frequency_matched_baseline")
        c1 = build_models(torch.device("cpu"), variant="base_optical_detail")
        c2 = build_models(torch.device("cpu"), variant="base_optical_detail_refine")
        self.assertIn("detail_fuse", c0)
        self.assertNotIn("detail_fuse", c1)
        self.assertNotIn("detail_refiner", c1)
        self.assertIsInstance(c2["detail_refiner"], OpticalDetailRefinement)

    def test_c2_refiner_is_identity_at_initialization(self):
        module = OpticalDetailRefinement(channels=64)
        optical = torch.rand(2, 64, 16, 16)
        self.assertTrue(torch.allclose(module(optical), optical, atol=1e-7))

    def test_c1_c2_fusion_only_trainability(self):
        c1 = build_models(torch.device("cpu"), variant="base_optical_detail")
        configure_trainability(c1, "fusion_only")
        self.assertTrue(all(p.requires_grad for p in c1["base_fuse"].parameters()))
        self.assertFalse(any(p.requires_grad for p in c1["encoder"].parameters()))
        self.assertFalse(any(p.requires_grad for p in c1["decoder"].parameters()))

        c2 = build_models(torch.device("cpu"), variant="base_optical_detail_refine")
        configure_trainability(c2, "fusion_only")
        self.assertTrue(all(p.requires_grad for p in c2["base_fuse"].parameters()))
        self.assertTrue(all(p.requires_grad for p in c2["detail_refiner"].parameters()))

    def test_c0_c1_c2_full_forward_losses_are_finite(self):
        torch.manual_seed(29)
        optical = torch.rand(1, 1, 16, 16)
        sar = torch.rand_like(optical)
        valid = torch.ones_like(optical)
        common = dict(
            optical_lowpass_sigma=2.0,
            frequency_highpass_sigma=1.0,
            frequency_midpass_sigma=2.0,
            optical_high_weight=1.0,
            structure_percentile=99.0,
            structure_charbonnier_epsilon=1e-3,
            decomp_weight=0.0,
        )
        for variant in (
            "frequency_matched_baseline",
            "frequency_minimax_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
        ):
            models = build_models(torch.device("cpu"), variant=variant)
            args = SimpleNamespace(variant=variant, **common)
            total, components = forward_loss(
                models,
                self.losses,
                optical,
                sar,
                valid,
                phase=2,
                args=args,
            )
            self.assertTrue(torch.isfinite(total))
            self.assertTrue(torch.isfinite(components["optical_high"]))
            self.assertTrue(torch.isfinite(components["sar_mid"]))
            self.assertTrue(torch.isfinite(components["output_sar_mae"]))

    def test_normalized_pcgrad_projects_conflicting_objectives(self):
        parameter = torch.tensor([1.0, 1.0], requires_grad=True)
        loss_app = parameter[0]
        loss_structure = -parameter[0] + parameter[1]
        stats = normalized_pcgrad_backward(
            loss_app, loss_structure, [parameter]
        )
        self.assertEqual(float(stats["gradient_conflict"]), 1.0)
        self.assertAlmostEqual(float(stats["gradient_app_norm"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(stats["gradient_str_norm"]), 2.0 ** 0.5, places=6
        )
        self.assertTrue(torch.all(torch.isfinite(parameter.grad)))
        self.assertGreater(float(parameter.grad[1]), 0.0)

    def test_reference_normalized_minimax_prioritizes_worse_relative_objective(self):
        optical = torch.tensor(0.8, requires_grad=True)
        structure = torch.tensor(1.2, requires_grad=True)
        total, relative_optical, relative_structure = (
            reference_normalized_smooth_minimax(
                optical,
                structure,
                reference_optical=1.0,
                reference_structure=1.0,
                temperature=0.1,
            )
        )
        total.backward()
        self.assertAlmostEqual(float(relative_optical), 0.8, places=6)
        self.assertAlmostEqual(float(relative_structure), 1.2, places=6)
        self.assertGreater(float(structure.grad), float(optical.grad))
        self.assertGreaterEqual(float(total), 1.2)

    def test_reference_normalized_minimax_rejects_invalid_reference(self):
        with self.assertRaises(ValueError):
            reference_normalized_smooth_minimax(
                torch.tensor(1.0),
                torch.tensor(1.0),
                reference_optical=0.0,
                reference_structure=1.0,
                temperature=0.1,
            )


class Stage0EvaluationTests(unittest.TestCase):
    def test_clamp_policy_uses_fixed_range(self):
        values = torch.tensor([[[[-1.0, 0.25, 2.0]]]])
        normalized = normalize_fused_tensor(values, "clamp")
        expected = torch.tensor([[[[0.0, 0.25, 1.0]]]])
        self.assertTrue(torch.equal(normalized, expected))

    def test_legacy_minmax_remains_available(self):
        values = torch.tensor([[[[2.0, 3.0, 4.0]]]])
        normalized = normalize_fused_tensor(values, "legacy_minmax")
        expected = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        self.assertTrue(torch.equal(normalized, expected))

    def test_a1_variant_is_detected_from_checkpoint(self):
        checkpoint = {"DemandDetailInjector": {}, "args": {"variant": "demand_only"}}
        self.assertEqual(resolve_variant(checkpoint, "auto"), "demand_only")
        with self.assertRaises(ValueError):
            resolve_variant(checkpoint, "original")

    def test_a1prime_variant_is_detected_from_checkpoint(self):
        checkpoint = {
            "ResidualDetailInjector": {},
            "args": {"variant": "residual_only"},
        }
        self.assertEqual(resolve_variant(checkpoint, "auto"), "residual_only")

    def test_frequency_variants_are_detected_from_checkpoint_metadata(self):
        for variant in (
            "frequency_matched_baseline",
            "frequency_minimax_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
            "fixed_line_roa",
            "fixed_line_phase_bridge",
        ):
            checkpoint = {"args": {"variant": variant}}
            self.assertEqual(resolve_variant(checkpoint, "auto"), variant)


class Stage0ManifestTests(unittest.TestCase):
    def test_manifest_digest_is_verified(self):
        splits = {"train": ["1"], "val": ["2"], "test": ["3"]}
        manifest = {"splits": splits, "splits_sha256": digest_splits(splits)}
        self.assertEqual(validate_manifest_splits(manifest), digest_splits(splits))

    def test_overlapping_splits_are_rejected(self):
        splits = {"train": ["1"], "val": ["1"], "test": ["3"]}
        manifest = {"splits": splits, "splits_sha256": digest_splits(splits)}
        with self.assertRaises(ValueError):
            validate_manifest_splits(manifest)

if __name__ == "__main__":
    unittest.main()
