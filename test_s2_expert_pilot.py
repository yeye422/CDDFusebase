"""CPU-only synthetic tests for S2 operator comparison (no user data required)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from run_s2_expert_pilot import DEFAULT_PROFILE, MODES, FusionEvaluator, digest, paired_summary, pilot_manifest, read_evaluation
from s2_expert_models import (
    HeterogeneousExpertFusion, count_params, make_detail_fusion,
)


class TestS2Pilot(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.o = torch.randn(1, 64, 32, 32)
        self.s = torch.randn(1, 64, 32, 32)
        self.b = torch.randn(1, 64, 32, 32)

    def test_matched_capacity(self):
        dense = make_detail_fusion('dense')
        top2 = make_detail_fusion('top2')
        conv = make_detail_fusion('conv')
        self.assertEqual(count_params(dense), count_params(top2))
        self.assertLess(abs(count_params(conv) / count_params(dense) - 1), .10)

    def test_dense_top2_share_initial_weights(self):
        torch.manual_seed(123)
        dense = make_detail_fusion('dense')
        torch.manual_seed(123)
        sparse = make_detail_fusion('top2')
        for key in dense.state_dict():
            torch.testing.assert_close(dense.state_dict()[key], sparse.state_dict()[key])

    def test_top2_is_per_region_and_normalized(self):
        m = HeterogeneousExpertFusion('top2', tile=16)
        weights = m.routing(self.o, self.s, self.b)
        self.assertEqual(tuple(weights.shape), (1, 3, 2, 2))
        torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]), rtol=1e-5, atol=1e-6)
        self.assertTrue((weights == 0).sum(1).eq(1).all().item())
        dense = m.routing(self.o, self.s, self.b, mode='dense')
        self.assertTrue((dense > 0).all().item())

    def test_forward_backward_finite(self):
        for mode in ('conv', 'dense', 'top2'):
            model = make_detail_fusion(mode)
            out = model(self.o, self.s, self.b)
            self.assertEqual(tuple(out.shape), tuple(self.o.shape))
            self.assertTrue(torch.isfinite(out).all().item())
            out.square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))

    def test_validation_rejects_bad_feature_shapes(self):
        m = HeterogeneousExpertFusion('dense')
        with self.assertRaises(ValueError):
            m(self.o, self.s[:, :32], self.b)
        with self.assertRaises(ValueError):
            m(self.o[:, :, :-1], self.s[:, :, :-1], self.b[:, :, :-1])


class TestS2EvaluationProtocol(unittest.TestCase):
    def test_pilot_uses_midpoints_and_has_no_test_samples(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / 'source.json'
            source.write_text(json.dumps(dict(
                optical_dir='optical', sar_dir='sar',
                splits=dict(train=['101', '102', '103'], val=list(map(str, range(10))), test=['201']),
            )), encoding='utf-8')
            args = SimpleNamespace(manifest=source, seed=42, train_limit=2, val_limit=4)
            manifest = pilot_manifest(args)
            self.assertEqual(manifest['splits']['val'], ['1', '3', '6', '8'])
            self.assertEqual(manifest['splits']['test'], [])
            self.assertEqual(len(manifest['splits']['train']), 2)
            self.assertEqual(manifest['source_manifest_sha256'], digest(source))

    def test_pairing_excludes_nonfinite_values_for_all_modes(self):
        scenes = ['4', '44']
        rows = {mode: [dict.fromkeys(FusionEvaluator.METRIC_NAMES, score) for score in (0.4, 0.8)]
                for mode in MODES}
        rows['top2'][0]['MS_SSIM'] = 0.6
        rows['top2'][1]['MS_SSIM'] = float('nan')
        rows['dense'][0]['Nabf'] = 0.2
        rows['dense'][0]['VIF'] = float('inf')
        rows['top2'][1]['VIF'] = float('nan')
        metrics = {row['metric']: row for row in paired_summary(rows, scenes)}
        ms = metrics['MS_SSIM']
        self.assertEqual(ms['common_count'], 1)
        self.assertEqual(ms['excluded_scene_ids'], '44')
        self.assertEqual(ms['e2_ft'], 0.4)
        self.assertEqual(ms['e2_ft_finite_count'], 2)
        self.assertEqual(ms['top2_finite_count'], 1)
        self.assertAlmostEqual(ms['top2_vs_dense'], 0.2)
        self.assertGreater(metrics['Nabf']['dense_vs_conv'], 0)
        self.assertEqual(metrics['VIF']['common_count'], 0)
        self.assertIsNone(metrics['VIF']['top2'])
        self.assertEqual(len(metrics), 14)

    def test_rejects_legacy_profile_and_misaligned_rows(self):
        with TemporaryDirectory() as temp:
            result = Path(temp)
            checkpoint, manifest = result / 'model.pth', result / 'manifest.json'
            checkpoint.write_bytes(b'synthetic checkpoint')
            manifest.write_text('{}', encoding='utf-8')
            scenes = ['4', '44']
            metadata = dict(checkpoint_sha256=digest(checkpoint), split_manifest_sha256=digest(manifest),
                            selected_scene_ids=scenes, split='val', offset=0, evaluated_pairs=2,
                            metric_profile='paper', output_range_policy='clamp', variant='s2_top2_v1', device='cpu')
            metadata_path = result / 'run_metadata.json'
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protocol'):
                read_evaluation(result, checkpoint, manifest, scenes, 'top2')
            metadata['metric_profile'] = DEFAULT_PROFILE
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            (result / 'metrics.csv').write_text('image,FMI\n44,0.5\n4,0.5\nmean,0.5\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'scenes'):
                read_evaluation(result, checkpoint, manifest, scenes, 'top2')


if __name__ == '__main__':
    unittest.main()
