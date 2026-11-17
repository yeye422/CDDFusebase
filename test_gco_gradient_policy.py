"""Synthetic CPU unit tests for matched-norm gradient intervention."""

import unittest

import torch

from gco_gradient_policy import (
    POLICIES, apply_delta, direction_metadata, dot,
    equal_displacement, generate_directions, vector_norm,
)


class GradientPolicyTests(unittest.TestCase):
    def test_conflict_projected_without_overwriting_source_grads(self):
        go, gs = [torch.tensor([1.0, 0.0])], [torch.tensor([-3.0, 4.0])]
        original_o = go[0].clone()
        original_s = gs[0].clone()
        directions = generate_directions(go, gs)
        self.assertEqual(set(directions), set(POLICIES))
        self.assertLess(dot(directions['raw'], go), 0)
        self.assertGreater(dot(directions['sar_one_sided'], go), 0)
        self.assertGreater(dot(directions['symmetric_pcgrad'], go), 0)
        self.assertTrue(torch.equal(go[0], original_o))
        self.assertTrue(torch.equal(gs[0], original_s))

    def test_no_conflict_keeps_raw_when_no_reweighting(self):
        go, gs = [torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 4.0])]
        all_dirs = generate_directions(go, gs)
        for key in ('symmetric_pcgrad', 'sar_one_sided'):
            torch.testing.assert_close(all_dirs[key][0], all_dirs['raw'][0])

    def test_update_norms_exactly_match(self):
        go, gs = [torch.tensor([1.0, 0.0])], [torch.tensor([-3.0, 4.0])]
        p = [torch.nn.Parameter(torch.tensor([2.0, -1.0]))]
        mag, d = equal_displacement(go, gs, p, fraction=0.01)
        self.assertGreater(mag, 0)
        for direction in d.values():
            self.assertAlmostEqual(vector_norm(direction), 1.0, places=6)
            start = p[0].detach().clone()
            apply_delta(p, direction, mag, [start])
            self.assertAlmostEqual(vector_norm([p[0].detach()-start]), mag, places=6)
            with torch.no_grad():
                p[0].copy_(start)
        meta = direction_metadata(go, gs, d)
        self.assertAlmostEqual(meta['sar_to_optical_gradient_norm'], 5.0)

    def test_balanced_changes_norm_not_grad_construction(self):
        go, gs = [torch.tensor([1.0, 0.0])], [torch.tensor([-3.0, 4.0])]
        values = generate_directions(go, gs)
        torch.testing.assert_close(values['norm_balance'][0], torch.tensor([0.4, 0.8]))

    def test_wrong_inputs(self):
        with self.assertRaises(ValueError):
            generate_directions([torch.ones(2)], [torch.ones(3)])
        with self.assertRaises(ValueError):
            generate_directions([torch.ones(2)], [torch.zeros(2)])
        with self.assertRaises(ValueError):
            equal_displacement([torch.ones(2)], [torch.ones(2)], [torch.ones(2)], 0)


if __name__ == '__main__':
    unittest.main()
