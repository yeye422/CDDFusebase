"""CPU-only synthetic tests for S2 operator comparison (no user data required)."""
import unittest
import torch
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


if __name__ == '__main__':
    unittest.main()
