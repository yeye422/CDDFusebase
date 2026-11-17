"""CPU unit tests for parameter-matched FSC local LL state refresh."""

from copy import deepcopy
import unittest

import torch

from e2_fsc_mamba import LLStateTransfer, scan_with_state
from e2_fsc_local_refresh import LocalRefreshTransfer, scan_with_refresh
from pac_mamba import BiAxialSelectiveStateSpace, SelectiveScan1D


class LocalRefreshTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        self.scan = SelectiveScan1D(8).eval()

    def test_no_refresh_matches_original_scan_and_is_differentiable(self):
        x = torch.randn(2, 13, 8, requires_grad=True)
        h = torch.randn(
            2, self.scan.inner, self.scan.state_channels, requires_grad=True)
        old = scan_with_state(self.scan, x, h)
        new = scan_with_refresh(self.scan, x, h)
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)
        new.square().mean().backward()
        self.assertGreater(h.grad.abs().sum().item(), 0.0)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_refresh_affects_only_tokens_at_or_after_boundary(self):
        x = torch.randn(2, 10, 8)
        h = torch.randn(2, self.scan.inner, self.scan.state_channels)
        changes = torch.stack((h, 2 * h, -h), dim=1)
        with torch.no_grad():
            zero = scan_with_refresh(self.scan, x, h)
            fresh = scan_with_refresh(
                self.scan, x, h, changes, chunk_size=4, carry=0.5)
        torch.testing.assert_close(
            zero[:, :4], fresh[:, :4], rtol=0, atol=0)
        self.assertGreater(
            (fresh[:, 4:] - zero[:, 4:]).abs().max().item(), 0.0)

    def test_one_chunk_equals_initial_state_and_carry_one_disables_refresh(self):
        x = torch.randn(1, 7, 8)
        h = torch.randn(1, self.scan.inner, self.scan.state_channels)
        states = torch.stack((h, 3 * h), dim=1)
        with torch.no_grad():
            initial = scan_with_refresh(self.scan, x, h)
            same = scan_with_refresh(
                self.scan, x, h, states, chunk_size=4, carry=1.0)
        torch.testing.assert_close(initial, same, atol=0, rtol=0)

    def test_matched_trainable_parameters_and_local_correspondence(self):
        previous = LLStateTransfer(
            BiAxialSelectiveStateSpace(8), "state").eval()
        modes = [LocalRefreshTransfer(deepcopy(previous), name) for name in
                 ("local_feature", "local_refresh", "local_shuffled")]
        for other in modes[1:]:
            self.assertEqual(
                set(modes[0].state_dict()), set(other.state_dict()))
            for key, value in modes[0].state_dict().items():
                torch.testing.assert_close(
                    value, other.state_dict()[key], atol=0, rtol=0)
        x = torch.randn(1, 8, 10, 11)
        o, s = torch.randn_like(x), torch.randn_like(x)
        with torch.no_grad():
            feature, refreshed, wrong = (
                m(x, o, s) for m in modes)
        self.assertEqual(tuple(refreshed.shape), tuple(x.shape))
        self.assertGreater(
            (refreshed - feature).square().mean().item(), 0)
        self.assertGreater(
            (refreshed - wrong).square().mean().item(), 0)

    def test_refresh_gradient_reaches_ll_projection_and_scan(self):
        previous = LLStateTransfer(
            BiAxialSelectiveStateSpace(8), "state")
        ssm = LocalRefreshTransfer(previous, "local_refresh", chunk_size=4)
        x = torch.randn(1, 8, 8, 8)
        o, s = torch.randn_like(x), torch.randn_like(x)
        out = ssm(x, o, s)
        out.square().mean().backward()
        for parameter in (
            ssm.ll_projection.weight,
            ssm.decay_logit,
            ssm.core.row_scan.a_log,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_validation(self):
        x = torch.randn(1, 6, 8)
        h = torch.zeros(1, self.scan.inner, self.scan.state_channels)
        with self.assertRaises(ValueError):
            scan_with_refresh(
                self.scan, x, h, torch.zeros(1, 1, 8, 4), chunk_size=2)
        with self.assertRaises(ValueError):
            scan_with_refresh(self.scan, x, h, carry=1.2)
        with self.assertRaises(ValueError):
            LocalRefreshTransfer(
                LLStateTransfer(BiAxialSelectiveStateSpace(8), "state"),
                "local_refresh", chunk_size=0)


if __name__ == "__main__":
    unittest.main()
