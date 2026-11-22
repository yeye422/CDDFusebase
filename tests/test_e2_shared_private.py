import unittest
from unittest.mock import patch

import torch

from e2_shared_private import SharedPrivateE2, patch_alignment
from run_e2_shared_private import PILOT, configure, private_readouts, register_pair, GROUP
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, training_args
from test_OS import DIRECT_MODEL_TYPES, resolve_variant
from train_lwpd import make_optimizer


class SharedPrivateTests(unittest.TestCase):
    def test_matching_patch_costs_less_than_rolled(self):
        x = torch.eye(4).reshape(4, 2, 2)[None].repeat_interleave(16, 2).repeat_interleave(16, 3)
        mask = torch.ones_like(x[:, :1])
        self.assertLess(patch_alignment(x, x, mask).item(), patch_alignment(x, x.roll(16, -1), mask).item())

    def test_sar_nodata_and_private_channels_are_excluded_from_alignment(self):
        torch.manual_seed(7)
        x = torch.randn(2, 32, 33, 45, requires_grad=True)
        y = torch.randn_like(x, requires_grad=True)
        valid = torch.ones_like(x[:, :1])
        valid[0] = 0
        valid[1, :, :, :4] = 0
        loss = patch_alignment(x[:, :16], y[:, :16], valid)
        loss.backward()
        for grad in (x.grad, y.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(grad[:, 16:].count_nonzero().item(), 0)
            self.assertEqual((grad * (1-valid)).count_nonzero().item(), 0)
            self.assertGreater(grad[:, :16].abs().sum().item(), 0)
        mask = torch.zeros_like(valid)
        self.assertEqual(patch_alignment(x[:, :16], y[:, :16], mask).item(), 0)
        mask[0, :, 0, 0] = 1
        self.assertEqual(patch_alignment(x[:, :16], y[:, :16], mask).item(), 0)

    def test_stage_configuration_and_normal_inference(self):
        model = PILOT.build_model()
        self.assertIs(DIRECT_MODEL_TYPES[resolve_variant({"args": {"variant": model.variant}}, "auto")], SharedPrivateE2)
        args = training_args(PILOT)
        optimizer = make_optimizer(model, args)
        head = model.detail_fusion
        configure(model, optimizer, args, True)
        group = next(g for g in optimizer.param_groups if g["group_name"] == "detail_fusion")
        self.assertGreater(group["lr"], 0)
        for param in (head.optical_projection.weight, head.sar_projection.weight, head.optical_readout.weight,
                      head.sar_readout.weight):
            self.assertTrue(param.requires_grad and any(p is param for p in group["params"]))
        self.assertFalse(head.private_gain.weight.requires_grad)
        configure(model, optimizer, args, False)
        self.assertTrue(head.private_gain.weight.requires_grad)
        model.eval()
        optical, sar = [torch.rand(1, 1, 25, 27) for _ in range(2)]
        actual = private_readouts(model, optical, sar)
        torch.testing.assert_close(actual["candidate"], model(optical, sar))
        for mode in ("optical_private_off", "sar_private_off"):
            self.assertFalse(torch.equal(actual["candidate"], actual[mode]))

    def test_registers_distinct_group_and_artifacts(self):
        runs = (("e2", ORIGINAL_LOSS_PILOT), ("candidate", PILOT))
        entries = [dict(checkpoint=e.final.as_posix(), result_dir=(PILOT.result/k).as_posix()) for k,e in runs]
        with patch("run_e2_shared_private.register_many") as save:
            register_pair(entries)
        submitted, = save.call_args.args
        self.assertEqual([r["id"] for r in submitted], [GROUP+"-e2", GROUP+"-candidate"])
