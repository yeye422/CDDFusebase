"""E2 initialization, unchanged losses, residual gradients and evaluator routing."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from direct_fusion import DirectFusionCDDFuse
from pac_mamba import PACMambaCDDFuse, PACMambaE2CDDFuse
from test_OS import load_variant_models, forward_variant, resolve_variant
from train_direct_fusion import DirectFusionLosses
from train_lwpd import forward_losses as e2_losses
from train_pac_mamba import forward_losses as pac_losses


def checkpoint(source):
    return dict(variant=source.variant, model=source.state_dict(),
                CDDF_Encoder=source.encoder.state_dict(), CDDF_Decoder=source.decoder.state_dict(),
                LWPD=source.lwpd.state_dict(), DirectBaseFusion=source.base_fusion.state_dict(),
                DirectDetailFusion=source.detail_fusion.state_dict())


class E2PACMambaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.source=DirectFusionCDDFuse('lwpd_e2').eval()
        self.model=PACMambaE2CDDFuse().eval()
        self.model.load_lwpd_checkpoint(checkpoint(self.source))
        self.o=torch.rand(1,1,16,16);self.s=torch.rand_like(self.o)

    def test_exact_e2_initialization_without_recovery_modules(self):
        self.assertEqual(self.model.lwpd.mode,'gated')
        self.assertFalse(hasattr(self.model.lwpd,'sar_recovery'))
        with torch.no_grad():
            a=self.source(self.o,self.s)
            b,features=self.model(self.o,self.s,return_features=True)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(features['pac']['update'].abs().sum().item(),0)

    def test_original_e2_loss_and_output_parameter_gradients_unchanged_at_init(self):
        args=SimpleNamespace(gradient_weight=10.,reconstruction_weight=.1,structure_weight=0.)
        losses=DirectFusionLosses()
        v=torch.ones_like(self.o);v[...,3:5,4:6]=0
        a,pa=e2_losses(self.source,losses,self.o,self.s,v,args)
        b,pb=pac_losses(self.model,losses,self.o,self.s,v,args)
        self.assertTrue(torch.equal(a,b))
        for key in ['fusion','intensity','gradient','reconstruction']:
            self.assertTrue(torch.equal(pa[key],pb[key]),key)
        ga=torch.autograd.grad(a,self.source.decoder.output[-1].weight)[0]
        gb=torch.autograd.grad(b,self.model.decoder.output[-1].weight)[0]
        self.assertTrue(torch.equal(ga,gb))

    def test_evaluator_round_trip_and_map_export(self):
        self.assertEqual(resolve_variant({'args':{'variant':'pac_mamba_e2'}},'auto'),'pac_mamba_e2')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pth'
            torch.save({'variant':'pac_mamba_e2','args':{'variant':'pac_mamba_e2'},'model':self.model.state_dict()},path)
            models,variant=load_variant_models(path,'cpu')
            with torch.no_grad():
                actual,maps=forward_variant(models,variant,self.o,self.s,return_maps=True)
                expected=self.model(self.o,self.s)
            self.assertTrue(torch.equal(actual,expected))
            self.assertIn('interaction_map',maps)

    def test_backbone_checkpoint_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.load_lwpd_checkpoint({'variant':'lwpd_e4'})
        with self.assertRaises(ValueError):
            PACMambaCDDFuse().load_lwpd_checkpoint(checkpoint(self.source))

    def test_residual_can_learn_then_propagate_to_ssm(self):
        fused=self.model(self.o,self.s)
        fused.mean().backward()
        projection=self.model.detail_fusion.update_projection[-1]
        self.assertGreater(projection.weight.grad.abs().sum().item(),0)
        # Zero initialization blocks upstream PAC gradients on the first step.
        # A nonzero projection must expose a live SSM path, not an unused branch.
        with torch.no_grad():projection.weight.normal_(0,.001)
        self.model.zero_grad()
        self.model(self.o,self.s).mean().backward()
        grad=self.model.detail_fusion.ssm.row_scan.parameter_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
