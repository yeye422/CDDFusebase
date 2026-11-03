import unittest
from types import SimpleNamespace

import numpy as np
import torch
from skimage.metrics import structural_similarity

from direct_fusion import DirectFusionCDDFuse
from train_direct_fusion import DirectFusionLosses
from train_lwpd import forward_losses
from utils.fusion_structure_loss import FusionStructureLoss


torch.set_num_threads(2)


class StructureLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.loss=FusionStructureLoss()
        self.a=torch.rand(1,1,32,34)
        self.b=torch.rand_like(self.a)
        self.f=((self.a+self.b)/2).detach().requires_grad_(True)
        self.valid=torch.ones_like(self.a)

    def test_identity(self):
        value,_=self.loss(self.a,self.a,self.a,self.valid)
        self.assertLess(abs(value.item()),1e-5)

    def test_against_independent_ssim(self):
        value,_=self.loss(self.a,self.b,self.f,self.valid)
        ss=lambda x:structural_similarity(x.squeeze().numpy(),self.f.detach().squeeze().numpy(),
                                          data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
        self.assertAlmostEqual(value.item(),1-(ss(self.a)+ss(self.b))/2,places=5)
        value.backward()
        self.assertTrue(torch.isfinite(self.f.grad).all())
        self.assertGreater(self.f.grad.abs().sum().item(),0)

    def test_invalid_sar_is_excluded(self):
        self.valid[:,:,12,13]=0
        b=self.b.clone().requires_grad_(True)
        value,_=self.loss(self.a,b,self.f,self.valid)
        altered=self.b.clone();altered[:,:,12,13]=float('nan')
        other,_=self.loss(self.a,altered,self.f,self.valid)
        self.assertEqual(value.item(),other.item())
        value.backward()
        self.assertEqual(b.grad[0,0,12,13].item(),0)

    def test_no_valid_sar_falls_back_without_halving(self):
        value,stats=self.loss(self.a,self.b,self.f,torch.zeros_like(self.valid))
        expected=self.loss.dissimilarity(self.a,self.f).mean()
        self.assertEqual(value.item(),expected.item())
        self.assertEqual(stats['structure_sar_available'].item(),0)

    def test_batch_reduces_per_image(self):
        a=self.a.repeat(2,1,1,1);b=self.b.repeat(2,1,1,1);f=self.f.repeat(2,1,1,1)
        valid=self.valid.repeat(2,1,1,1);valid[1]=0
        value,_=self.loss(a,b,f,valid)
        x,_=self.loss(self.a,self.b,self.f,self.valid)
        y,_=self.loss(self.a,self.b,self.f,torch.zeros_like(self.valid))
        self.assertAlmostEqual(value.item(),((x+y)/2).item(),places=6)

    def test_zero_weight_preserves_original_loss_and_gradients(self):
        model=DirectFusionCDDFuse('lwpd_e2').eval()
        losses=DirectFusionLosses()
        args=SimpleNamespace(gradient_weight=10.,reconstruction_weight=.1,structure_weight=0.)
        total,parts=forward_losses(model,losses,self.a,self.b,self.valid,args)
        # No new module is present; reaching it would raise AttributeError.
        self.assertNotIn('structure',parts)
        expected=parts['fusion']+.1*parts['reconstruction']
        self.assertEqual(total.item(),expected.item())
        params=[p for p in model.parameters() if p.requires_grad]
        g1=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
        g2=torch.autograd.grad(expected,params,allow_unused=True)
        for a,b in zip(g1,g2):
            self.assertEqual(a is None,b is None)
            if a is not None:self.assertTrue(torch.equal(a,b))

    def test_complete_e2_loss_forward_backward(self):
        model=DirectFusionCDDFuse('lwpd_e2').eval()
        losses=DirectFusionLosses();losses.fusion_structure=self.loss
        args=SimpleNamespace(gradient_weight=10.,reconstruction_weight=.1,structure_weight=.5)
        total,parts=forward_losses(model,losses,self.a,self.b,self.valid,args)
        expected=parts['fusion']+.1*parts['reconstruction']+.5*parts['structure']
        self.assertEqual(total.item(),expected.item())
        total.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0)


if __name__=='__main__':unittest.main()
