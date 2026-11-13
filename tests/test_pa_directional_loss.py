import unittest
import torch
from types import SimpleNamespace
from utils.pa_directional_loss import PADirectionalGradientLoss
from train_direct_fusion import DirectFusionLosses
from train_lwpd import forward_losses
from direct_fusion import DirectFusionCDDFuse

torch.set_num_threads(2)

class PADirectionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.loss=PADirectionalGradientLoss()
        self.o=torch.rand(1,1,32,34)
        self.s=torch.rand_like(self.o)
        self.v=torch.ones_like(self.o)

    def test_identity_and_backward(self):
        f=self.o.clone().requires_grad_(True)
        loss=self.loss(self.o,self.o,f,self.v)
        self.assertLess(abs(loss.item()),1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_invalid_sar_and_detached_mask(self):
        self.v[:,:,:,:8]=0
        bad=self.s.clone();bad[:,:,:,:8]=float('nan')
        f=self.o.clone().requires_grad_(True)
        self.assertEqual(self.loss(self.o,self.s,f,self.v).item(),self.loss(self.o,bad,f,self.v).item())
        m,t=self.loss.evidence(self.o.requires_grad_(True),self.s,self.v)
        self.assertFalse(m.requires_grad or t.requires_grad)
        self.assertEqual(m[:,:,:,:9].sum().item(),0)

    def test_no_valid_sar_ignores_sar(self):
        f=torch.rand_like(self.o)
        self.assertEqual(self.loss(self.o,self.s,f,self.v*0).item(),self.loss(self.o,self.s*100,f,self.v*0).item())

    def test_direction_convention_and_periodicity(self):
        step=torch.zeros(1,1,64,64);step[:,:,:,32:]=1
        m,t=self.loss.evidence(step,step,torch.ones_like(step))
        self.assertGreater(m[:,:,:,31:33].mean().item(),.01)
        self.assertGreater(torch.cos(2*t[:,:,:,31:33]).mean().item(),.9)
        self.assertTrue(torch.allclose(self.loss.agreement(t,t+torch.pi),torch.ones_like(t),atol=1e-5))
        self.assertLess(self.loss.agreement(t,t+torch.pi/2).max().item(),1e-5)

    def test_complete_e2_backward(self):
        model=DirectFusionCDDFuse('lwpd_e2').eval()
        losses=DirectFusionLosses();losses.pa_directional=self.loss
        args=SimpleNamespace(gradient_weight=10.,gradient_loss='pa_directional',reconstruction_weight=.1,structure_weight=0.)
        total,parts=forward_losses(model,losses,self.o,self.s,self.v,args)
        self.assertTrue(torch.allclose(total,parts['intensity']+10*parts['gradient']+.1*parts['reconstruction']))
        total.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
