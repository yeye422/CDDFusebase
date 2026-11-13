import unittest

import torch

from utils.e2_target_diagnostics import direction_evidence, make_targets, output_diagnostics
from train_direct_fusion import DirectFusionLosses


class TargetDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.o, self.s = torch.rand(1, 1, 24, 24), torch.rand(1, 1, 24, 24)
        self.v = torch.ones_like(self.o)

    def test_eta_zero_matches_production_values_and_output_gradient(self):
        t = make_targets(self.o, self.s, self.v, eta=0)
        f = torch.rand_like(self.o, requires_grad=True)
        production = DirectFusionLosses().fusion(self.o, self.s, f, self.v, 10)[0]
        from utils.loss import Sobelxy
        diagnostic = (f-t['intensity']).abs().mean()+10*(Sobelxy()(f)-t['corrected']).abs().mean()
        self.assertTrue(torch.equal(t['original'], t['corrected']))
        self.assertTrue(torch.equal(production, diagnostic))
        self.assertTrue(torch.equal(torch.autograd.grad(production, f, retain_graph=True)[0], torch.autograd.grad(diagnostic, f)[0]))

    def test_bounds_and_optical_dominant_unchanged(self):
        t = make_targets(self.o, self.s, self.v)
        from utils.loss import Sobelxy
        self.assertTrue((t['corrected'] >= .9*t['original']-1e-6).all())
        self.assertTrue((t['corrected'] <= t['original']).all())
        self.assertTrue((t['corrected'] >= Sobelxy()(self.o)-1e-6).all())
        self.assertTrue(torch.equal(t['original'][t['excess']==0],t['corrected'][t['excess']==0]))

    def test_full_support_and_external_border(self):
        v = self.v.clone(); v[..., 12, 12] = 0
        _, m = direction_evidence(self.s, v)
        self.assertEqual(m[..., 9:16, 9:16].sum(), 0)
        self.assertEqual(m[..., :3, :].sum(), 0)
        self.assertEqual(m[..., -3:, :].sum(), 0)
        self.assertEqual(m[..., :, :3].sum(), 0)
        self.assertEqual(m[..., :, -3:].sum(), 0)
        self.assertEqual(m[..., 5, 5].item(), 1)

    def test_invalid_nan_cannot_change_evidence(self):
        v=self.v.clone();v[...,12,12]=0
        a=self.s.clone();a[...,12,12]=0
        b=a.clone();b[...,12,12]=float('nan')
        x,m=direction_evidence(a,v);y,n=direction_evidence(b,v)
        self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(m,n))

    def test_invalid_fallback_and_detachment(self):
        t=make_targets(self.o.requires_grad_(True),self.s.requires_grad_(True),torch.zeros_like(self.v))
        from utils.loss import Sobelxy
        self.assertTrue(torch.equal(t['corrected'],Sobelxy()(self.o.detach())))
        self.assertTrue(all(not x.requires_grad for x in t.values()))

    def test_opposite_edges_prove_target_inconsistency(self):
        o=torch.zeros_like(self.o);o[...,12:]=1
        t=make_targets(o,1-o,self.v)
        self.assertGreater(t['target_gap'][...,4:-4,10:14].abs().max().item(), 0)

    def test_diagnostics_finite_and_zero_target_preserves_gradients(self):
        t=make_targets(self.o,self.s,self.v,eta=0)
        stats,maps=output_diagnostics((self.o+self.s)/2,self.o,self.s,self.v,t)
        self.assertEqual(stats['output_gradient_change_norm'],0)
        self.assertTrue(all(torch.isfinite(x).all() for x in maps.values()))
        self.assertTrue(all(x is None or torch.isfinite(torch.tensor(x)) for x in stats.values()))

    def test_small_l1_target_shift_can_reverse_derivative(self):
        x=torch.tensor(.95,requires_grad=True)
        before=torch.autograd.grad((x-1.).abs(),x,retain_graph=True)[0]
        after=torch.autograd.grad((x-.9).abs(),x)[0]
        self.assertEqual(before.item(),-1.)
        self.assertEqual(after.item(),1.)


if __name__ == '__main__':
    unittest.main()
