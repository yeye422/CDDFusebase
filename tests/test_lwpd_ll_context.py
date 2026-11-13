import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from direct_fusion import DirectFusionCDDFuse
from lwpd_ll_context import CrossModalLLContext
from train_direct_fusion import DirectFusionLosses
from train_lwpd import forward_losses
from train_lwpd import parse_args, make_optimizer, configure_stage
from test_OS import load_variant_models


class LLContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.base=DirectFusionCDDFuse('lwpd_e2').eval()
        self.model=DirectFusionCDDFuse('lwpd_e2_llctx_v1').eval()
        self.model.load_e2_llctx_checkpoint({'variant':'lwpd_e2','model':self.base.state_dict()})
        self.o=torch.rand(1,1,17,19);self.s=torch.rand_like(self.o)

    def test_initial_outputs_losses_and_representative_gradients_equal(self):
        with torch.no_grad():
            a,fa=self.base(self.o,self.s,return_features=True)
            b,fb=self.model(self.o,self.s,return_features=True)
        self.assertTrue(torch.equal(a,b))
        for key in ['optical_base','optical_detail','sar_base','sar_detail']:
            self.assertTrue(torch.equal(fa[key],fb[key]),key)
        args=SimpleNamespace(gradient_weight=10.,reconstruction_weight=.1,structure_weight=0.)
        loss=DirectFusionLosses();v=torch.ones_like(self.o)
        x,_=forward_losses(self.base,loss,self.o,self.s,v,args)
        y,_=forward_losses(self.model,loss,self.o,self.s,v,args)
        self.assertTrue(torch.equal(x,y))
        gx=torch.autograd.grad(x,self.base.decoder.output[-1].weight)[0]
        gy=torch.autograd.grad(y,self.model.decoder.output[-1].weight)[0]
        self.assertTrue(torch.equal(gx,gy))

    def test_parameter_count_shapes_and_bounded_changes(self):
        module=self.model.lwpd.ll_context
        self.assertEqual(sum(p.numel() for p in module.parameters()),9520)
        with torch.no_grad():
            module.optical_head.weight.normal_();module.sar_head.weight.normal_()
        a=torch.randn(1,64,9,11);b=torch.randn_like(a)
        da,db,_=module(a,b)
        self.assertEqual(da.shape,a.shape)
        self.assertTrue((da.abs()<=.5).all() and (db.abs()<=.5).all())
        olda,oldb,_=self.base.lwpd._modulate_low_bands(a,b)
        newa,newb,_=self.model.lwpd._modulate_low_bands(a,b)
        self.assertTrue(((newa-olda).abs()<=.05*a.abs()+1e-6).all())
        self.assertTrue(((newb-oldb).abs()<=.05*b.abs()+1e-6).all())

    def test_context_gradients_after_zero_projection_is_released(self):
        module=CrossModalLLContext(8,4)
        a=torch.randn(1,8,9,11);b=torch.randn_like(a)
        da,db,_=module(a,b);(da.square().sum()+da.sum()+db.sum()).backward()
        self.assertGreater(module.optical_head.weight.grad.abs().sum().item(),0)
        with torch.no_grad():module.optical_head.weight.normal_(0,.01)
        module.zero_grad();da,_,_=module(a,b);da.square().mean().backward()
        self.assertTrue(torch.isfinite(module.phi[0].weight.grad).all())
        self.assertGreater(module.phi[0].weight.grad.abs().sum().item(),0)

    def test_checkpoint_eval_round_trip_and_invalid_parent(self):
        with self.assertRaises(ValueError):self.model.load_e2_llctx_checkpoint({'variant':'lwpd_e4'})
        state=self.base.state_dict().copy();state.pop('lwpd.optical_base.0.weight')
        with self.assertRaises(RuntimeError):self.model.load_e2_llctx_checkpoint({'variant':'lwpd_e2','model':state})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pth'
            torch.save({'args':{'variant':'lwpd_e2_llctx_v1'},'model':self.model.state_dict()},path)
            models,variant=load_variant_models(path,'cpu')
            self.assertEqual(variant,'lwpd_e2_llctx_v1')
            with torch.no_grad():self.assertTrue(torch.equal(models['direct'](self.o,self.s),self.model(self.o,self.s)))

    def test_new_context_has_distinct_optimizer_group(self):
        with patch('sys.argv',['train_lwpd.py','--init-checkpoint','unused','--run-name','test',
                               '--variant','lwpd_e2_llctx_v1','--warmup-lwpd-lr','1e-5',
                               '--warmup-fusion-lr','1e-5']):
            args=parse_args()
        optimizer=make_optimizer(self.model,args)
        configure_stage(self.model,optimizer,'warmup',args)
        groups={g['group_name']:g for g in optimizer.param_groups}
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual({id(p) for p in groups['ll_context']['params']},
                         {id(p) for p in self.model.lwpd.ll_context.parameters()})
        self.assertEqual(groups['ll_context']['lr'],1e-4)
        self.assertEqual(groups['lwpd']['lr'],1e-5)
        self.assertEqual(groups['decoder']['lr'],1e-5)


if __name__=='__main__':unittest.main()
