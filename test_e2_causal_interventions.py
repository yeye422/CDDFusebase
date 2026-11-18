"""Synthetic CPU tests for controlled E2 Detail Fusion interventions.

No QXS images, checkpoint or full 14-metric validation is required here.
"""
import unittest

import numpy as np
import torch
from torch import nn

from diagnose_e2_causal_interventions import (
    STAGES, DIRECTIONS, SIGNS, direction_tensors, evaluate_metrics,
    intervene_detail, normalized_direction, paired_summary, matched_contrasts, projection_context,
    shuffled_direction, tensor_rms, uint8_output,
)


class TinyDetail(nn.Module):
    def __init__(self, channels=8, projected=4, bproj=2):
        super().__init__()
        self.optical_projection=nn.Sequential(nn.Conv2d(channels,projected,1),nn.GELU())
        self.sar_projection=nn.Sequential(nn.Conv2d(channels,projected,1),nn.GELU())
        self.base_projection=nn.Sequential(nn.Conv2d(channels,bproj,1),nn.GELU())
        self.scale_branches=nn.ModuleList([
            nn.Sequential(nn.Conv2d(projected*2,projected,3,padding=1),
                          nn.GELU()) for _ in range(3)])
        self.output=nn.Sequential(nn.Conv2d(projected*5+bproj,channels,1),
                                  nn.GELU(),nn.Conv2d(channels,channels,3,padding=1))

    def _project_inputs(self,o,s,b):
        return (self.optical_projection(o),self.sar_projection(s),
                self.base_projection(b))

    def forward(self,o,s,b):
        op,sp,bp=self._project_inputs(o,s,b)
        joint=torch.cat((op,sp),1)
        scales=[branch(joint) for branch in self.scale_branches]
        return self.output(torch.cat((op,sp,*scales,bp),1))


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_fusion=TinyDetail()
        self.decoder=nn.Conv2d(16,1,3,padding=1)

    def reconstruct(self,optical,base,detail):
        return torch.sigmoid(self.decoder(torch.cat((base,detail),1)))


class CausalInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=TinyNetwork().eval()
        self.o=torch.randn(1,8,24,24)
        self.s=torch.randn(1,8,24,24)
        self.b=torch.randn(1,8,24,24)
        self.image=torch.rand(1,1,24,24)
        self.ctx=projection_context(self.model,
                                    (self.b,self.o,self.b,self.s),self.b)

    def test_context_matches_unmodified_detail(self):
        actual=self.model.detail_fusion(self.o,self.s,self.b)
        torch.testing.assert_close(self.ctx["detail"],actual,atol=1e-7,rtol=0)

    def test_null_direction_preserves_projected_global_rms(self):
        x=self.ctx["op"]
        shuffled=shuffled_direction(x)
        torch.testing.assert_close(shuffled,shuffled_direction(x))
        self.assertAlmostEqual(tensor_rms(x),tensor_rms(shuffled),places=6)
        self.assertFalse(torch.allclose(x,shuffled))

    def test_equal_absolute_amplitude_for_sources(self):
        anchor=0.03
        for direction,x in direction_tensors(self.ctx).items():
            scaled=normalized_direction(x,anchor)
            self.assertAlmostEqual(tensor_rms(scaled),anchor,places=6)

    def test_all_interventions_leave_original_context_untouched(self):
        baseline=self.ctx["detail"].clone()
        original_aggregate=self.ctx["agg"].clone()
        original_optical=self.ctx["op"].clone()
        original_sar=self.ctx["sp"].clone()
        for stage in STAGES:
            for direction in DIRECTIONS:
                for sign in SIGNS:
                    d=intervene_detail(self.model,self.ctx,stage,direction,sign,0.02)
                    self.assertEqual(d.shape,baseline.shape)
                    self.assertTrue(torch.isfinite(d).all())
        torch.testing.assert_close(self.ctx["detail"],baseline)
        torch.testing.assert_close(self.ctx["agg"],original_aggregate)
        torch.testing.assert_close(self.ctx["op"],original_optical)
        torch.testing.assert_close(self.ctx["sp"],original_sar)

    def test_late_only_direct_slot(self):
        module=self.model.detail_fusion
        amplitude=0.02
        delta=normalized_direction(self.ctx["op"],amplitude)
        agg=self.ctx["agg"].clone()
        agg[:,:delta.shape[1]] += delta
        want=module.output(agg)
        got=intervene_detail(self.model,self.ctx,"late","optical",+1,amplitude)
        torch.testing.assert_close(got,want,atol=1e-7,rtol=0)

    def test_early_recomputes_scales(self):
        module=self.model.detail_fusion
        amplitude=0.025
        o=self.ctx["op"]+normalized_direction(self.ctx["op"],amplitude)
        s=self.ctx["sp"]
        scales=[branch(torch.cat((o,s),1))
                for branch in module.scale_branches]
        reference=module.output(torch.cat((o,s,*scales,self.ctx["bp"]),1))
        actual=intervene_detail(self.model,self.ctx,"early","optical",+1,amplitude)
        torch.testing.assert_close(actual,reference,atol=1e-7,rtol=0)

    def test_quantization_and_output_shapes(self):
        data=torch.tensor([[[[0.,0.499,1.,0.9]]]])
        output=uint8_output(data)
        np.testing.assert_array_equal(output,np.array([[0,127,255,230]],dtype=np.uint8))
        with self.assertRaises(FloatingPointError):
            uint8_output(torch.tensor([[[[float("nan")]]]]))

    def test_finite_pairwise_metric_comparison(self):
        rows=[]
        all_methods=[("baseline","none",0)]
        all_methods.extend((stage,direction,sign)
                           for stage in STAGES for direction in DIRECTIONS
                           for sign in SIGNS)
        for scene in ("100","200"):
            for stage,direction,sign in all_methods:
                candidate=stage!="baseline"
                rows.append(dict(
                    scene=scene,stage=stage,direction=direction,sign=sign,
                    FMI=0.5+(0.01 if candidate else 0),
                    Qabf=0.6+(0.01 if candidate else 0),
                    Nabf=0.1-(0.01 if candidate else 0),
                    SSIM=0.55,VIF=0.4,MS_SSIM=0.58,SCD=1.05,
                    source_optical_transfer=0.3+(0.01 if candidate else 0),
                    source_sar_transfer=0.9,
                    quantized_changed_pixel_fraction=0.015 if candidate else 0,
                    image_mae_8bit_unquantized=0.25 if candidate else 0,
                    realized_detail_delta_rms=0.002 if candidate else 0,
                ))
        report=paired_summary(rows)
        contrasts=matched_contrasts(rows)
        self.assertEqual(len(contrasts),2*2*2*2)
        self.assertEqual(set(item["comparison"] for item in contrasts),
                         {"optical_minus_sar","optical_minus_shuffled_optical"})
        self.assertTrue(all(abs(item["Qabf_contrast"])<1e-12
                            for item in contrasts))
        key="early:optical:+1"
        self.assertEqual(report[key]["Qabf"]["n"],2)
        self.assertEqual(report[key]["Qabf"]["improved_scene_count"],2)
        self.assertEqual(report[key]["Nabf"]["improved_scene_count"],2)
        self.assertAlmostEqual(report[key]["source_optical_transfer"]["mean_delta"],0.01)


if __name__=="__main__":
    unittest.main()
