"""CPU synthetic tests for frozen E2 Detail routing localization (no QXS)."""
import unittest

import torch
from torch import nn

from diagnose_e2_detail_route_localization import (
    unit_rms_perturbation, cosine, route_trace, route_probe,
    original_winner_stats, optical_edge_loss, gradient_conflict, summary
)
from diagnose_fcsr_decoder import gradients


class TinyDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical_projection=nn.Conv2d(4,3,1)
        self.sar_projection=nn.Conv2d(4,3,1)
        self.base_projection=nn.Conv2d(4,2,1)
        self.scale_branches=nn.ModuleList([nn.Conv2d(6,3,3,padding=1)
                                            for _ in range(3)])
        self.output=nn.Conv2d(17,4,1)

    def _project_inputs(self,o,s,b):
        return self.optical_projection(o), self.sar_projection(s), self.base_projection(b)

    def forward(self,o,s,b):
        op,sp,bp=self._project_inputs(o,s,b)
        joined=torch.cat((op,sp),dim=1)
        scales=[layer(joined) for layer in self.scale_branches]
        return self.output(torch.cat((op,sp,*scales,bp),dim=1))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_fusion=TinyDetail()
        self.decoder=nn.Conv2d(8,1,3,padding=1)

    def reconstruct(self,source,base,detail):
        return torch.sigmoid(self.decoder(torch.cat((base,detail),dim=1)))


class TinyLoss(nn.Module):
    def sobel(self,image):
        return gradients(image).abs().sum(1,keepdim=True)

    def fusion(self,optical,sar,fused,valid,gradient_weight):
        intensity=torch.where(valid,torch.maximum(optical,sar),optical)
        target_gradient=torch.where(valid,
                torch.maximum(self.sobel(optical),self.sobel(sar)),
                self.sobel(optical))
        i=(fused-intensity).abs().mean()
        g=(self.sobel(fused)-target_gradient).abs().mean()
        return i+gradient_weight*g,i,g


class RoutingTests(unittest.TestCase):
    def test_equal_feature_rms(self):
        source=torch.randn(1,4,16,16)
        modified=unit_rms_perturbation(source,0.024)
        self.assertAlmostEqual(float(modified.square().mean().sqrt()),0.024,places=6)
        self.assertIsNone(cosine(torch.zeros_like(source),source))

    def test_intermediate_trace_is_exact(self):
        torch.manual_seed(14)
        model=TinyModel().eval()
        o,s,b=[torch.randn(1,4,32,32) for _ in range(3)]
        image=torch.ones(1,1,32,32)
        stages=route_trace(model,o,s,b,image)
        torch.testing.assert_close(stages["detail"],model.detail_fusion(o,s,b))
        torch.testing.assert_close(stages["image"],
                                   model.reconstruct(image,b,model.detail_fusion(o,s,b)))
        self.assertEqual(stages["scales"].shape[1],9)

    def test_matched_rms_routes(self):
        torch.manual_seed(15)
        model=TinyModel().eval()
        o,s,b=[torch.randn(1,4,32,32) for _ in range(3)]
        image=torch.ones(1,1,32,32)
        mask=torch.ones(1,1,32,32,dtype=torch.bool)
        baseline=route_trace(model,o,s,b,image)
        rows=route_probe(model,image,image,o,s,b,baseline,
                         {"optical":mask,"sar":mask},0.015)
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[0]["input_perturbation_rms"],
                               rows[1]["input_perturbation_rms"],places=7)
        for row in rows:
            self.assertGreaterEqual(row["projection_gain"],0)
            self.assertGreaterEqual(row["scales_gain"],0)
            self.assertGreaterEqual(row["detail_gain"],0)
            self.assertGreaterEqual(row["image_gain"],0)
            self.assertGreaterEqual(row["decoder_gain_from_actual_detail_direction"],0)

    def test_loss_winner_metrics(self):
        losses=TinyLoss()
        o=torch.ones(1,1,32,32)*0.2
        s=torch.ones(1,1,32,32)*0.8
        valid=torch.ones(1,1,32,32,dtype=torch.bool)
        mask=torch.ones_like(valid)
        result=original_winner_stats(o,s,valid,{"optical":mask,"sar":mask},losses)
        self.assertEqual(result["valid_all_pixels"],1024)
        self.assertEqual(result["valid_all_sar_intensity_winner_fraction"],1.)

    def test_gradient_conflict_runs_with_frozen_model(self):
        torch.manual_seed(16)
        model=TinyModel().eval().requires_grad_(False)
        o,s=torch.rand(1,1,32,32),torch.rand(1,1,32,32)
        bo,do,bs,ds=[torch.randn(1,4,32,32) for _ in range(4)]
        bf=bo+bs*0.1
        with torch.no_grad():
            df=model.detail_fusion(do,ds,bf)
        mask=torch.ones(1,1,32,32,dtype=torch.bool)
        result=gradient_conflict(model,TinyLoss(),o,s,
                                 torch.ones_like(mask),bf,df,
                                 {"optical":mask,"sar":mask},10.0)
        self.assertEqual(result["status"],"ok")
        self.assertTrue(torch.isfinite(
            torch.tensor(result["loss_original_optical"])))
        self.assertTrue(torch.isfinite(
            torch.tensor(result["loss_original_sar"])))

    def test_local_proxy_gradient(self):
        a=torch.randn(1,1,32,32)
        b=a.clone()
        mask=torch.ones_like(a,dtype=torch.bool)
        loss=optical_edge_loss(b,a,mask)
        self.assertAlmostEqual(float(loss),0.,places=7)


if __name__=="__main__":
    unittest.main()
