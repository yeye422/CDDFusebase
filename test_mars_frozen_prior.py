"""CPU synthetic tests for MaRS frozen source-edge probes; no weights/data."""
import unittest

import cv2
import numpy as np
import torch

from diagnose_mars_frozen_prior import (
    PRETRAIN_MEAN, PRETRAIN_STD, RidgeStats, edge_target, feature_grid,
    line_score, mars_nchw, pixel_metrics, predict_ridge, shuffled_grid,
    standardize_mars_rgb, summary_rows
)


class MaRSPriorTests(unittest.TestCase):
    def test_ridge_learns_signal_and_shift_breaks_alignment(self):
        rng=np.random.default_rng(42)
        ridge=RidgeStats(4,alpha=0.01)
        train=[]
        for scene in range(12):
            fmap=rng.normal(0,1,(4,32,32)).astype("float32")
            truth=np.clip(0.5+0.2*fmap[0]-0.1*fmap[1],0,1)
            train.append((fmap,truth))
            ridge.add(fmap,truth,600,scene)
        fitted=ridge.solve()
        held=rng.normal(0,1,(4,32,32)).astype("float32")
        target=np.clip(.5+.2*held[0]-.1*held[1],0,1)
        pred=predict_ridge(fitted,held)
        shuffled=predict_ridge(fitted,shuffled_grid(held))
        self.assertLess(pixel_metrics(pred,target)["mse"],0.003)
        self.assertGreater(pixel_metrics(shuffled,target)["mse"],
                           pixel_metrics(pred,target)["mse"]*10)

    def test_mars_preprocess_grayscale_and_rgb(self):
        rgb=np.zeros((16,16,3),dtype=np.uint8)
        rgb[:,:,0]=11
        rgb[:,:,1]=23
        rgb[:,:,2]=89
        gray=np.full((16,16),45,dtype=np.uint8)
        g=standardize_mars_rgb(rgb,gray,"gray")
        r=standardize_mars_rgb(rgb,gray,"rgb")
        self.assertEqual(tuple(g.shape),(1,3,512,512))
        np.testing.assert_allclose(g[0,:,5,7].numpy(),
            (np.full(3,45,dtype=np.float32)-PRETRAIN_MEAN)/PRETRAIN_STD,
            atol=1e-5)
        self.assertGreater(float((g-r).abs().max()),0.1)

    def test_swin_feature_layout(self):
        a=torch.randn(1,128,128,128)
        b=torch.randn(1,64,64,256)
        nhwc=mars_nchw([a,b],[128,256])
        self.assertEqual(tuple(nhwc[1].shape),(1,256,64,64))
        an=torch.randn(1,128,128,128)
        bn=torch.randn(1,256,64,64)
        nchw=mars_nchw([an,bn],[128,256])
        self.assertEqual(tuple(nchw[1].shape),(1,256,64,64))
        with self.assertRaises(ValueError):
            mars_nchw([a,torch.randn(1,64,64,64)],[128,256])

    def test_line_roi_holds_true_constant(self):
        row=dict(x1="8",y1="24",x2="56",y2="24")
        map256=np.full((256,256),.25,dtype=np.float32)
        score=line_score(map256,row)
        self.assertAlmostEqual(score["line_mean"],.25,places=6)
        self.assertAlmostEqual(score["line_continuity"],1,places=6)
        map256=np.zeros((256,256),np.float32)
        self.assertEqual(line_score(map256,row)["line_continuity"],0)

    def test_source_edge_target_shape_and_bounds(self):
        gray=np.zeros((256,256),np.uint8)
        gray[:,128:]=255
        gradient=edge_target(gray)
        self.assertEqual(gradient.shape,gray.shape)
        self.assertGreater(float(gradient[:,127:130].mean()),0)
        self.assertAlmostEqual(float(gradient[:,30:50].mean()),0,places=6)
        self.assertLessEqual(float(gradient.max()),1)

    def test_feature_grid_area_match(self):
        x=torch.ones((1,64,256,256),dtype=torch.float32)
        v=feature_grid(x,64)
        self.assertEqual(v.shape,(64,64,64))
        self.assertAlmostEqual(float(v.mean()),1,places=6)

    def test_paired_stats_have_matching_scene_stage(self):
        rows=[]
        for scene in ("a","b"):
            for stage in (0,1):
                rows.append(dict(scene=scene,stage=stage,method="e2_detail",
                                 mse=.2,mae=.2,high_edge_mae=.2,r2=.1))
                rows.append(dict(scene=scene,stage=stage,method="mars_gray",
                                 mse=.1,mae=.1,high_edge_mae=.1,r2=.2))
        result=summary_rows(rows,("stage","scene"),
                            ("mse","mae","high_edge_mae","r2"),"e2_detail")
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]["n_pairs"],2)
        self.assertAlmostEqual(result[0]["mse_mean_delta"],-.1)


if __name__=="__main__":
    unittest.main()
