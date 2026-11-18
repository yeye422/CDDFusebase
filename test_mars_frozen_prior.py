"""CPU synthetic tests for MaRS frozen source-edge probes; no weights/data."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from diagnose_mars_frozen_prior import (
    PRETRAIN_MEAN, PRETRAIN_STD, RidgeStats, edge_target, feature_grid,
    line_score, mars_nchw, pixel_metrics, predict_ridge, shuffled_grid,
    standardize_mars_rgb, summary_rows, load_truncated_mars_checkpoint,
    checkpoint_path_for_metadata
)


class MaRSPriorTests(unittest.TestCase):
    def test_python38_compatible_checkpoint_path_serialization(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            nested=root/"models"/"external"/"mars.pth"
            self.assertEqual(
                checkpoint_path_for_metadata(nested,root),
                "models/external/mars.pth")
            self.assertEqual(
                checkpoint_path_for_metadata(root.parent/"elsewhere.pth",root),
                str((root.parent/"elsewhere.pth").resolve()))

    def test_official_full_checkpoint_can_load_truncated_features_strictly(self):
        class TwoStage(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0=nn.Linear(2,3)
                self.layers_1=nn.Linear(3,4)

        model=TwoStage()
        original={name:torch.full_like(value,0.25)
                  for name,value in model.state_dict().items()}
        full={**original,
              "layers_2.blocks.0.attn.qkv.weight":torch.ones(2,2),
              "layers_3.blocks.1.norm.weight":torch.ones(3)}
        with TemporaryDirectory() as folder:
            path=Path(folder)/"full.pth"
            torch.save(full,path)
            load_truncated_mars_checkpoint(model,path)
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,original[name])

    def test_checkpoint_loader_rejects_missing_shapes_and_unknown_keys(self):
        class TwoStage(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0=nn.Linear(2,3)
                self.layers_1=nn.Linear(3,4)

        model=TwoStage()
        full={name:value.clone() for name,value in model.state_dict().items()}
        full["layers_2.blocks.0.weight"]=torch.ones(1)
        with TemporaryDirectory() as folder:
            path=Path(folder)/"invalid.pth"
            missing=full.copy()
            missing.pop("layers_0.weight")
            torch.save(missing,path)
            with self.assertRaisesRegex(RuntimeError,"missing="):
                load_truncated_mars_checkpoint(model,path)
            wrong_shape=full.copy()
            wrong_shape["layers_1.weight"]=torch.ones(1,1)
            torch.save(wrong_shape,path)
            with self.assertRaisesRegex(RuntimeError,"shape_mismatch="):
                load_truncated_mars_checkpoint(model,path)
            unknown=full.copy()
            unknown["unexpected_layer.weight"]=torch.ones(1)
            torch.save(unknown,path)
            with self.assertRaisesRegex(RuntimeError,"unexpected="):
                load_truncated_mars_checkpoint(model,path)

    def test_ridge_intercept_with_large_nonzero_feature_means(self):
        """Reject the double-centering bug from the first MaRS run."""
        rng=np.random.default_rng(701)
        offsets=np.array([20.,-12.,7.,2.],dtype=np.float32)
        fit=RidgeStats(4,alpha=0.01)
        for scene in range(12):
            feat=(rng.normal(0,1,(4,32,32)).astype(np.float32)
                  +offsets[:,None,None])
            target=np.clip(0.45+0.12*(feat[0]-offsets[0])
                           -0.06*(feat[1]-offsets[1]),0,1)
            fit.add(feat,target,800,scene)
        weights=fit.solve()
        self.assertAlmostEqual(
            float(weights["intercept"]),fit.sy/fit.n,places=10)
        held=(rng.normal(0,1,(4,32,32)).astype(np.float32)
              +offsets[:,None,None])
        truth=np.clip(0.45+0.12*(held[0]-offsets[0])
                      -0.06*(held[1]-offsets[1]),0,1)
        prediction=predict_ridge(weights,held)
        self.assertLess(float(np.mean((prediction-truth)**2)),0.0005)
        self.assertGreater(float(prediction.std()),0.05)

    def test_constant_gradient_target_does_not_shift_with_feature_mean(self):
        fit=RidgeStats(3,alpha=0.01)
        rng=np.random.default_rng(71)
        for i in range(8):
            feat=(rng.normal(0,1,(3,16,16)).astype(np.float32)
                  +np.array([31.,-27.,8.],dtype=np.float32)[:,None,None])
            fit.add(feat,np.full((16,16),0.35,dtype=np.float32),
                    256,100+i)
        coeff=fit.solve()
        self.assertAlmostEqual(float(coeff["intercept"]),0.35,places=5)
        held=rng.normal(0,1,(3,16,16)).astype(np.float32)
        held+=np.array([31.,-27.,8.],dtype=np.float32)[:,None,None]
        pred=predict_ridge(coeff,held)
        self.assertLess(float(abs(pred-0.35).max()),1e-4)

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
