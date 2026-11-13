import math
import unittest
import numpy as np
import torch
import torch.nn as nn
import s2_region_diagnostic as d

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(1.0))
    def encode_pair(self,o,s):
        return (o.expand(-1,64,-1,-1), o.expand(-1,64,-1,-1),
                s.expand(-1,64,-1,-1),s.expand(-1,64,-1,-1))
    def fuse_features(self,bo,do,bs,ds):
        base=bo
        return base,self.detail_fusion(do,ds,base)
    def detail_fusion(self,do,ds,b):
        return do*0.3+ds*0.7
    def reconstruct(self,o,b,d):
        return (b[:,0:1]*0 + d[:,0:1]*self.scale)
    def forward(self,o,s):
        bo,do,bs,ds=self.encode_pair(o,s)
        b,d=self.fuse_features(bo,do,bs,ds)
        return self.reconstruct(o,b,d)

class ConditionalDummyModel(DummyModel):
    def detail_fusion(self,do,ds,b):
        alpha=torch.where(b[:,:1]>.5,torch.tensor(.8,device=b.device),torch.tensor(.2,device=b.device))
        return alpha*do+(1-alpha)*ds

class TestRegions(unittest.TestCase):
    def test_known_mixing_ratio(self):
        o=np.full((64,64),128,dtype=np.uint8)
        s=np.full((64,64),128,dtype=np.uint8)
        rows=d.region_rows(DummyModel().eval(),o,s,'synthetic',tile_size=32,step=.05)
        self.assertEqual(len(rows),4)
        for r in rows:
            self.assertAlmostEqual(r['optical_gain_per_feature_rms'],.3,places=5)
            self.assertAlmostEqual(r['sar_gain_per_feature_rms'],.7,places=5)
            self.assertAlmostEqual(r['log_sar_over_optical_gain'],math.log(7/3),places=5)
    def test_detects_spatial_condition_variation(self):
        o=np.full((64,64),51,dtype=np.uint8)
        o[:,32:]=204
        s=np.full((64,64),128,dtype=np.uint8)
        rows=d.region_rows(ConditionalDummyModel().eval(),o,s,'synthetic',tile_size=32,step=.05)
        self.assertEqual(len(rows),4)
        stats=d.summary(rows)
        self.assertAlmostEqual(stats['sar_gain_higher_fraction'],.5,places=7)
        self.assertGreater(rows[0]['log_sar_over_optical_gain'],0)
        self.assertLess(rows[1]['log_sar_over_optical_gain'],0)
    def test_spearman(self):
        self.assertAlmostEqual(d.spearman([1,2,3,4,5],[5,4,3,2,1]),-1,places=7)
        self.assertIsNone(d.spearman([1,1,1,1,1],[1,2,3,4,5]))
    def test_invalid_roi(self):
        o=np.ones((63,64),dtype=np.uint8)
        with self.assertRaises(ValueError):
            d.region_rows(DummyModel().eval(),o,o,'synthetic',tile_size=32)

if __name__=='__main__':unittest.main()