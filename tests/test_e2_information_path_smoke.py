"""Synthetic-only smoke test; no claim of real QXS data or checkpoint validation."""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import torch
import torch.nn as nn

em=types.ModuleType('e2_base_anchor')
em.OpticalBaseAnchoredE2=type('OpticalBaseAnchoredE2',(),{'VARIANT':'lwpd_e2_base_anchor_v1'})
sys.modules['e2_base_anchor']=em
sys.modules['utils']=types.ModuleType('utils')
dset=types.ModuleType('utils.dataset_os'); dset.validate_manifest_splits=lambda m: None
sys.modules['utils.dataset_os']=dset
fusion=types.ModuleType('utils.fusion_metrics')
fusion.DEFAULT_PROFILE='reference_v2_candidate_with_paper_fmi'
class FakeEvaluator:
    @staticmethod
    def _ssim_pair(a,b):
        return float(1-np.mean((a.astype(np.float64)-b.astype(np.float64))**2)/255**2)
fusion.FusionEvaluator=FakeEvaluator
sys.modules['utils.fusion_metrics']=fusion
spec=importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1] / 'diagnose_e2_information_path.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Fake(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1));self.detail_fusion=lambda od,sd,b: od*.2+sd*.7+b*.1
    def encode_pair(self,o,s,return_diagnostics=False):
        base=(o.expand(-1,64,-1,-1)*.7, o.expand(-1,64,-1,-1)*.3,
              s.expand(-1,64,-1,-1)*.6, s.expand(-1,64,-1,-1)*.4)
        if return_diagnostics: return base,{'optical_gates':torch.ones(1,64,3,128,128)*.5,'sar_gates':torch.ones(1,64,3,128,128)*.6}
        return base
    def fuse_features(self,ob,od,sb,sd):
        b=ob+sb*.1; return b,self.detail_fusion(od,sd,b)
    def reconstruct(self,o,b,d):
        return torch.sigmoid((b.mean(1,keepdim=True)+d.mean(1,keepdim=True))*.4)
    def forward(self,o,s):
        return self.reconstruct(o,*self.fuse_features(*self.encode_pair(o,s)))

rng=np.random.default_rng(42)
o=rng.integers(0,255,size=(256,256),dtype=np.uint8)
s=rng.integers(0,255,size=(256,256),dtype=np.uint8)
model=Fake().eval()
res=probe.probe_one(model,o,s,.05)
assert len(res['source_path_sensitivity'])==4
assert len(res['decoder_detail_scale_sensitivity'])==3
assert all(v['response_derivative_rms_per_unit_gain']>=0 for v in res['source_path_sensitivity'].values())
assert res['source_reconstruction']['optical']['ssim'] is not None
agg=probe.aggregate([{'diagnostics':res}]); assert agg['baseline.ssim_optical']['n']==1
z = torch.randn(1,64,256,256)
assert torch.allclose(sum(probe.split_lmh(z).values()), z, rtol=0, atol=1e-5)
print('PASS synthetic numerical smoke test; real E2 weights and QXS inputs not available')