"""Zero-initialized cross-modal multiscale LL modulation for the E2 backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lwpd import LearnableWaveletPriorDecomposition


LL_CONTEXT_STATS = ('ll_delta_abs_mean', 'll_delta_abs_p95', 'll_delta_saturation',
                    'll_change_rms', 'll_optical_delta_mean', 'll_sar_delta_mean',
                    'll_context_scale1_rms', 'll_context_scale2_rms', 'll_context_scale4_rms')


class CrossModalLLContext(nn.Module):
    def __init__(self, channels=64, hidden_channels=16):
        super().__init__()
        self.phi=nn.Sequential(nn.Conv2d(3*channels,hidden_channels,1),nn.GELU(),
            nn.Conv2d(hidden_channels,hidden_channels,3,padding=1,groups=hidden_channels),nn.GELU())
        self.optical_head=nn.Conv2d(3*hidden_channels,channels,1)
        self.sar_head=nn.Conv2d(3*hidden_channels,channels,1)
        for head in (self.optical_head,self.sar_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self, optical, sar):
        own,other=optical,sar
        optical_context=[];sar_context=[];rms=[]
        for scale in (1,2,4):
            if scale!=1:
                own=F.avg_pool2d(own,2,stride=2,ceil_mode=True,count_include_pad=False)
                other=F.avg_pool2d(other,2,stride=2,ceil_mode=True,count_include_pad=False)
            difference=(own-other).abs()
            a=self.phi(torch.cat((own,other,difference),dim=1))
            b=self.phi(torch.cat((other,own,difference),dim=1))
            rms.append((.5*(a.detach().float().square().mean()+b.detach().float().square().mean())).sqrt())
            optical_context.append(F.interpolate(a,size=optical.shape[-2:],mode='bilinear',align_corners=False))
            sar_context.append(F.interpolate(b,size=sar.shape[-2:],mode='bilinear',align_corners=False))
        a=.5*torch.tanh(self.optical_head(torch.cat(optical_context,dim=1)))
        b=.5*torch.tanh(self.sar_head(torch.cat(sar_context,dim=1)))
        return a,b,rms


class LLContextWaveletDecomposition(LearnableWaveletPriorDecomposition):
    def __init__(self, channels=64, mode='gated', **kwargs):
        if mode!='gated':raise ValueError('LL-context v1 requires the E2 gated mode')
        super().__init__(channels=channels,mode=mode,**kwargs)
        self.ll_context=CrossModalLLContext(channels)

    def _modulate_low_bands(self, optical_ll, sar_ll):
        da,db,rms=self.ll_context(optical_ll,sar_ll)
        pa,pb=self.optical_base(optical_ll),self.sar_base(sar_ll)
        a=(1+self.base_scale*torch.tanh(pa+da))*optical_ll
        b=(1+self.base_scale*torch.tanh(pb+db))*sar_ll
        with torch.no_grad():
            delta=torch.cat((da.detach().float().flatten(),db.detach().float().flatten()))
            old_a=(1+self.base_scale*torch.tanh(pa))*optical_ll
            old_b=(1+self.base_scale*torch.tanh(pb))*sar_ll
            stats=dict(ll_delta_abs_mean=delta.abs().mean(),ll_delta_abs_p95=torch.quantile(delta.abs(),.95),
                ll_delta_saturation=(delta.abs()>=.49).float().mean(),
                ll_change_rms=(.5*((a-old_a).float().square().mean()+(b-old_b).float().square().mean())).sqrt(),
                ll_optical_delta_mean=da.detach().float().mean(),ll_sar_delta_mean=db.detach().float().mean(),
                **{f'll_context_scale{s}_rms':v for s,v in zip((1,2,4),rms)})
        return a,b,stats
