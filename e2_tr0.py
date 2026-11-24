"""TR0 model entry points sharing the optical Base-anchored LWPD E2 encoder."""

from e2_base_anchor import OpticalBaseAnchoredE2
from tr0_joint_decoder import ProgressiveJointDecoder


class E2ComplementaryLearning(OpticalBaseAnchoredE2):
    def reconstruct_sources(self, optical, sar, encoded):
        optical_base, optical_detail, sar_base, sar_detail = encoded
        return (
            self.reconstruct(optical, optical_base, optical_detail),
            self.reconstruct(sar, sar_base, sar_detail),
        )

    def fuse_and_decode(self, optical_base, optical_detail, sar_base, sar_detail):
        base, detail = self.fuse_features(optical_base, optical_detail, sar_base, sar_detail)
        return self.decoder(None, base, detail)[0]


class E2ProgressiveJointDecoding(E2ComplementaryLearning):
    VARIANT = "lwpd_e2_tr0_pjd_v1"
    EARLY_VARIANT = "lwpd_e2_tr0_pjd_early_v1"

    def __init__(self, *, early=False):
        super().__init__(self.VARIANT)
        self.variant = self.EARLY_VARIANT if early else self.VARIANT
        del self.detail_fusion
        self.decoder = ProgressiveJointDecoder(early=early, restormer=self.decoder)

    def reconstruct_sources(self, optical, sar, encoded):
        optical_base, optical_detail, sar_base, sar_detail = encoded
        return (
            self.decoder(optical_base, optical_detail, None, (True, False)),
            self.decoder(sar_base, None, sar_detail, (False, True)),
        )

    def fuse_and_decode(self, optical_base, optical_detail, sar_base, sar_detail):
        fused_base = optical_base + self.base_fusion(optical_base, sar_base)
        return self.decoder(fused_base, optical_detail, sar_detail)

    def forward(self, optical, sar):
        return self.fuse_and_decode(*self.encode_pair(optical, sar))
