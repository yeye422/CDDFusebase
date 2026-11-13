"""Optical Base anchored E2 with the existing PAC-Mamba Detail residual."""

from torch import nn

from e2_base_anchor import OpticalBaseAnchoredE2
from pac_mamba import DirectionalPhaseSymmetry, PACMambaDetailFusion, PACMambaE2CDDFuse


class AnchoredBaseFusion(nn.Module):
    def __init__(self, correction):
        super().__init__()
        self.correction = correction

    def forward(self, optical, sar):
        return optical + self.correction(optical, sar)


class OpticalBaseAnchoredPACE2(PACMambaE2CDDFuse):
    VARIANT = "lwpd_e2_base_anchor_pac_v1"

    def __init__(self):
        nn.Module.__init__(self)
        # Construct the shared modules in baseline order, with the same seed.
        backbone = OpticalBaseAnchoredE2()
        self.variant = self.VARIANT
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.lwpd = backbone.lwpd
        self.base_fusion = AnchoredBaseFusion(backbone.base_fusion)
        self.detail_fusion = PACMambaDetailFusion()
        self.detail_fusion.keep_fusion = backbone.detail_fusion
        self.phase_evidence = DirectionalPhaseSymmetry(orientations=4)

    def encode_pair(self, optical, sar, evidence=None, return_diagnostics=False):
        # Reconstruction does not call the Detail fusion branch or use PA.
        if evidence is None and not return_diagnostics:
            evidence = {"sar_ps": None}
        return super().encode_pair(optical, sar, evidence, return_diagnostics)


def main():
    from unittest.mock import patch

    import test_OS

    with patch.dict(test_OS.PAC_MODEL_TYPES, {OpticalBaseAnchoredPACE2.VARIANT: OpticalBaseAnchoredPACE2}):
        test_OS.main()


if __name__ == "__main__":
    main()
