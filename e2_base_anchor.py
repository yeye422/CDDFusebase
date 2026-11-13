"""LWPD E2 with an optical Base anchor and a learned cross-modal correction."""

import torch.nn as nn

from direct_fusion import DirectFusionCDDFuse


class OpticalBaseAnchoredE2(DirectFusionCDDFuse):
    VARIANT = "lwpd_e2_base_anchor_v1"

    def __init__(self, variant=VARIANT):
        if variant not in (OpticalBaseAnchoredE2.VARIANT, self.VARIANT):
            raise ValueError(f"Unsupported Base anchor variant: {variant}")
        super().__init__(variant="lwpd_e2")
        self.variant = variant
        # The last pointwise convolution precedes GELU, which maps zero to zero.
        projection = self.base_fusion.transform[1]
        nn.init.zeros_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail):
        correction = self.base_fusion(optical_base, sar_base)
        fused_base = optical_base + correction
        fused_detail = self.detail_fusion(optical_detail, sar_detail, fused_base)
        return fused_base, fused_detail


def main():
    """Expose the new checkpoint to the existing evaluator for this process."""
    from unittest.mock import patch

    import test_OS

    class EvaluationModel:
        VARIANTS = (*DirectFusionCDDFuse.VARIANTS, OpticalBaseAnchoredE2.VARIANT)

        def __new__(cls, variant):
            if variant == OpticalBaseAnchoredE2.VARIANT:
                return OpticalBaseAnchoredE2()
            return DirectFusionCDDFuse(variant)

    with patch.object(test_OS, "DirectFusionCDDFuse", EvaluationModel):
        test_OS.main()


if __name__ == "__main__":
    main()
