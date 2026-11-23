"""Optical-Base E2 with a dedicated Detail head variant."""

from biship_detail import build_detail
from e2_base_anchor import OpticalBaseAnchoredE2


class OpticalBaseAnchoredBiSHIP(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_base_anchor_biship_v1"

    def __init__(self, method="B7"):
        super().__init__(variant=self.VARIANT)
        self.detail_fusion = build_detail(method)
