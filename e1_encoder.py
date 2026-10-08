"""E1: modality-specific input stems with a shared frozen C0 encoder body."""

import copy

from net import Restormer_Encoder


class E1DualStemEncoder(Restormer_Encoder):
    """Optical/SAR stems share the pretrained Restormer and Base/Detail heads.

    Both stems initially copy C0's patch embedding exactly. This makes E1 an
    exact C0 identity before any gradient updates. The original patch_embed
    remains a frozen checkpoint anchor and is never called during E1 forward.
    """

    def __init__(self):
        super().__init__()
        self.optical_stem = copy.deepcopy(self.patch_embed)
        self.sar_stem = copy.deepcopy(self.patch_embed)

    def initialize_stems_from_c0(self):
        self.optical_stem.load_state_dict(self.patch_embed.state_dict())
        self.sar_stem.load_state_dict(self.patch_embed.state_dict())

    def forward(self, inp_img, modality):
        if modality == "optical":
            stem = self.optical_stem
        elif modality == "sar":
            stem = self.sar_stem
        else:
            raise ValueError("E1 encoder requires modality='optical' or 'sar'")
        features = self.encoder_level1(stem(inp_img))
        base_feature = self.baseFeature(features)
        detail_feature = self.detailFeature(features)
        return base_feature, detail_feature, features
