"""Default optical–SAR fusion metric profile."""

from utils.Evaluator import PaperEvaluator
from utils.reference_metrics import ReferenceEvaluator


DEFAULT_PROFILE = "reference_v2_candidate_with_paper_fmi"


class FusionEvaluator(ReferenceEvaluator):
    """ReferenceEvaluator with the original whole-image DCT FMI."""

    @classmethod
    def FMI(cls, image_F, image_A, image_B):
        return PaperEvaluator.FMI(image_F, image_A, image_B)
