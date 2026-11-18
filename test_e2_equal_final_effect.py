"""Synthetic tests for equal-final-effect calibration; no QXS data required."""
import unittest
import math

import torch

from test_e2_causal_interventions import TinyNetwork
from diagnose_e2_causal_interventions import projection_context
from diagnose_e2_equal_final_effect import (
    MODES, PAIRS, calibrate, conditions, finite_difference,
    measure_effect, nominal_measurements, paired_results, targets_from_nominal,
)


class EqualFinalEffectTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.net=TinyNetwork().eval()
        self.image=torch.rand(1,1,24,24)
        self.optical=torch.randn(1,8,24,24)
        self.sar=torch.randn(1,8,24,24)
        self.base=torch.randn(1,8,24,24)
        self.ctx=projection_context(
            self.net,(self.base,self.optical,self.base,self.sar),self.base)
        self.baseline=self.net.reconstruct(self.image,self.base,
                                            self.ctx["detail"])
        self.upper=0.025

    def test_global_min_target_and_all_directions_calibrated(self):
        with torch.no_grad():
            nominal=nominal_measurements(
                self.net,self.ctx,self.image,self.base,self.baseline,self.upper)
            self.assertEqual(len(nominal),12)
            targets=targets_from_nominal(nominal,0.9)
            for mode in MODES:
                self.assertAlmostEqual(targets[mode],
                    0.9*min(v[mode] for v in nominal.values()),places=7)
                actuals=[]
                for stage,direction,sign in conditions():
                    def probe(amp):
                        return measure_effect(
                            self.net,self.ctx,self.image,self.base,self.baseline,
                            stage,direction,sign,amp,mode)
                    amplitude,achieved,relative=calibrate(
                        probe,targets[mode],self.upper,steps=18,tolerance=0.02)
                    self.assertGreater(amplitude,0)
                    self.assertLessEqual(amplitude,self.upper)
                    self.assertLessEqual(relative,0.02)
                    actuals.append(achieved)
                self.assertLessEqual(
                    (max(actuals)-min(actuals))/targets[mode],0.04)

    def test_calibrator_no_quality_selection(self):
        def linear_measure(x):
            return 3*x
        amp,actual,error=calibrate(linear_measure,0.3,0.2,steps=18)
        self.assertAlmostEqual(amp,0.1,places=4)
        self.assertAlmostEqual(actual,0.3,places=3)
        self.assertLess(error,0.02)
        with self.assertRaises(ValueError):
            calibrate(linear_measure,1.0,0.2)
        with self.assertRaises(ValueError):
            targets_from_nominal({},0.9)

    def test_paired_schema_and_missing_condition(self):
        rows=[]
        for mode in MODES:
            rows.append(dict(scene="101",match_mode=mode,stage="baseline",
                             direction="none",sign=0,Qabf=0.5,Nabf=0.1,
                             calibration_error=0.,target_effect=0.03))
            for stage,direction,sign in conditions():
                extra=0.01 if direction=="optical" else 0
                rows.append(dict(scene="101",match_mode=mode,stage=stage,
                    direction=direction,sign=sign,target_effect=0.03,
                    calibration_error=0.001,realized_effect=0.03,
                    realized_detail_delta_rms=0.01,
                    image_mae_8bit_float=0.3,Qabf=0.5+extra,
                    Nabf=0.1,source_optical_transfer=0.2,
                    source_sar_transfer=0.8,
                    FMI=0.4,VIF=0.25,SSIM=0.45,MS_SSIM=0.5,SCD=1.2))
        pairs=paired_results(rows,0.02)
        self.assertEqual(len(pairs),len(MODES)*2*2*len(PAIRS))
        self.assertAlmostEqual(pairs[0]["Qabf_difference"],0.01)
        self.assertIsNone(finite_difference({"Qabf":None},{"Qabf":0.5},"Qabf"))
        rows[-1]["calibration_error"]=0.1
        with self.assertRaises(ValueError):
            paired_results(rows,0.02)

    def test_calibration_preserves_frozen_params(self):
        before=[p.detach().clone() for p in self.net.parameters()]
        with torch.no_grad():
            nominal=nominal_measurements(
                self.net,self.ctx,self.image,self.base,self.baseline,self.upper)
            targets=targets_from_nominal(nominal,0.9)
            for mode in MODES:
                stage,direction,sign="late","sar",-1
                calibrate(
                    lambda a:measure_effect(self.net,self.ctx,self.image,
                        self.base,self.baseline,stage,direction,sign,a,mode),
                    targets[mode],self.upper)
        for old,current in zip(before,self.net.parameters()):
            torch.testing.assert_close(old,current)


if __name__=="__main__":
    unittest.main()
