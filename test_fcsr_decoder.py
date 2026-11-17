"""Synthetic CPU tests for the frozen FCSR Decoder diagnostic; no QXS data."""

import unittest

import torch
from torch import nn

from diagnose_fcsr_decoder import (
    BANDS, STAGES, decoder_taps, equal_rms_delta, feature_bands, gradients,
    haar_detail_fraction, mixing_stats, probe_response, rms, signed_transfer,
    source_specific_mask,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = nn.Module()
        decoder.reduce_channel = nn.Conv2d(128,64,1,bias=False)
        decoder.encoder_level2 = nn.Identity()
        decoder.output = nn.Conv2d(64,1,3,padding=1)
        decoder.sigmoid = nn.Sigmoid()
        self.decoder = decoder


class FCSRDecoderTests(unittest.TestCase):

    def test_additive_spatial_bands(self):
        x = torch.randn(1,64,32,32)
        parts = feature_bands(x)
        self.assertEqual(set(parts), set(BANDS))
        torch.testing.assert_close(sum(parts.values()), x, atol=1e-6, rtol=0)

    def test_haar_energy_fraction(self):
        flat = torch.ones(1,2,16,16)
        checkerboard = torch.ones_like(flat)
        checkerboard[..., ::2, 1::2] = -1
        checkerboard[..., 1::2, ::2] = -1
        self.assertAlmostEqual(haar_detail_fraction(flat),0.,places=7)
        self.assertAlmostEqual(haar_detail_fraction(checkerboard),1.,places=7)
        self.assertEqual(haar_detail_fraction(torch.zeros_like(flat)),0.)

    def test_equal_rms_amplitude(self):
        x = torch.randn(1,8,16,16)
        moved = equal_rms_delta(x,0.03)
        self.assertAlmostEqual(rms(moved),0.03,places=6)
        with self.assertRaises(ValueError):
            equal_rms_delta(torch.zeros_like(x),0.03)

    def test_decoder_taps_and_mixer(self):
        torch.manual_seed(8)
        net = TinyModel().eval()
        b = torch.randn(1,64,16,16)
        d = torch.randn(1,64,16,16)
        taps = decoder_taps(net,b,d)
        self.assertEqual(set(taps),set(STAGES))
        self.assertEqual(tuple(taps["image"].shape),(1,1,16,16))
        self.assertEqual(tuple(taps["reduce"].shape),(1,64,16,16))
        mix = mixing_stats(net,b,d)
        self.assertTrue(-1.0001 <= mix["projection_cosine"] <= 1.0001)

    def test_equal_rms_probe_six_conditions(self):
        torch.manual_seed(9)
        net = TinyModel().eval()
        b = torch.randn(1,64,16,16)
        d = torch.randn(1,64,16,16)
        rows = probe_response(net,b,d,fraction=0.015)
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertEqual(row["status"],"ok")
            self.assertAlmostEqual(row["perturb_rms"],0.015*
                ((rms(b)**2+rms(d)**2)/2)**0.5,places=6)
            self.assertTrue(0<=row["image_high_fraction"]<=1.00001)
            self.assertGreaterEqual(row["image_highpass_gain"],0)

    def test_signed_edge_transfer(self):
        g = torch.randn(1,2,32,32)
        mask = torch.ones(1,1,32,32,dtype=torch.bool)
        self.assertAlmostEqual(signed_transfer(g,2*g,mask),2.,places=6)
        self.assertAlmostEqual(signed_transfer(g,-g,mask),-1.,places=6)
        mask.zero_()
        self.assertIsNone(signed_transfer(g,g,mask))

    def test_flat_source_not_declared_edge(self):
        a = torch.zeros(1,2,64,64)
        other = torch.randn_like(a)
        valid = torch.ones(1,1,64,64,dtype=torch.bool)
        selected = source_specific_mask(a,other,valid)
        self.assertEqual(selected.dtype,torch.bool)
        self.assertEqual(int(selected.sum()),0)

    def test_smoothed_gradient_shape(self):
        image = torch.randn(1,1,64,64)
        self.assertEqual(tuple(gradients(image).shape),(1,2,64,64))


if __name__ == "__main__":
    unittest.main()
