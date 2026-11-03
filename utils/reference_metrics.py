"""Versioned paper-formula candidate; see METRIC_AUDIT_20261101.md.

Never changes legacy/paper metrics. Inputs are fixed 8-bit grayscale levels.
FMI implements Haghighat FAST-FMI (DCT, window=3); BSD attribution is in
tests/metric_reference/fmi/LICENSE.md. MATLAB cross-language certification
is separate from Python unit verification; the profile does not imply it.
"""
import warnings

import numpy as np
from scipy.fft import dctn
from scipy.signal import convolve2d, correlate2d
from numpy.lib.stride_tricks import sliding_window_view

from utils.Evaluator import PaperEvaluator


PROFILE = 'reference_v2_candidate'
STATUS = 'python_verified_matlab_parity_pending'


def _entropy_rows(p):
    # real(-p*log2(p)) in the author code, including negative roundoff.
    return -np.sum(p * np.log2(np.where(p != 0, np.abs(p), 1)), axis=1)


def _fmi_pair(a, f):
    """FAST-FMI local marginal/copula calculation, MATLAB column-major order."""
    aw = sliding_window_view(a, (3, 3)).swapaxes(-1, -2).reshape(-1, 9)
    fw = sliding_window_view(f, (3, 3)).swapaxes(-1, -2).reshape(-1, 9)
    results = []
    for start in range(0, len(aw), 512):
        ap, fp = aw[start:start+512].copy(), fw[start:start+512].copy()
        identical = np.all(ap == fp, axis=1)
        for p in (ap, fp):
            lo = p.min(axis=1, keepdims=True)
            span = p.max(axis=1, keepdims=True) - lo
            flat = span[:, 0] == 0
            p -= lo
            p /= np.where(span == 0, 1, span)
            p[flat] = 1
            p /= p.sum(axis=1, keepdims=True)
        ac, fc = ap.cumsum(axis=1), fp.cumsum(axis=1)
        at, ft = ap - ap.mean(axis=1, keepdims=True), fp - fp.mean(axis=1, keepdims=True)
        cross = (at * ft).sum(axis=1)
        denom = np.sqrt((at * at).sum(axis=1) * (ft * ft).sum(axis=1))
        rho = np.divide(cross, denom, out=np.zeros_like(cross), where=(cross != 0) & (denom != 0))
        idx = np.arange(1, 10, dtype=float)
        sa = np.sqrt(np.maximum(ap @ (idx**2) - (ap @ idx)**2, 0))
        sf = np.sqrt(np.maximum(fp @ (idx**2) - (fp @ idx)**2, 0))
        independent_cdf = fc[:, :, None] * ac[:, None, :]
        upper = np.minimum(fc[:, :, None], ac[:, None, :])
        lower = np.maximum(fc[:, :, None] + ac[:, None, :] - 1, 0)
        bound = np.where((rho >= 0)[:, None, None], upper, lower)
        covariance = (bound - independent_cdf).sum(axis=(1, 2))
        corr_bound = np.divide(covariance, sa * sf, out=np.zeros_like(rho), where=(sa * sf) != 0)
        mixture = np.divide(rho, corr_bound, out=np.zeros_like(rho), where=(rho != 0) & (corr_bound != 0))
        bound_pdf = np.diff(np.diff(np.pad(bound, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
        joint = mixture[:, None, None] * bound_pdf + (1-mixture[:, None, None]) * fp[:, :, None] * ap[:, None, :]
        ha, hf = _entropy_rows(ap), _entropy_rows(fp)
        mi = ha + hf - _entropy_rows(joint.reshape(-1, 81))
        value = np.divide(2 * mi, ha + hf, out=np.zeros_like(mi), where=(ha + hf) != 0)
        value[identical] = 1
        results.append(value)
    return float(np.concatenate(results).mean())


def _gaussian_ssim(a, b):
    """Wang 11x11 Gaussian, population covariance, valid spatial pooling."""
    t = np.arange(-5, 6)
    k = np.exp(-(t[:, None]**2 + t[None, :]**2) / (2 * 1.5**2))
    k /= k.sum()
    if min(a.shape) < 11:
        raise ValueError('SSIM needs at least 11x11 at each scale')
    conv = lambda x: convolve2d(x, k, mode='valid')
    ma, mb = conv(a), conv(b)
    va, vb = conv(a*a)-ma*ma, conv(b*b)-mb*mb
    cov = conv(a*b)-ma*mb
    cs = (2*cov + 7.65**2) / (va + vb + 7.65**2)
    lum = (2*ma*mb + 2.55**2) / (ma*ma + mb*mb + 2.55**2)
    return float((lum*cs).mean()), float(cs.mean())


class ReferenceEvaluator(PaperEvaluator):
    """Explicit reference_v2 conventions, not a universal fusion standard.

    Pairwise PSNR pools source MSE; CC/SSIM/MS-SSIM/VIFp use pair means;
    MI and SCD use source sums. Undefined correlations/VIF are NaN, never 1.
    """

    @classmethod
    def metric_vector(cls, image_F, image_A, image_B):
        arrays = [np.asarray(x) for x in (image_F, image_A, image_B)]
        if any(x.ndim != 2 or x.shape != arrays[0].shape for x in arrays):
            raise ValueError('Expected same-size 2D grayscale images')
        if min(arrays[0].shape) < 176:
            raise ValueError('Fixed five-scale reference_v2 requires at least 176x176')
        for x in arrays:
            if not np.isfinite(x).all() or x.min() < 0 or x.max() > 255 or np.any(x != np.rint(x)):
                raise ValueError('reference_v2 expects finite integer levels in [0,255]; no implicit minmax/quantization')
        return super().metric_vector(*(x.astype(np.float64) for x in arrays))

    @staticmethod
    def _correlation(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        a, b = a-a.mean(), b-b.mean()
        denominator = np.sqrt(np.sum(a*a) * np.sum(b*b))
        return float(np.sum(a*b)/denominator) if denominator > 0 else float('nan')

    @classmethod
    def FMI(cls, image_F, image_A, image_B):
        f, a, b = [dctn(np.asarray(x, dtype=float), norm='ortho') for x in (image_F, image_A, image_B)]
        return (_fmi_pair(a, f) + _fmi_pair(b, f))/2

    @classmethod
    def AG(cls, image_F):
        image = np.asarray(image_F, dtype=float)
        gy, gx = np.gradient(image)
        return float(np.sqrt((gx*gx + gy*gy)/2).sum()/((image.shape[0]-1)*(image.shape[1]-1)))

    @classmethod
    def SF(cls, image_F):
        image = np.asarray(image_F, dtype=float)
        return float(np.sqrt(np.mean(np.diff(image, axis=0)**2)+np.mean(np.diff(image, axis=1)**2)))

    @staticmethod
    def Qabf_getQabf(aA, gA, aF, gF):
        # XP formula: equal positive magnitudes have ratio 1. The historical
        # Hong MATLAB port instead assigns gF here; compatibility is legacy.
        denominator = np.maximum(gA, gF)
        ratio = np.divide(np.minimum(gA, gF), denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)
        angular = np.abs(np.abs(aA-aF)-np.pi/2)/(np.pi/2)
        qg = .9994/(1+np.exp(-15*(ratio-.5)))
        qa = .9879/(1+np.exp(-22*(angular-.8)))
        return qg*qa

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        # XP 2000 Eq. (8), using the paper's experiment setting L=1.5.
        ga, aa = cls.Qabf_getArray(np.asarray(image_A, dtype=float))
        gb, ab = cls.Qabf_getArray(np.asarray(image_B, dtype=float))
        gf, af = cls.Qabf_getArray(np.asarray(image_F, dtype=float))
        wa, wb = ga**1.5, gb**1.5
        denominator = np.sum(wa+wb)
        if denominator == 0:
            return float('nan')
        return float(np.sum(cls.Qabf_getQabf(aa,ga,af,gf)*wa + cls.Qabf_getQabf(ab,gb,af,gf)*wb)/denominator)

    @staticmethod
    def _kumar_edges(image):
        image = np.pad(np.asarray(image, dtype=float), 1, mode='reflect')
        vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)/8
        horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)/8
        gv = correlate2d(image, vertical, mode='valid')
        gh = correlate2d(image, horizontal, mode='valid')
        angle = np.arctan(np.divide(gv, gh, out=np.zeros_like(gv), where=gh != 0))
        angle[(gh == 0) & (gv > 0)] = np.pi/2
        angle[(gh == 0) & (gv < 0)] = -np.pi/2
        return np.hypot(gv, gh), angle

    @classmethod
    def Nabf(cls, image_F, image_A, image_B):
        gf, af = cls._kumar_edges(image_F)
        ga, aa = cls._kumar_edges(image_A)
        gb, ab = cls._kumar_edges(image_B)
        def quality(g, a):
            d = np.maximum(g, gf)
            ratio = np.divide(np.minimum(g, gf), d, out=np.zeros_like(d), where=d > 0)
            angular = np.abs(np.abs(a-af)-np.pi/2)*2/np.pi
            qg = .9999/(1+np.exp(-19*(ratio-.5)))
            qa = .9995/(1+np.exp(-22*(angular-.5)))
            return np.sqrt(qg*qa)
        wa, wb = np.where(ga >= 2, ga**1.5, .001), np.where(gb >= 2, gb**1.5, .001)
        mask = (gf > ga) & (gf > gb)
        return float(np.sum(mask*((1-quality(ga, aa))*wa + (1-quality(gb, ab))*wb))/np.sum(wa+wb))

    @staticmethod
    def _ssim_pair(a, b):
        return _gaussian_ssim(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0]

    @classmethod
    def _ms_ssim_pair(cls, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if min(a.shape) < 176:
            raise ValueError('Five-scale MS-SSIM cannot silently drop scales')
        weights = np.array([.0448,.2856,.3001,.2363,.1333])
        terms = []
        for level in range(5):
            ss, cs = _gaussian_ssim(a, b)
            term = ss if level == 4 else cs
            if term < 0:
                # Fractional powers of negative CS have no real-valued paper
                # definition. Keep undefined; do not silently clip to epsilon.
                return float('nan')
            terms.append(term)
            if level != 4:
                # MATLAB imfilter(ones(2)/4,'symmetric','same') uses the
                # upper-left anchor for even kernels, then indices 1:2:end.
                filt = lambda x: correlate2d(np.pad(x, ((0,1),(0,1)), mode='symmetric'), np.ones((2,2))/4, mode='valid')[::2,::2]
                a, b = filt(a), filt(b)
        return float(np.prod(np.asarray(terms)**weights))

    @classmethod
    def compare_viff(cls, ref, dist):
        # Retain LIVE pixel-domain VIF arithmetic; do not replace an
        # undefined zero-reference-information result by perfect quality.
        ref, dist = np.asarray(ref,dtype=float), np.asarray(dist,dtype=float)
        if float(np.var(ref)) == 0:
            return float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                return super().compare_viff(ref, dist)
            except RuntimeWarning:
                return float('nan')
