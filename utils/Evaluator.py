
import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter, uniform_filter
import math
from skimage.metrics import structural_similarity as ssim

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F,image_A)+ssim(image_F,image_B)


class PaperEvaluator(Evaluator):
    """Fusion metrics following Section 4.3.2 of Wang Haolong's thesis.

    Inputs are single-channel images on the 8-bit [0, 255] scale.  This class
    intentionally leaves :class:`Evaluator` unchanged because the original
    CDDFuse README values use sum-based VIF/SSIM and other legacy conventions.

    The thesis specifies the metric equations but not the complete evaluation
    source.  FMI therefore uses the stated whole-image orthonormal DCT followed
    by a 256-bin mutual-information estimate.  MS-SSIM uses the standard
    five-scale Wang weights.  NABF marks a pixel as an artifact when the fused
    gradient is stronger than both source gradients, matching Eq. (4-32/33).
    """

    METRIC_NAMES = (
        "EN",
        "MI",
        "FMI",
        "PSNR",
        "Qabf",
        "Nabf",
        "AG",
        "SF",
        "SD",
        "CC",
        "SCD",
        "SSIM",
        "MS_SSIM",
        "VIF",
    )
    _MS_SSIM_WEIGHTS = np.array(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64
    )
    _EPS = np.finfo(np.float64).eps

    @staticmethod
    def _float_image(image):
        return np.asarray(image, dtype=np.float64)

    @staticmethod
    def _u8_levels(image):
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    @classmethod
    def _mutual_information_bits(cls, image_x, image_y):
        x = cls._u8_levels(image_x).ravel()
        y = cls._u8_levels(image_y).ravel()
        joint = np.histogram2d(
            x, y, bins=256, range=((0, 256), (0, 256))
        )[0]
        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        expected = px[:, None] * py[None, :]
        valid = joint > 0
        return float(np.sum(joint[valid] * np.log2(joint[valid] / expected[valid])))

    @classmethod
    def _quantize_pair(cls, image_x, image_y):
        x = cls._float_image(image_x)
        y = cls._float_image(image_y)
        minimum = min(float(x.min()), float(y.min()))
        maximum = max(float(x.max()), float(y.max()))
        if maximum <= minimum:
            return np.zeros_like(x, dtype=np.uint8), np.zeros_like(y, dtype=np.uint8)
        scale = 255.0 / (maximum - minimum)
        return cls._u8_levels((x - minimum) * scale), cls._u8_levels(
            (y - minimum) * scale
        )

    @classmethod
    def _correlation(cls, image_x, image_y):
        x = cls._float_image(image_x)
        y = cls._float_image(image_y)
        x = x - x.mean()
        y = y - y.mean()
        denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if denominator <= cls._EPS:
            return 1.0 if np.allclose(image_x, image_y) else 0.0
        return float(np.sum(x * y) / denominator)

    @classmethod
    def EN(cls, image_F):
        cls.input_check(image_F)
        values = cls._u8_levels(image_F).ravel()
        probabilities = np.bincount(values, minlength=256).astype(np.float64)
        probabilities /= probabilities.sum()
        nonzero = probabilities > 0
        return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls._mutual_information_bits(
            image_A, image_F
        ) + cls._mutual_information_bits(image_B, image_F)

    @classmethod
    def FMI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        feature_f = dctn(cls._float_image(image_F), norm="ortho")
        feature_a = dctn(cls._float_image(image_A), norm="ortho")
        feature_b = dctn(cls._float_image(image_B), norm="ortho")
        feature_a, feature_f_a = cls._quantize_pair(feature_a, feature_f)
        feature_b, feature_f_b = cls._quantize_pair(feature_b, feature_f)
        return cls._mutual_information_bits(
            feature_a, feature_f_a
        ) + cls._mutual_information_bits(feature_b, feature_f_b)

    @classmethod
    def MSE(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_f = cls._float_image(image_F)
        image_a = cls._float_image(image_A)
        image_b = cls._float_image(image_B)
        return float(
            (np.mean((image_a - image_f) ** 2) + np.mean((image_b - image_f) ** 2))
            / 2.0
        )

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        mse = cls.MSE(image_F, image_A, image_B)
        if mse <= cls._EPS:
            return float("inf")
        return float(10.0 * np.log10((255.0 ** 2) / mse))

    @classmethod
    def AG(cls, image_F):
        cls.input_check(image_F)
        image = cls._float_image(image_F)
        if min(image.shape) < 2:
            return 0.0
        gradient_x = image[:-1, :-1] - image[1:, :-1]
        gradient_y = image[:-1, :-1] - image[:-1, 1:]
        return float(np.mean(np.sqrt((gradient_x ** 2 + gradient_y ** 2) / 2.0)))

    @classmethod
    def SF(cls, image_F):
        cls.input_check(image_F)
        image = cls._float_image(image_F)
        pixels = float(image.size)
        row_frequency = np.sqrt(np.sum((image[:, 1:] - image[:, :-1]) ** 2) / pixels)
        column_frequency = np.sqrt(np.sum((image[1:, :] - image[:-1, :]) ** 2) / pixels)
        return float(np.sqrt(row_frequency ** 2 + column_frequency ** 2))

    @classmethod
    def SD(cls, image_F):
        cls.input_check(image_F)
        return float(np.std(cls._float_image(image_F)))

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (
            cls._correlation(image_A, image_F)
            + cls._correlation(image_B, image_F)
        ) / 2.0

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_f = cls._float_image(image_F)
        image_a = cls._float_image(image_A)
        image_b = cls._float_image(image_B)
        return cls._correlation(image_a, image_f - image_b) + cls._correlation(
            image_b, image_f - image_a
        )

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        g_a, a_a = cls.Qabf_getArray(cls._float_image(image_A))
        g_b, a_b = cls.Qabf_getArray(cls._float_image(image_B))
        g_f, a_f = cls.Qabf_getArray(cls._float_image(image_F))
        q_af = cls.Qabf_getQabf(a_a, g_a, a_f, g_f)
        q_bf = cls.Qabf_getQabf(a_b, g_b, a_f, g_f)
        denominator = np.sum(g_a + g_b)
        if denominator <= cls._EPS:
            return 0.0
        return float(np.sum(q_af * g_a + q_bf * g_b) / denominator)

    @classmethod
    def Nabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        g_a, a_a = cls.Qabf_getArray(cls._float_image(image_A))
        g_b, a_b = cls.Qabf_getArray(cls._float_image(image_B))
        g_f, a_f = cls.Qabf_getArray(cls._float_image(image_F))
        q_af = cls.Qabf_getQabf(a_a, g_a, a_f, g_f)
        q_bf = cls.Qabf_getQabf(a_b, g_b, a_f, g_f)
        artifact_map = g_f > np.maximum(g_a, g_b)
        denominator = np.sum(g_a + g_b)
        if denominator <= cls._EPS:
            return 0.0
        numerator = np.sum(
            artifact_map * ((1.0 - q_af) * g_a + (1.0 - q_bf) * g_b)
        )
        return float(numerator / denominator)

    @classmethod
    def _ssim_pair(cls, image_x, image_y):
        return float(
            ssim(
                cls._float_image(image_x),
                cls._float_image(image_y),
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        )

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (
            cls._ssim_pair(image_A, image_F)
            + cls._ssim_pair(image_B, image_F)
        ) / 2.0

    @classmethod
    def _ms_ssim_pair(cls, image_x, image_y):
        x = cls._float_image(image_x)
        y = cls._float_image(image_y)
        weights = cls._MS_SSIM_WEIGHTS
        levels = min(len(weights), max(1, int(np.floor(np.log2(min(x.shape)))) - 3))
        weights = weights[:levels]
        weights = weights / weights.sum()
        contrast_structure = []

        for level in range(levels):
            mu_x = gaussian_filter(x, sigma=1.5, mode="reflect", truncate=3.5)
            mu_y = gaussian_filter(y, sigma=1.5, mode="reflect", truncate=3.5)
            sigma_x = gaussian_filter(x * x, sigma=1.5, mode="reflect", truncate=3.5) - mu_x ** 2
            sigma_y = gaussian_filter(y * y, sigma=1.5, mode="reflect", truncate=3.5) - mu_y ** 2
            sigma_xy = gaussian_filter(x * y, sigma=1.5, mode="reflect", truncate=3.5) - mu_x * mu_y
            sigma_x = np.maximum(sigma_x, 0.0)
            sigma_y = np.maximum(sigma_y, 0.0)
            c1 = (0.01 * 255.0) ** 2
            c2 = (0.03 * 255.0) ** 2
            luminance = (2.0 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
            cs_map = (2.0 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
            ssim_value = float(np.mean(luminance * cs_map))
            contrast_structure.append(float(np.mean(cs_map)))
            if level < levels - 1:
                x = uniform_filter(x, size=2, mode="reflect")[::2, ::2]
                y = uniform_filter(y, size=2, mode="reflect")[::2, ::2]

        if levels == 1:
            return ssim_value
        cs = np.maximum(np.asarray(contrast_structure[:-1]), cls._EPS)
        return float(
            np.prod(cs ** weights[:-1])
            * max(ssim_value, cls._EPS) ** weights[-1]
        )

    @classmethod
    def MS_SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (
            cls._ms_ssim_pair(image_A, image_F)
            + cls._ms_ssim_pair(image_B, image_F)
        ) / 2.0

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (
            cls.compare_viff(
                cls._float_image(image_A), cls._float_image(image_F)
            )
            + cls.compare_viff(
                cls._float_image(image_B), cls._float_image(image_F)
            )
        ) / 2.0

    @classmethod
    def metric_vector(cls, image_F, image_A, image_B):
        return np.array(
            [
                cls.EN(image_F),
                cls.MI(image_F, image_A, image_B),
                cls.FMI(image_F, image_A, image_B),
                cls.PSNR(image_F, image_A, image_B),
                cls.Qabf(image_F, image_A, image_B),
                cls.Nabf(image_F, image_A, image_B),
                cls.AG(image_F),
                cls.SF(image_F),
                cls.SD(image_F),
                cls.CC(image_F, image_A, image_B),
                cls.SCD(image_F, image_A, image_B),
                cls.SSIM(image_F, image_A, image_B),
                cls.MS_SSIM(image_F, image_A, image_B),
                cls.VIFF(image_F, image_A, image_B),
            ],
            dtype=np.float64,
        )


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB
