import unittest
import warnings

import numpy as np
from scipy.fft import dctn
from scipy.signal import correlate2d
from skimage.metrics import structural_similarity

from utils.Evaluator import PaperEvaluator
from utils.reference_metrics import ReferenceEvaluator as R, _fmi_pair


def scalar_fmi(a, f):
    """Independent small-window translation of the author's CDF construction."""
    vals = []
    for row in range(1, a.shape[0]-1):
        for col in range(1, a.shape[1]-1):
            p = a[row-1:row+2,col-1:col+2].flatten(order='F').copy()
            q = f[row-1:row+2,col-1:col+2].flatten(order='F').copy()
            if np.array_equal(p,q):
                vals.append(1.)
                continue
            def norm(x):
                x = np.ones_like(x) if x.max()==x.min() else (x-x.min())/(x.max()-x.min())
                return x/x.sum()
            p,q = norm(p),norm(q)
            cp,cq = np.cumsum(p),np.cumsum(q)
            cov = np.sum((p-p.mean())*(q-q.mean()))
            den = np.sqrt(np.sum((p-p.mean())**2)*np.sum((q-q.mean())**2))
            rho = cov/den if cov!=0 else 0.
            idx=np.arange(1,10)
            sp=np.sqrt(max(np.sum(idx**2*p)-np.sum(idx*p)**2,0))
            sq=np.sqrt(max(np.sum(idx**2*q)-np.sum(idx*q)**2,0))
            bound=np.zeros((9,9)); independent=np.zeros((9,9))
            for i in range(9):
                for j in range(9):
                    bound[i,j]=min(cq[i],cp[j]) if rho>=0 else max(cq[i]+cp[j]-1,0)
                    independent[i,j]=cq[i]*cp[j]
            cb=np.sum(bound-independent)/(sp*sq) if sp*sq else 0
            mix=rho/cb if cb and rho else 0
            joint=np.zeros((9,9))
            for i in range(9):
                for j in range(9):
                    cell=bound[i,j]-(bound[i-1,j] if i else 0)-(bound[i,j-1] if j else 0)+(bound[i-1,j-1] if i and j else 0)
                    joint[i,j]=mix*cell+(1-mix)*q[i]*p[j]
            ent=lambda x: -sum(v*np.log2(abs(v)) for v in x.flatten() if v!=0)
            hp,hq=ent(p),ent(q)
            vals.append(2*(hp+hq-ent(joint))/(hp+hq))
    return np.mean(vals)


class ReferenceMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(42)
        cls.a=rng.integers(0,256,(256,256)).astype(float)
        cls.b=np.roll(cls.a,1,axis=1)
        cls.f=np.rint((cls.a+cls.b)/2)

    def test_fmi_author_identity_and_local_scalar(self):
        self.assertEqual(R.FMI(self.a,self.a,self.a),1.)
        rng=np.random.default_rng(11)
        for a,f in [(rng.normal(size=(7,8)),rng.normal(size=(7,8))),
                    (np.ones((6,7)),rng.normal(size=(6,7)))]:
            self.assertAlmostEqual(_fmi_pair(a,f),scalar_fmi(a,f),places=10)
        self.assertGreater(PaperEvaluator.FMI(self.a,self.a,self.a),1.)

    def test_en_mi_analytic_levels(self):
        a=np.tile(np.arange(256),(256,1)).astype(float)
        self.assertEqual(R.EN(a),8.)
        self.assertEqual(R.MI(a,a,a),16.)

    def test_psnr_fixed_peak_pooled_mse(self):
        f=np.full((32,32),10.);a=np.zeros_like(f);b=np.full_like(f,20.)
        self.assertAlmostEqual(R.PSNR(f,a,b),20*np.log10(255/10),places=12)

    def test_sd_cc_scd_analytic_formulas(self):
        self.assertAlmostEqual(R.SD(self.a),np.sqrt(np.mean((self.a-self.a.mean())**2)),places=12)
        corr=lambda x,y: np.corrcoef(x.ravel(),y.ravel())[0,1]
        self.assertAlmostEqual(R.CC(self.f,self.a,self.b),(corr(self.f,self.a)+corr(self.f,self.b))/2,places=12)
        self.assertAlmostEqual(R.SCD(self.f,self.a,self.b),corr(self.a,self.f-self.b)+corr(self.b,self.f-self.a),places=12)
        self.assertTrue(np.isnan(R.CC(np.ones((4,4)),np.ones((4,4)),np.ones((4,4)))))

    def test_ag_sf_declared_finite_difference_conventions(self):
        y,x=np.mgrid[:7,:9];a=3*x+4*y
        self.assertAlmostEqual(R.AG(a),np.sqrt(25/2)*63/(6*8),places=12)
        self.assertAlmostEqual(R.SF(a),5.,places=12)

    def test_qabf_equal_ratio_formula_and_legacy_preservation(self):
        a=np.zeros((1,1));g=np.full((1,1),.5)
        expected=.9994/(1+np.exp(-7.5))*.9879/(1+np.exp(-4.4))
        self.assertAlmostEqual(R.Qabf_getQabf(a,g,a,g).item(),expected,places=12)
        self.assertNotAlmostEqual(PaperEvaluator.Qabf_getQabf(a,g,a,g).item(),expected,places=3)
        self.assertAlmostEqual(R.Qabf_getQabf(np.deg2rad(a+89),g,np.deg2rad(a-89),g).item(),R.Qabf_getQabf(np.deg2rad(a),g,np.deg2rad(a+2),g).item(),places=12)
        self.assertAlmostEqual(R.Qabf(self.f,self.a,self.b),R.Qabf(self.f,self.b,self.a),places=12)

    def test_nabf_flat_and_independent_scalar_reference(self):
        a=np.full((32,32),128.);f=np.full((32,32),140.)
        self.assertEqual(R.Nabf(f,a,a),0.)
        # Scalar correlation/quality construction matches the pinned MATLAB.
        arrays=[self.f[:12,:13],self.a[:12,:13],self.b[:12,:13]]
        edges=[]
        for image in arrays:
            pad=np.pad(image,1,mode='reflect');gs=[];angs=[]
            kx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/8
            ky=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/8
            for i,j in np.ndindex(image.shape):
                v=np.sum(pad[i:i+3,j:j+3]*kx);h=np.sum(pad[i:i+3,j:j+3]*ky)
                gs.append(np.hypot(v,h));angs.append(0 if h==v==0 else np.arctan(v/h) if h else np.sign(v)*np.pi/2)
            edges.append((np.array(gs),np.array(angs)))
        numerator=denominator=0.
        for ((gf,af),(ga,aa),(gb,ab)) in zip(*[zip(*edge) for edge in edges]):
            wa=ga**1.5 if ga>=2 else .001;wb=gb**1.5 if gb>=2 else .001
            denominator+=wa+wb
            if gf>ga and gf>gb:
                q=[]
                for g,angle in [(ga,aa),(gb,ab)]:
                    ratio=min(g,gf)/max(g,gf) if g and gf else 0
                    angle=abs(abs(angle-af)-np.pi/2)*2/np.pi
                    q.append(np.sqrt(.9999/(1+np.exp(-19*(ratio-.5)))*.9995/(1+np.exp(-22*(angle-.5)))))
                numerator+=(1-q[0])*wa+(1-q[1])*wb
        self.assertAlmostEqual(R.Nabf(*arrays),numerator/denominator,places=12)

    def test_ssim_against_wang_configured_skimage(self):
        pair=lambda a,b: structural_similarity(a,b,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
        self.assertAlmostEqual(R.SSIM(self.f,self.a,self.b),(pair(self.f,self.a)+pair(self.f,self.b))/2,places=12)

    def test_ms_ssim_fixed_scales_and_identity(self):
        self.assertAlmostEqual(R.MS_SSIM(self.a,self.a,self.a),1.,places=12)
        with self.assertRaises(ValueError):R.MS_SSIM(self.a[:64,:64],self.a[:64,:64],self.a[:64,:64])
        self.assertAlmostEqual(R.MS_SSIM(self.f,self.a,self.b),R.MS_SSIM(self.f,self.b,self.a),places=12)

    def test_vif_live_core_and_undefined_not_perfect(self):
        self.assertAlmostEqual(R.VIFF(self.f,self.a,self.b),PaperEvaluator.VIFF(self.f,self.a,self.b),places=12)
        flat=np.full_like(self.a,128)
        self.assertTrue(np.isnan(R.VIFF(self.f,flat,flat)))

    def test_complete_vector_and_input_contract(self):
        values=R.metric_vector(self.f,self.a,self.b)
        self.assertEqual(values.shape,(14,))
        self.assertTrue(np.isfinite(values).all())
        with self.assertRaises(ValueError):R.metric_vector(self.f/255,self.a/255,self.b/255)


if __name__=='__main__':unittest.main()
