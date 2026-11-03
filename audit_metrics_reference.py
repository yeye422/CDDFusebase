"""Export one synthetic fixture and compare all 14 old/candidate metrics.

No dataset/checkpoint/inference/training is involved. MATLAB is an optional,
separate certification step, and its failure must not be reported as success.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy.io import savemat

from utils.Evaluator import PaperEvaluator
from utils.reference_metrics import ReferenceEvaluator, PROFILE, STATUS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rng=np.random.default_rng(42)
    a=rng.integers(0,256,(256,256)).astype(float)
    b=np.roll(a,1,axis=1);f=np.rint((a+b)/2)
    old=PaperEvaluator.metric_vector(f,a,b)
    candidate=ReferenceEvaluator.metric_vector(f,a,b)
    assert np.isfinite(old).all() and np.isfinite(candidate).all()
    with (args.output_dir/'synthetic_comparison.csv').open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.writer(out);w.writerow(['metric','paper_legacy','reference_v2_candidate','difference'])
        w.writerows(zip(PaperEvaluator.METRIC_NAMES,old,candidate,candidate-old))
    savemat(args.output_dir/'synthetic_inputs.mat',dict(A=a,B=b,F=f,expected=candidate))
    ref=Path('tests/metric_reference')
    hashes={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in ref.rglob('*') if p.is_file() and p.suffix in ('.m','.md','.json')}
    meta=dict(profile=PROFILE,status=STATUS,synthetic_only=True,seed=42,image_shape=[256,256],
              input_policy='integer levels [0,255], float64 arithmetic',numpy_version=np.__version__,scipy_version=scipy.__version__,skimage_version=skimage.__version__,
              reference_files_sha256=hashes,evaluator_sha256=hashlib.sha256(Path('utils/reference_metrics.py').read_bytes()).hexdigest(),
              base_evaluator_sha256=hashlib.sha256(Path('utils/Evaluator.py').read_bytes()).hexdigest(),
              note='No model performance result; no MATLAB success is implied. Legacy files untouched.')
    (args.output_dir/'audit_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Synthetic all-metric audit exported:',args.output_dir.resolve())
    print('MATLAB certification remains pending; run tests/metric_reference/check_matlab_oracle.m on the fixture.')


if __name__=='__main__':main()
