"""Build side-by-side (hazy | output | GT) comparison images for given names."""
import sys

import cv2
import numpy as np

names = sys.argv[1:] or ['01859', '01589', '07619', '06931', '00917']
outdir = 'research/003_deanet_global/failure_cases'
for n in names:
    lq = cv2.imread(f'datasets/OWH/Dense/test/hazy/{n}.jpg')
    out = cv2.imread(f'results/001_dense_eval/visualization/OWH-Dense-test/{n}_001_dense_eval.png')
    gt = cv2.imread(f'datasets/OWH/Dense/test/clear/{n}.jpg')
    h, w = gt.shape[:2]
    lq, out = cv2.resize(lq, (w, h)), cv2.resize(out, (w, h))
    combo = np.concatenate([lq, out, gt], axis=1)
    cv2.imwrite(f'{outdir}/{n}_hazy_out_gt.jpg', combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(n, gt.shape)
