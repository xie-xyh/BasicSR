"""002 评估工具：LPIPS 计算 + 三方对比拼图（lq | model_a | model_b | gt）。

用法：
    # 对某组结果图算 LPIPS
    python research/002_sdlora/eval_tools.py lpips --restored <结果图目录> --gt datasets/OWH/Dense/test/clear

    # 生成结构检查拼图（001 vs 002 vs 输入/GT）
    python research/002_sdlora/eval_tools.py montage \
        --lq datasets/OWH/Dense/test/hazy --a <001结果目录> --b <002结果目录> \
        --gt datasets/OWH/Dense/test/clear --out research/002_sdlora/montage --n 8
"""

import argparse
import glob
import os
import os.path as osp

import cv2
import numpy as np


def calc_lpips(folder_restored, folder_gt):
    import torch
    import lpips as lpips_lib
    from basicsr.utils import img2tensor

    loss_fn = lpips_lib.LPIPS(net='vgg').cuda()
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    vals = []
    img_list = sorted(glob.glob(osp.join(folder_restored, '*')))
    for img_path in img_list:
        basename = osp.splitext(osp.basename(img_path))[0].split('_')[0]
        gt_path = None
        for ext in ('.png', '.jpg'):
            p = osp.join(folder_gt, basename + ext)
            if osp.exists(p):
                gt_path = p
                break
        if gt_path is None:
            continue
        img = img2tensor(cv2.imread(img_path), bgr2rgb=True, float32=True)
        gt = img2tensor(cv2.imread(gt_path), bgr2rgb=True, float32=True)
        img = (img.unsqueeze(0).cuda() - torch.tensor(mean).view(1, 3, 1, 1).cuda()) / torch.tensor(std).view(1, 3, 1, 1).cuda()
        gt = (gt.unsqueeze(0).cuda() - torch.tensor(mean).view(1, 3, 1, 1).cuda()) / torch.tensor(std).view(1, 3, 1, 1).cuda()
        with torch.no_grad():
            vals.append(loss_fn(img, gt).item())
    print(f'LPIPS ({len(vals)} imgs): {np.mean(vals):.4f}')
    return np.mean(vals)


def _find(folder, basename):
    for ext in ('.png', '.jpg'):
        hits = glob.glob(osp.join(folder, basename + ext)) or glob.glob(osp.join(folder, basename + '_*' + ext))
        if hits:
            return hits[0]
    return None


def montage(folder_lq, folder_a, folder_b, folder_gt, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    names = [osp.splitext(osp.basename(p))[0] for p in sorted(glob.glob(osp.join(folder_lq, '*')))[:n]]
    for name in names:
        tiles = []
        for folder in (folder_lq, folder_a, folder_b, folder_gt):
            p = _find(folder, name)
            tiles.append(cv2.imread(p) if p else np.zeros((10, 10, 3), np.uint8))
        h = min(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        labels = ['hazy-in', 'A(001)', 'B(002)', 'GT']
        for t, lb in zip(tiles, labels):
            cv2.putText(t, lb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out = cv2.hconcat(tiles)
        cv2.imwrite(osp.join(out_dir, f'{name}_cmp.png'), out)
    print(f'montage saved to {out_dir} ({len(names)} images)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('lpips')
    p.add_argument('--restored', required=True)
    p.add_argument('--gt', required=True)

    p = sub.add_parser('montage')
    p.add_argument('--lq', required=True)
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--gt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=8)

    args = parser.parse_args()
    if args.cmd == 'lpips':
        calc_lpips(args.restored, args.gt)
    else:
        montage(args.lq, args.a, args.b, args.gt, args.out, args.n)
