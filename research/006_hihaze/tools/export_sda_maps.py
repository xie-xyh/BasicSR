"""Export SDA density/supervision maps (T_d/T_r/T_s) for visualization.

用法：
    python research/006_hihaze/tools/export_sda_maps.py \
        --ckpt experiments/006a_.../models/net_g_20000.pth \
        --input datasets/OWH/real-test --out research/006_hihaze/sda_maps [--limit 20]

输出每张图一行拼图：input | output | T_d | T_r | T_s（后三者为 jet 伪彩，各自 min-max 归一）。
T_d 高响应 ≈ 结构衰减重（雾浓）；应呈现 Hihaze Fig.3 式的空间对应（机理判据 2）。
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from basicsr.archs.deanet_v4_arch import DEANetV4
from basicsr.utils.img_util import img2tensor

DEFAULT_BACKBONE = ('experiments/004a_DEANetV3-LFM_OWH-Mixed512_L1FFTGrad_60k_B20G1'
                    '/models/net_g_60000.pth')


def load_model(ckpt, backbone, device):
    net = DEANetV4(backbone_ckpt=backbone, base_dim=32, global_block='lfm')
    if ckpt:
        sd = torch.load(ckpt, map_location='cpu')
        key = 'params_ema' if 'params_ema' in sd else 'params'
        net.load_state_dict(sd[key], strict=True)
    return net.to(device).eval()


def colorize(map_t):
    """(H,W) float tensor -> uint8 BGR jet heatmap（min-max 归一）."""
    m = map_t.cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, help='006a 训练 checkpoint；缺省为纯骨架（适配器零初始化）')
    parser.add_argument('--backbone', default=DEFAULT_BACKBONE)
    parser.add_argument('--input', required=True, help='图片文件或目录')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(args.ckpt, args.backbone, device)

    if os.path.isdir(args.input):
        files = sorted(p for p in glob.glob(os.path.join(args.input, '*'))
                       if p.lower().endswith(('.png', '.jpg', '.jpeg')))
    else:
        files = [args.input]
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
        x = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, preds = net.sda_encoder(x)
        out = net(x)[:, :, :h, :w]

        t_d = torch.nn.functional.interpolate(
            preds[0][:, 0:1], size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        t_r = torch.nn.functional.interpolate(
            preds[0][:, 1:2], size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        t_s = torch.nn.functional.interpolate(
            preds[0][:, 2:3], size=(h, w), mode='bilinear', align_corners=False)[0, 0]

        out_img = (out[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
        row = np.concatenate(
            [img, out_img, colorize(t_d), colorize(t_r), colorize(t_s)], axis=1)
        name = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(args.out, f'{name}_sda.png'), row)
    print(f'saved {len(files)} maps to {args.out}')


if __name__ == '__main__':
    main()
