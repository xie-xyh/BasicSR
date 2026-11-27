#!/usr/bin/env python3
"""Render standalone HTML/SVG academic figures to crisp high-DPI PNG via headless Chrome.

Usage:
    /Users/xieyh/miniconda3/envs/d2l/bin/python \
        research/014_maritime_sfnet/paper_method_kit/scripts/render_html_to_png.py \
        input.html output.png --width 1420 --height 860 --scale 2.5 --pad 16
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

CHROME_BIN = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def crop_whitespace(img: Image.Image, pad: int = 16, bg_color=(255, 255, 255)) -> Image.Image:
    rgb = img.convert('RGB')
    bg = Image.new('RGB', rgb.size, bg_color)
    diff = ImageChops.difference(rgb, bg)
    # Ignore sub-pixel anti-aliasing noise (< 4)
    diff = diff.point(lambda p: 255 if p > 4 else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(rgb.width, bbox[2] + pad)
    y1 = min(rgb.height, bbox[3] + pad)
    return rgb.crop((x0, y0, x1, y1))


def render(html_path: Path, png_path: Path, width: int, height: int, scale: float, pad: int, dpi: int):
    html_abs = html_path.resolve()
    png_abs = png_path.resolve()
    png_abs.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='chrome_render_') as tmp:
        raw_png = Path(tmp) / 'raw.png'
        user_data = Path(tmp) / 'profile'
        cmd = [
            CHROME_BIN,
            '--headless=new',
            '--disable-gpu',
            '--no-sandbox',
            '--hide-scrollbars',
            f'--user-data-dir={user_data}',
            f'--window-size={width},{height}',
            f'--force-device-scale-factor={scale}',
            f'--screenshot={raw_png}',
            f'file://{html_abs}',
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if not raw_png.is_file():
            raise RuntimeError(
                f'Chrome headless failed (code={res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}'
            )
        img = Image.open(raw_png)
        cropped = crop_whitespace(img, pad=int(round(pad * scale)))
        cropped.save(png_abs, format='PNG', dpi=(dpi, dpi), optimize=True)
        print(f'Rendered {html_abs.name} -> {png_abs.name} ({cropped.width}x{cropped.height} px @ {dpi} DPI)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('html', type=Path)
    parser.add_argument('png', type=Path)
    parser.add_argument('--width', type=int, default=1460)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--scale', type=float, default=2.5)
    parser.add_argument('--pad', type=int, default=14)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    render(args.html, args.png, args.width, args.height, args.scale, args.pad, args.dpi)


if __name__ == '__main__':
    main()
