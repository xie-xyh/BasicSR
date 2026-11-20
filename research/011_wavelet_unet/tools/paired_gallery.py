import argparse
import csv
from pathlib import Path

import cv2
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps

from basicsr.data import build_dataset
from basicsr.utils import tensor2img
from paired_eval import load_network, predict


@torch.inference_mode()
def gallery(args):
    control_config = yaml.safe_load(args.control_config.read_text())
    candidate_config = yaml.safe_load(args.candidate_config.read_text())
    control = load_network(control_config, args.control_weights, args.device)
    candidate = load_network(candidate_config, args.candidate_weights, args.device)
    with args.pairs.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=False)
    for phase, options in control_config['datasets'].items():
        if not phase.startswith('val'):
            continue
        split = options['name']
        selected = sorted((row for row in rows if row['dataset'] == split),
                          key=lambda row: float(row['candidate_psnr']) - float(row['control_psnr']))
        examples = selected[:args.count] + selected[-args.count:][::-1]
        dataset = build_dataset(dict(options, phase='val', scale=1))
        indices = {Path(pair['gt_path']).name: index for index, pair in enumerate(dataset.paths)}
        panel = Image.new('RGB', (960, 24 + 188 * len(examples)), 'white')
        draw = ImageDraw.Draw(panel)
        for column, label in enumerate(('Hazy', 'Control', 'Candidate', 'Target')):
            draw.text((240 * column + 3, 4), label, fill='black')
        for line, row in enumerate(examples):
            item = dataset[indices[row['image']]]
            hazy = item['lq'].to(args.device)
            images = (tensor2img(item['lq']), predict(control, hazy), predict(candidate, hazy), tensor2img(item['gt']))
            height = 24 + 188 * line
            for column, image in enumerate(images):
                thumb = ImageOps.contain(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (234, 160))
                panel.paste(thumb, (column * 240 + 3, height))
            delta = float(row['candidate_psnr']) - float(row['control_psnr'])
            draw.text((3, height + 162), f'{row["image"]} | PSNR delta {delta:+.2f} dB', fill='black')
        panel.save(args.output / f'{split}.jpg', quality=94)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('control_config', 'control_weights', 'candidate_config', 'candidate_weights', 'pairs', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--count', type=int, default=2)
    gallery(parser.parse_args())
