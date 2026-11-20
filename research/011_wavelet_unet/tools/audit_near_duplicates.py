import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from split_probe import DENSITIES, ROOT


def signature(path):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    thumbnail = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coeffs = cv2.dct(thumbnail.astype(np.float32))[:8, :8].ravel()[1:]
    return np.packbits(coeffs > np.median(coeffs)), thumbnail, image.shape[1] / image.shape[0]


def search(output, queries, references, max_hamming):
    reference_hashes, reference_thumbs, reference_ratios = zip(*(signature(path) for path in references))
    reference_hashes = np.array(reference_hashes)
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(('query', 'reference', 'phash_hamming', 'thumb_rmse', 'aspect_ratio_delta'))
        for offset in range(0, len(queries), 128):
            paths = queries[offset:offset + 128]
            hashes, thumbs, ratios = zip(*(signature(path) for path in paths))
            distances = popcount[np.bitwise_xor(np.array(hashes)[:, None, :], reference_hashes)].sum(axis=2)
            for row, col in zip(*np.where(distances <= max_hamming)):
                rmse = np.sqrt(np.mean((thumbs[row].astype(np.float32) - reference_thumbs[col]) ** 2))
                shape_delta = abs(np.log(ratios[row] / reference_ratios[col]))
                writer.writerow((paths[row].name, references[col], distances[row, col], round(float(rmse), 3),
                                 round(float(shape_delta), 5)))
    print(f'{output.name}: {len(queries)} queries against {len(references)} references.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('--max-hamming', type=int, default=6)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    train_root = ROOT / 'Mixed/train/clear'
    training = [train_root / name for name in (args.protocol / 'train.txt').read_text().splitlines()]
    internal = [train_root / name for density in DENSITIES
                for name in (args.protocol / f'val_{density}.txt').read_text().splitlines()]
    official = [path for density in DENSITIES for path in sorted((ROOT / density / 'test/clear').iterdir())]
    search(args.protocol / 'near_internal_vs_official.tsv', internal, official, args.max_hamming)
    search(args.protocol / 'near_train_vs_eval.tsv', training, official + internal, args.max_hamming)


if __name__ == '__main__':
    main()
