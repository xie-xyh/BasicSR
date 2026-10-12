"""Organize the Overwater-Haze (OWH) dataset into the BasicSR datasets layout.

The raw data lives in /share/Overwater-Haze with three synthetic subsets
(Dense / Mist / Moderate, each has haze/ and gt/ folders with paired files)
plus a Real-Test folder (500 real hazy images without GT).

This script creates symlink trees under datasets/OWH/ so that the original
files stay in /share and no disk space is duplicated:

    datasets/OWH/
    ├── Dense/{train,test}/{hazy,clear}     # per-subset splits
    ├── Mist/{train,test}/{hazy,clear}
    ├── Moderate/{train,test}/{hazy,clear}
    ├── Mixed/{train,test}/{hazy,clear}     # three subsets merged,
    │                                       # file names prefixed with subset
    └── real-test/                          # real hazy images (no GT)

The train/test split follows the existing split from the DEA-Net repo
(--split_root, e.g. /home/xieyuhang/Learn/DEA-Net/dataset), so that results
stay comparable with previous experiments. If the split root is missing, a
random 90/10 split is generated with a fixed seed.

Note: the DEA-Net repo copies of Mist/Moderate were RENAMED (e.g.
09048.jpg), while the /share dump uses per-subset serial numbers
(00001.jpg ...). Dense names are identical on both sides. To keep using the
old splits, the script matches renamed files by content (md5) and maps them
back to the /share file names. The mapping is cached in
<out_dir>/.name_map.json.

Usage:
    python scripts/data_preparation/setup_owh.py \
        --share_dir /share/Overwater-Haze \
        --split_root /home/xieyuhang/Learn/DEA-Net/dataset \
        --out_dir datasets/OWH
"""

import argparse
import hashlib
import json
import os
import random
from os import path as osp

SUBSETS = ['Dense', 'Mist', 'Moderate']
HAZY_NAME = 'hazy'
CLEAR_NAME = 'clear'
MAP_CACHE = '.name_map.json'


def list_files(folder, suffix):
    return sorted(v for v in os.listdir(folder) if v.lower().endswith(suffix))


def md5_of_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def build_name_map(subset, args):
    """Map repo split file names -> /share file names for one subset.

    Dense names match directly; Mist/Moderate are matched by md5.
    """
    share_hazy = osp.join(args.share_dir, subset, 'haze')
    repo_hazy_dirs = [
        osp.join(args.split_root, f'OWH-{subset}', 'train', HAZY_NAME),
        osp.join(args.split_root, f'OWH-{subset}', 'test', HAZY_NAME)
    ]

    if all(osp.isdir(d) for d in repo_hazy_dirs):
        repo_names = [n for d in repo_hazy_dirs for n in list_files(d, '.jpg')]
        share_names = set(list_files(share_hazy, '.jpg'))

        if set(repo_names) <= share_names:
            print(f'[{subset}] file names match /share directly')
            return {n: n for n in repo_names}

        print(f'[{subset}] names differ from /share, matching by md5 ...')
        # one md5 may correspond to several identical files; use a queue so
        # that each repo file is assigned a distinct share file
        share_index = {}
        for i, name in enumerate(sorted(share_names)):
            digest = md5_of_file(osp.join(share_hazy, name))
            share_index.setdefault(digest, []).append(name)
            if (i + 1) % 3000 == 0:
                print(f'  hashed {i + 1}/{len(share_names)} share files')

        name_map = {}
        for i, name in enumerate(sorted(repo_names)):
            repo_dir = repo_hazy_dirs[0] if osp.exists(osp.join(repo_hazy_dirs[0], name)) else repo_hazy_dirs[1]
            digest = md5_of_file(osp.join(repo_dir, name))
            assert digest in share_index and len(share_index[digest]) > 0, (
                f'[{subset}] {name}: no matching file found in {share_hazy} by md5')
            name_map[name] = share_index[digest].pop(0)
            if (i + 1) % 3000 == 0:
                print(f'  matched {i + 1}/{len(repo_names)} repo files')
        return name_map

    # no existing split: identity mapping over all share files
    return {n: n for n in list_files(share_hazy, '.jpg')}


def get_split_names(subset, split, args):
    """Return the repo file names of the given subset/split.

    Prefer the existing split from the DEA-Net repo; fall back to a seeded
    random 90/10 split.
    """
    split_dir = osp.join(args.split_root, f'OWH-{subset}', split, HAZY_NAME)
    if args.split_root and osp.isdir(split_dir):
        names = list_files(split_dir, '.jpg')
        print(f'[{subset}/{split}] use existing split: {len(names)} files')
        return names

    all_names = list_files(osp.join(args.share_dir, subset, 'haze'), '.jpg')
    rng = random.Random(f'{subset}-owh')
    shuffled = all_names[:]
    rng.shuffle(shuffled)
    num_test = int(len(shuffled) * 0.1)
    names = shuffled[:num_test] if split == 'test' else shuffled[num_test:]
    print(f'[{subset}/{split}] generate new 90/10 split: {len(names)} files')
    return names


def make_link(src, dst):
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if osp.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def link_subset(subset, split, names, name_map, args, mixed=False):
    """Create symlinks for one subset/split."""
    src_hazy_dir = osp.join(args.share_dir, subset, 'haze')
    src_clear_dir = osp.join(args.share_dir, subset, 'gt')
    root = osp.join(args.out_dir, 'Mixed' if mixed else subset, split)
    dst_hazy_dir = osp.join(root, HAZY_NAME)
    dst_clear_dir = osp.join(root, CLEAR_NAME)

    for name in names:
        share_name = name_map[subset][name]
        dst_name = f'{subset}_{share_name}' if mixed else share_name
        make_link(osp.join(src_hazy_dir, share_name), osp.join(dst_hazy_dir, dst_name))
        make_link(osp.join(src_clear_dir, share_name), osp.join(dst_clear_dir, dst_name))


def main():
    parser = argparse.ArgumentParser(description='Organize OWH dataset into BasicSR layout')
    parser.add_argument('--share_dir', type=str, default='/share/Overwater-Haze', help='raw OWH dataset dir')
    parser.add_argument(
        '--split_root',
        type=str,
        default='/home/xieyuhang/Learn/DEA-Net/dataset',
        help='dir that contains the existing OWH-<Subset> train/test split; empty to re-split')
    parser.add_argument('--out_dir', type=str, default='datasets/OWH', help='output symlink root')
    parser.add_argument('--rebuild_map', action='store_true', help='rebuild the md5 name map even if cached')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load or build the repo-name -> share-name mapping
    cache_path = osp.join(args.out_dir, MAP_CACHE)
    if osp.exists(cache_path) and not args.rebuild_map:
        with open(cache_path, 'r') as f:
            name_map = json.load(f)
        print(f'load cached name map from {cache_path}')
    else:
        name_map = {}
        for subset in SUBSETS:
            name_map[subset] = build_name_map(subset, args)
        with open(cache_path, 'w') as f:
            json.dump(name_map, f, indent=1)
        print(f'save name map to {cache_path}')

    for subset in SUBSETS:
        for split in ['train', 'test']:
            names = get_split_names(subset, split, args)
            link_subset(subset, split, names, name_map, args, mixed=False)
            link_subset(subset, split, names, name_map, args, mixed=True)

    # real hazy images without GT (for SingleImageDataset inference/demo)
    real_src = osp.join(args.share_dir, 'Real-Test')
    if osp.isdir(real_src):
        for name in list_files(real_src, '.jpeg'):
            make_link(osp.join(real_src, name), osp.join(args.out_dir, 'real-test', name))

    # sanity check: every symlink resolves and has a GT counterpart
    print('verifying symlinks ...')
    broken = 0
    for subset in SUBSETS + ['Mixed']:
        for split in ['train', 'test']:
            for kind in [HAZY_NAME, CLEAR_NAME]:
                d = osp.join(args.out_dir, subset, split, kind)
                if not osp.isdir(d):
                    continue
                for name in os.listdir(d):
                    if not osp.exists(osp.join(d, name)):
                        broken += 1
    assert broken == 0, f'{broken} broken symlinks found'

    print('Done. Statistics:')
    for subset in SUBSETS + ['Mixed']:
        for split in ['train', 'test']:
            d = osp.join(args.out_dir, subset, split, HAZY_NAME)
            if osp.isdir(d):
                print(f'  {subset}/{split}/{HAZY_NAME}: {len(list_files(d, ".jpg"))} files')
    real_dir = osp.join(args.out_dir, 'real-test')
    if osp.isdir(real_dir):
        print(f'  real-test: {len(list_files(real_dir, ".jpeg"))} files')


if __name__ == '__main__':
    main()
