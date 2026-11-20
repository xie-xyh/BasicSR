import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

import yaml


ROOT = Path('datasets/OWH')
DENSITIES = ('Dense', 'Mist', 'Moderate')
BASE = Path('options/train/WaveletDehaze/train_WaveletUNet_011b.yml')


def image_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(path, names):
    path.write_text(''.join(f'{name}\n' for name in sorted(names)))
    return str(path)


def prepare(output, val_per_density, stop):
    output.mkdir(parents=True, exist_ok=False)
    test_groups = defaultdict(list)
    for density in DENSITIES:
        for path in sorted((ROOT / density / 'test/clear').iterdir()):
            test_groups[image_digest(path)].append(f'{density}/{path.name}')

    train_root = ROOT / 'Mixed/train/clear'
    groups = defaultdict(list)
    overlap = []
    for path in sorted(train_root.iterdir()):
        digest = image_digest(path)
        if digest in test_groups:
            overlap.append(f'{path.name}\t{",".join(test_groups[digest])}')
        else:
            groups[digest].append(path.name)
    (output / 'exact_test_overlap.tsv').write_text('train\ttest\n' + ''.join(row + '\n' for row in overlap))

    validation = defaultdict(list)
    train_names = []
    ranked = sorted(groups, key=lambda digest: hashlib.sha256(f'owh-011-v1:{digest}'.encode()).digest())
    for digest in ranked:
        group = groups[digest]
        available = {name.split('_', 1)[0] for name in group}
        if any(len(validation[density]) < val_per_density for density in available):
            for density in sorted(available):
                if len(validation[density]) < val_per_density:
                    validation[density].append(next(name for name in group if name.startswith(density + '_')))
        else:
            train_names.extend(group)
    assert all(len(validation[density]) == val_per_density for density in DENSITIES)

    write_configs(output, train_names, validation, 'v1', stop)
    print('Exact test overlaps:', len(overlap))
    print('Training before min-size filter:', dict(Counter(name.split('_', 1)[0] for name in train_names)))
    print('Internal validation:', {density: len(names) for density, names in validation.items()})


def write_configs(output, train_names, validation, version, stop):
    train_meta = save_manifest(output / 'train.txt', train_names)
    val_meta = {density: save_manifest(output / f'val_{density}.txt', validation[density]) for density in DENSITIES}
    config = yaml.safe_load(BASE.read_text())
    train_set = config['datasets']['train']
    train_set['meta_info_file'] = train_meta
    for index, density in enumerate(DENSITIES):
        key = 'val' if index == 0 else f'val_{index + 1}'
        config['datasets'][key] = {
            'name': f'OWH-{density}-internal-{version}', 'type': 'PairedImageDataset', 'io_backend': {'type': 'disk'},
            'dataroot_gt': train_set['dataroot_gt'], 'dataroot_lq': train_set['dataroot_lq'],
            'meta_info_file': val_meta[density],
        }
    config['train']['total_iter'] = stop
    config['val']['save_img'] = False
    config['logger']['use_swanlab'] = False
    config['logger']['save_checkpoint_freq'] = 2000
    variants = {'011b_control': (False, None), '011c_detail_stem': (True, None),
                '011d_multiscale': (False, [1, 2, 3]), '011d_single_scale': (False, [1, 1, 1])}
    for variant, (detail_stem, context_dilations) in variants.items():
        config['name'] = f'probe_{variant}_owh_internal_{version}_{stop}iter'
        config['network_g']['detail_stem'] = detail_stem
        if context_dilations is not None:
            config['network_g']['context_dilations'] = context_dilations
        else:
            config['network_g'].pop('context_dilations', None)
        (output / f'{variant}.yml').write_text(yaml.safe_dump(config, sort_keys=False))


def screened(previous, output, stop):
    def flagged(filename):
        with (previous / filename).open() as stream:
            return {row['query'] for row in csv.DictReader(stream, delimiter='\t')}

    training = set((previous / 'train.txt').read_text().splitlines())
    remove_train = flagged('near_train_vs_eval.tsv')
    remove_val = flagged('near_internal_vs_official.tsv')
    validation = {density: [name for name in (previous / f'val_{density}.txt').read_text().splitlines()
                            if name not in remove_val] for density in DENSITIES}
    output.mkdir(parents=True, exist_ok=False)
    save_manifest(output / 'excluded_near_train.txt', remove_train)
    save_manifest(output / 'excluded_near_val.txt', remove_val)
    write_configs(output, training - remove_train, validation, 'v2', stop)
    print(f'Near-match screening removed {len(remove_train)} train and {len(remove_val)} validation candidates.')
    print(f'{len(training - remove_train)} train before size filter; validation:',
          {density: len(names) for density, names in validation.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--val-per-density', type=int, default=96)
    parser.add_argument('--stop', type=int, default=8000)
    parser.add_argument('--screened-from', type=Path)
    args = parser.parse_args()
    if args.screened_from:
        screened(args.screened_from, args.output, args.stop)
    else:
        prepare(args.output, args.val_per_density, args.stop)
