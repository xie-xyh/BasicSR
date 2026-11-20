import argparse
import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

from split_probe import BASE


def collect(options, min_size=None):
    clear = Path(options['dataroot_gt'])
    hazy = Path(options['dataroot_lq'])
    manifest = options.get('meta_info_file')
    names = (Path(manifest).read_text().splitlines() if manifest else
             sorted(path.name for path in clear.iterdir() if path.is_file()))
    if not names or len(names) != len(set(names)):
        raise ValueError(f'Empty split or repeated name: {options["name"]}')
    hashes = set()
    kept = 0
    for name in names:
        target, source = clear / name, hazy / name
        with Image.open(target) as target_image, Image.open(source) as source_image:
            if target_image.size != source_image.size:
                raise ValueError(f'Image size mismatch: {target} / {source}')
            if min_size is None or min(source_image.size) >= min_size:
                kept += 1
        hashes.add(hashlib.sha256(target.read_bytes()).hexdigest())
    return hashes, dict(pairs=len(names), eligible_pairs=kept, distinct_clear_hashes=len(hashes))


def verify(config):
    train = config['datasets']['train']
    official = yaml.safe_load(BASE.read_text())['datasets']
    train_hashes, train_count = collect(train, train.get('gt_size') if train.get('filter_min_size') else None)
    internal = {}
    test = {}
    val_union = set()
    test_union = set()
    for group, options in config['datasets'].items():
        if group.startswith('val'):
            hashes, internal[options['name']] = collect(options)
            val_union.update(hashes)
    for group, options in official.items():
        if group.startswith('val'):
            hashes, test[options['name']] = collect(options)
            test_union.update(hashes)
    if not internal or not test:
        raise ValueError('Missing validation or official test split.')
    result = dict(train=train_count, internal=internal, official=test,
                  exact_overlap=dict(train_internal=len(train_hashes & val_union),
                                     train_official=len(train_hashes & test_union),
                                     internal_official=len(val_union & test_union)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = verify(yaml.safe_load(args.config.read_text()))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    if any(report['exact_overlap'].values()):
        raise SystemExit('Exact overlap detected.')
