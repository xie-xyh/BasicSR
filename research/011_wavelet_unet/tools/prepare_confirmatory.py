import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from split_probe import BASE


def load_config(path):
    return yaml.safe_load(path.read_text())


def recipe(config):
    return {key: value for key, value in config.items() if key not in ('name', 'network_g')}


def prepare(control_path, candidate_path, label, output):
    control = load_config(control_path)
    candidate = load_config(candidate_path)
    if recipe(control) != recipe(candidate):
        raise ValueError('Control and candidate must have an identical training recipe.')
    for key, value in control['datasets'].items():
        if key.startswith('val') and ('meta_info_file' not in value or '-internal-' not in value['name']):
            raise ValueError(f'Expected an internal validation manifest for {key}.')
    if control['path'].get('pretrain_network_g') or control['path'].get('resume_state'):
        raise ValueError('Confirmatory arms must start from scratch.')
    if control['train']['scheduler']['periods'] != [60000]:
        raise ValueError('Expected a 60k cosine schedule.')

    official = {key: value for key, value in load_config(BASE)['datasets'].items() if key.startswith('val')}
    output.mkdir(parents=True, exist_ok=False)
    for phase, options in control['datasets'].items():
        source = Path(options['meta_info_file'])
        frozen = output / f'{phase}_manifest.txt'
        frozen.write_bytes(source.read_bytes())
        for config in (control, candidate):
            config['datasets'][phase]['meta_info_file'] = str(frozen)

    for group, config in (('control', control), ('candidate', candidate)):
        config['name'] = f'{label}_{group}_WaveletUNet_OWHscreened_60k_B20G1'
        config['train']['total_iter'] = 60000
        config['val']['val_freq'] = 4000
        config['logger']['save_checkpoint_freq'] = 4000
        (output / f'{group}_train.yml').write_text(yaml.safe_dump(config, sort_keys=False))
        evaluation = deepcopy(config)
        evaluation['datasets'] = official
        (output / f'{group}_official.yml').write_text(yaml.safe_dump(evaluation, sort_keys=False))

    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.iterdir())}
    source_files = [control_path, candidate_path, Path('basicsr/archs/wavelet_unet_arch.py')]
    fingerprints.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files})
    (output / 'sha256.json').write_text(json.dumps(fingerprints, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('label')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.control, args.candidate, args.label, args.output)
