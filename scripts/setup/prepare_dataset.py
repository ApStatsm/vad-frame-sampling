"""Create a local immutable setup manifest from test-only inputs (user-run)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import LOCAL_KEYS, read_json
from vadfs.paths import project_path
from vadfs.preparation import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path, help='Dataset config JSON')
    parser.add_argument('--local', type=Path, help='Ignored machine-local resource overrides')
    parser.add_argument('--input', choices=('raw', 'official', 'both'), default='both')
    parser.add_argument('--videos-root', type=Path)
    parser.add_argument('--features-root', type=Path)
    args = parser.parse_args()
    config = read_json(project_path(args.dataset))
    resources = dict(config['resources'])
    if args.local:
        local = read_json(project_path(args.local))
        if set(local) - LOCAL_KEYS:
            raise ValueError('Local file may only override machine resources')
        resources.update(local)
    if args.videos_root:
        resources['videos_root'] = str(args.videos_root.expanduser().resolve())
    if args.features_root:
        resources['features_root'] = str(args.features_root.expanduser().resolve())
    path = prepare(config, resources, args.input)
    print(f'Prepared manifest: {path}')
    print('Validated input identity and GT alignment; decoding/model evaluation not performed.')


if __name__ == '__main__':
    main()
