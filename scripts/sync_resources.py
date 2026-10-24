#!/usr/bin/env python3
"""Download project resources or check the local resource index."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from psychemme_eval.resources import catalog, download_resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resources', nargs='*', default=['benchmark'],
                        help='benchmark, logs, sft or grpo (default: benchmark)')
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--metadata-only', action='store_true', help='Skip benchmark videos (about 27.6 GB)')
    parser.add_argument('--website', type=Path, help='Sync resource links into a local project website')
    parser.add_argument('--check', action='store_true', help='Validate resource index without network access')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'manifests/resources.json').read_text())
    if registry != catalog():
        parser.error('Update code/psychemme_eval/resources.json from manifests/resources.json.')
    for name, resource in registry['resources'].items():
        if resource['type'] not in {'dataset', 'model'} or resource['status'] not in {'published', 'planned'}:
            parser.error(f'Invalid resource: {name}')
        if resource['status'] == 'published' and not resource['revision']:
            parser.error(f'Missing revision: {name}')
    if args.website:
        links = {'code': 'https://github.com/psychemme/PsycheMME',
                 'dataset': 'https://huggingface.co/datasets/' + registry['resources']['benchmark']['repo'],
                 'models': 'https://huggingface.co/psychemme/models',
                 'logs': 'https://huggingface.co/datasets/' + registry['resources']['logs']['repo']}
        status = {name: 'Available ↗' for name in links}
        outputs = {
            'assets/resources.json': json.dumps(registry, indent=2) + '\n',
            'js/resources.js': '/* Generated from manifests/resources.json. */\n' +
                'window.PSYCHEMME_RESOURCES = Object.freeze(' + json.dumps(links, indent=2) + ');\n\n' +
                'window.PSYCHEMME_RESOURCE_STATUS = Object.freeze(' + json.dumps(status, indent=2) + ');\n',
        }
        for relative, content in outputs.items():
            target = args.website / relative
            if args.check:
                if not target.is_file() or target.read_text() != content:
                    parser.error(f'Website resource index differs: {relative}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
    if args.check or args.website:
        print('Resource index: PASS')
        return
    for name in args.resources:
        if name not in registry['resources']:
            parser.error(f'Unknown resource: {name}')
        resource = registry['resources'][name]
        try:
            result = download_resource(name, args.output / resource['directory'],
                                       args.metadata_only and name == 'benchmark')
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
