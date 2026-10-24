"""Download versioned project resources with the Hugging Face client."""
from __future__ import annotations
import json
from pathlib import Path


def catalog():
    return json.loads(Path(__file__).with_suffix('.json').read_text())


def download_resource(name, output, metadata_only=False, downloader=None):
    resources = catalog()['resources']
    if name not in resources:
        raise ValueError(f'Unknown resource: {name}')
    resource = resources[name]
    if resource['status'] != 'published' or not resource['revision']:
        raise ValueError(f'{name.capitalize()} is planned and is not available yet.')
    if metadata_only and name != 'benchmark':
        raise ValueError('--metadata-only applies to the benchmark.')
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ValueError('Install download support: python -m pip install ".[data]"') from exc
        downloader = snapshot_download
    kwargs = dict(repo_id=resource['repo'], repo_type=resource['type'],
                  revision=resource['revision'], local_dir=str(output))
    if metadata_only:
        kwargs['ignore_patterns'] = ['benchmark/media/**']
    # The client verifies downloaded objects and resumes interrupted transfers.
    path = downloader(**kwargs)
    return {'status': 'PASS', 'resource': name, 'repo': resource['repo'],
            'revision': resource['revision'], 'directory': str(path),
            'media_included': name == 'benchmark' and not metadata_only}
