"""Local, content-addressed full-timeline 1 fps / cap64 frame extraction."""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from .data import sha256, dump_json


def frames_1fps(media, cache_root):
    media = Path(media)
    identity = sha256(media)
    target = Path(cache_root) / identity / 'fps1-cap64-jpeg-q3'
    marker = target / 'frames.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        frames = [target / f['name'] for f in record['frames']]
        if record['media_sha256'] == identity and frames and all(p.is_file() and sha256(p) == f['sha256'] for p, f in zip(frames, record['frames'])):
            return frames
        raise ValueError('Changed frame cache: choose a fresh cache directory')
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target, prefix='.extract-') as temporary:
        work = Path(temporary)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(media), '-vf', 'fps=1', '-q:v', '3', '-y', str(work/'%06d.jpg')], check=True)
        frames = sorted(work.glob('*.jpg'))
        if not frames:
            raise ValueError('No frames extracted')
        indices = [round(i * (len(frames)-1) / 63) for i in range(64)] if len(frames) > 64 else list(range(len(frames)))
        # Same indices and JPEG quality as the frozen final comparison.
        selected = []
        for index in indices:
            source = frames[index]
            dest = target / f'{index:06d}.jpg'
            source.replace(dest)
            selected.append(dest)
        dump_json(marker, {'media_sha256': identity, 'sampling': 'ffmpeg fps=1, full timeline, uniform cap64',
                           'frames': [{'name': p.name, 'sha256': sha256(p)} for p in selected]})
        return selected


def resolve_media(item, media_root, locators):
    path = (Path(media_root).resolve() / item['media_id']).resolve()
    if not path.is_relative_to(Path(media_root).resolve()):
        raise ValueError('Media path escapes media root')
    if not path.is_file():
        raise FileNotFoundError('missing_media')
    source = locators.get(item['episode_id'])
    if not source or source.get('media_id', item['media_id']) != item['media_id']:
        raise ValueError('Missing or mismatched episode/media mapping')
    expected = source.get('media_sha256') or source.get('expected_frozen_media_sha256')
    if expected:
        if sha256(path) != expected:
            raise ValueError('media_hash_mismatch')
    else:
        # Released media is downloaded and verified by the Hub client. The public
        # index binds each episode to its video and recorded byte size.
        size = source.get('file_size_bytes')
        if not isinstance(size, int) or size <= 0 or path.stat().st_size != size:
            raise ValueError('Missing or mismatched release video size')
    return path
