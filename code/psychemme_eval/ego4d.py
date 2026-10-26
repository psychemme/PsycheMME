"""Prepare benchmark clips from the user's official Ego4D download."""
from __future__ import annotations

from contextlib import contextmanager
import configparser
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from .checkpoint import digest
from .data import dump_json, dump_jsonl, read_jsonl, sha256, unique_index


def plan(root):
    root = Path(root)
    episodes = read_jsonl(root / 'metadata/episodes.jsonl')
    media = read_jsonl(root / 'metadata/media.jsonl')
    indexed = unique_index(media, 'episode_id')
    unique_index(episodes, 'episode_id')
    if {x['episode_id'] for x in episodes} != indexed.keys():
        raise ValueError('Episode/media index mismatch.')
    selected = []
    for row in episodes:
        target = indexed[row['episode_id']]
        if row['media_id'] != target['media_id'] or row['source_dataset'] != target['source_dataset']:
            raise ValueError('Episode/media binding mismatch.')
        path = Path(row['media_id'])
        if path.is_absolute() or '..' in path.parts or path.parts[:1] != ('video',):
            raise ValueError('Invalid benchmark media path.')
        if row['source_dataset'] != 'Ego4D':
            continue
        uid = row['source_video_id']
        if str(uuid.UUID(uid)) != uid:
            raise ValueError('Invalid Ego4D video UID.')
        start = row['source_start_seconds']
        end = row['source_end_seconds']
        duration = float(target['duration_seconds'])
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (start, end, duration)) or start < 0 or end <= start or duration <= 0:
            raise ValueError('Invalid Ego4D source interval.')
        if target['source_video_id'] != uid or target['source_start_seconds'] != start or target['source_end_seconds'] != end:
            raise ValueError('Ego4D source interval indexes disagree.')
        selected.append((row, target))
    if not selected:
        raise ValueError('No Ego4D episodes in the benchmark index.')
    return selected, media


@contextmanager
def credentials():
    """Pass credentials to the official CLI in a private temporary AWS profile."""
    key = os.environ.get('EGO4D_ACCESS_KEY_ID', '').strip()
    secret = os.environ.get('EGO4D_SECRET_ACCESS_KEY', '').strip()
    token = os.environ.get('EGO4D_SESSION_TOKEN', '').strip()
    profile = os.environ.get('EGO4D_PROFILE', '').strip()
    if key or secret or token:
        if not key or not secret:
            raise ValueError('Fill both EGO4D_ACCESS_KEY_ID and EGO4D_SECRET_ACCESS_KEY in .env.')
        with tempfile.TemporaryDirectory(prefix='psychemme-ego4d-') as directory:
            path = Path(directory) / 'credentials'
            cfg = configparser.RawConfigParser()
            cfg['psychemme'] = {'aws_access_key_id': key, 'aws_secret_access_key': secret}
            if token:
                cfg['psychemme']['aws_session_token'] = token
            with path.open('w') as stream:
                path.chmod(0o600)
                cfg.write(stream)
            env = {**os.environ, 'AWS_SHARED_CREDENTIALS_FILE': str(path),
                   'AWS_DEFAULT_REGION': 'us-west-1', 'AWS_EC2_METADATA_DISABLED': 'true'}
            yield env, 'psychemme'
    elif profile:
        yield dict(os.environ), profile
    else:
        raise ValueError('Apply at https://ego4d-data.org/docs/start-here/ and fill the Ego4D AWS credentials in .env, or set EGO4D_PROFILE / EGO4D_SOURCE_DIR.')


def source_file(root, uid):
    for relative in (f'v2/full_scale/{uid}.mp4', f'full_scale/{uid}.mp4', f'{uid}.mp4'):
        path = Path(root) / relative
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def fetch_sources(uids, output):
    """Always select explicit UIDs; never fall back to a whole-dataset download."""
    uids = sorted(set(uids))
    if not uids:
        return
    for uid in uids:
        if str(uuid.UUID(uid)) != uid:
            raise ValueError('Invalid Ego4D video UID.')
    if not shutil.which('ego4d'):
        raise ValueError('Install the official downloader: python -m pip install ".[data]"')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    uid_file = output / 'psychemme_uids.txt'
    uid_file.write_text('\n'.join(uids) + '\n')
    with credentials() as (env, profile):
        cmd = ['ego4d', '--output_directory', str(output), '--datasets', 'full_scale',
               '--video_uid_file', str(uid_file), '--version', 'v2',
               '--aws_profile_name', profile, '--no-metadata', '-y']
        # SDK errors can contain signed URLs. Keep credentials and URLs out of logs.
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if completed.returncode:
            raise ValueError('Ego4D download failed. Check the connection and renew expired official credentials, then rerun.')
    if any(source_file(output, uid) is None for uid in uids):
        raise ValueError('Official Ego4D download is incomplete. Rerun with valid credentials.')


def probe(path):
    run = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def crop(source, target, episode, locator):
    duration = float(locator['duration_seconds'])
    original = probe(source)
    video = next((s for s in original['streams'] if s['codec_type'] == 'video'), None)
    if video is None or not any(s['codec_type'] == 'audio' for s in original['streams']):
        raise ValueError('Ego4D source must have video and audio.')
    if (video['width'], video['height']) != (locator['width'], locator['height']):
        raise ValueError('Ego4D source dimensions differ from the benchmark source version.')
    if episode['source_start_seconds'] + duration > float(original['format']['duration']) + 0.001:
        raise ValueError('Ego4D crop exceeds the original source duration.')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ego4d-', dir=target.parent) as directory:
        temporary = Path(directory) / 'clip.mp4'
        # Same source-cut and loudness recipe as materialize_benchmark_media_repairs.py.
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-ss', str(episode['source_start_seconds']), '-i', str(source), '-t', str(duration),
               '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',
               '-movflags', '+faststart', '-threads', '2', str(temporary)]
        subprocess.run(cmd, capture_output=True, check=True)
        result = probe(temporary)
        new_video = next(s for s in result['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in result['streams'] if s['codec_type'] == 'audio')
        if (new_video['width'], new_video['height']) != (locator['width'], locator['height']):
            raise ValueError('Prepared Ego4D dimensions differ from the benchmark.')
        if abs(float(result['format']['duration']) - duration) > 0.0011:
            raise ValueError('Prepared Ego4D duration differs from the benchmark.')
        if str(audio.get('sample_rate')) != str(locator['audio_sample_rate']):
            raise ValueError('Prepared Ego4D audio sample rate differs from the benchmark.')
        temporary.replace(target)
    return {'file_size_bytes': target.stat().st_size, 'media_sha256': sha256(target)}


def prepare(root, selected, media, fetch_other=None):
    root = Path(root)
    source_root = Path(os.environ.get('EGO4D_SOURCE_DIR') or root / '.sources/Ego4D').expanduser()
    local_index = root / 'metadata/media.local.jsonl'
    state_path = root / '.sources/ego4d-preparation.json'
    binding = digest({'episodes': [x[0] for x in selected], 'media': media, 'recipe': sha256(Path(__file__))})
    state = json.loads(state_path.read_text()) if state_path.exists() else {'binding': binding, 'clips': {}}
    if state['binding'] != binding:
        raise ValueError('Ego4D preparation settings changed. Use a new DATA_DIR.')
    prepared = state['clips']
    pending = []
    for episode, locator in selected:
        target = root / 'benchmark/media' / episode['media_id']
        saved = prepared.get(episode['episode_id'])
        if saved and target.is_file() and target.stat().st_size == saved['file_size_bytes'] and sha256(target) == saved['media_sha256']:
            continue
        # Existing Hub-verified release clips can be reused without re-encoding.
        if not saved and target.is_file() and target.stat().st_size == locator['file_size_bytes']:
            prepared[episode['episode_id']] = {'file_size_bytes': target.stat().st_size, 'media_sha256': sha256(target)}
            continue
        pending.append((episode, locator))
    missing = {e['source_video_id'] for e, _ in pending if source_file(source_root, e['source_video_id']) is None}
    if missing:
        print(f'Downloading {len(missing)} selected Ego4D source videos from the official service.', file=sys.stderr, flush=True)
        fetch_sources(missing, source_root)
    if fetch_other:
        fetch_other()
    for index, (episode, locator) in enumerate(pending, 1):
        source = source_file(source_root, episode['source_video_id'])
        target = root / 'benchmark/media' / episode['media_id']
        prepared[episode['episode_id']] = crop(source, target, episode, locator)
        dump_json(state_path, state)
        print(f'Ego4D clips {index}/{len(pending)}', file=sys.stderr, flush=True)
    dump_json(state_path, state)
    rows = [{**row, **prepared.get(row['episode_id'], {})} for row in media]
    for row in rows:
        target = root / 'benchmark/media' / row['media_id']
        if not target.is_file() or target.stat().st_size != row['file_size_bytes']:
            raise ValueError('Benchmark video download is incomplete.')
    dump_jsonl(local_index, rows)
    return {'status': 'PASS', 'ego4d_sources': len({e['source_video_id'] for e, _ in selected}),
            'ego4d_clips': len(selected), 'media_files': len(rows), 'sources': str(local_index)}


def download_benchmark(output, resource, fetch):
    from .checkpoint import run_lock
    root = Path(output)
    if not all(shutil.which(name) for name in ('ffmpeg', 'ffprobe')):
        raise ValueError('Install FFmpeg before downloading and preparing benchmark videos.')
    with run_lock(root / '.sources/download'):
        kwargs = dict(repo_id=resource['repo'], repo_type=resource['type'], revision=resource['revision'], local_dir=str(root))
        fetch(**kwargs, ignore_patterns=['benchmark/media/**'])
        selected, media = plan(root)
        others = ['benchmark/media/' + x['media_id'] for x in media if x['source_dataset'] != 'Ego4D']
        # Fetch the other seven sources only; Ego4D media comes from the official service.
        def fetch_other():
            if others:
                fetch(**kwargs, allow_patterns=others)
        result = prepare(root, selected, media, fetch_other=fetch_other)
    return {**result, 'resource': 'benchmark', 'repo': resource['repo'], 'revision': resource['revision'],
            'directory': str(root), 'media_included': True}
