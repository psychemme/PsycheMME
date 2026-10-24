"""Configuration-bound checkpoints and exclusive run ownership."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from .data import dump_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@contextmanager
def run_lock(output):
    path = Path(str(output) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('This output is already in use by another process.') from exc
        try:
            yield
        finally:
            if __import__('os').name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def bind_output(output, settings, resume):
    output = Path(output)
    manifest = output.with_suffix('.run.json')
    if output.exists() or manifest.exists():
        if not resume:
            raise ValueError('Output exists. Use --resume to continue this run, or choose a new output.')
        if not manifest.is_file() or json.loads(manifest.read_text()).get('binding') != digest(settings):
            raise ValueError('Run settings or inputs changed. Choose a new output directory.')
    else:
        dump_json(manifest, {**settings, 'binding': digest(settings)})
    return manifest
