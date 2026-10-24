"""Run a local video through the same Qwen backend used by evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from .data import dump_json
from .inference import LocalQwen
from .media import frames_1fps


BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def add_arguments(parser, default_adapter=None):
    parser.add_argument('--video', type=Path, required=True, help='Local video file')
    parser.add_argument('--question', required=True, help='Question about the video')
    parser.add_argument('--asr', type=Path, help='Optional UTF-8 transcript file')
    parser.add_argument('--model', default=BASE_MODEL)
    parser.add_argument('--revision', default=BASE_REVISION)
    parser.add_argument('--adapter', default=default_adapter)
    parser.add_argument('--adapter-revision', help='Pinned revision for a remote adapter')
    parser.add_argument('--output', type=Path, help='Save the generated response as JSON')


def run(args):
    if not args.video.is_file():
        raise ValueError('Video not found. Supply a local file with --video.')
    if not args.question.strip():
        raise ValueError('Question must not be empty.')
    if args.output and args.output.exists():
        raise ValueError('Output already exists. Choose a fresh --output file.')
    if args.adapter and not Path(args.adapter).is_dir() and not args.adapter_revision:
        raise ValueError('Set --adapter-revision when using an adapter from the Hub.')
    prompt = args.question.strip()
    if args.asr:
        transcript = args.asr.read_text(encoding='utf-8').strip()
        if transcript:
            prompt += '\n\nTranscript:\n' + transcript
    with tempfile.TemporaryDirectory(prefix='psychemme-video-') as cache:
        frames = frames_1fps(args.video, cache)
        try:
            model = LocalQwen(args.model, args.revision, args.adapter, args.adapter_revision)
        except ImportError as exc:
            raise ValueError("Install local model dependencies: python -m pip install '.[local-model]'") from exc
        response, finish = model.generate(frames, prompt)
        result = {
            'status': 'OK' if finish == 'stop' else 'INCOMPLETE',
            'model': args.model, 'base_revision': args.revision,
            'adapter': args.adapter, 'adapter_revision': args.adapter_revision,
            'frame_count': len(frames), 'question': args.question,
            'response': response, 'finish_reason': finish,
        }
    if args.output:
        dump_json(args.output, result)
    return result


def main(default_adapter=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, default_adapter)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'OK' else 1)


if __name__ == '__main__':
    main()
