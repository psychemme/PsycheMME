"""Official source selection, secret handling and clip preparation without cloud keys."""
import importlib.util
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from psychemme_eval import ego4d
from psychemme_eval.data import dump_jsonl, read_jsonl, sha256
from psychemme_eval.media import resolve_media
from psychemme_eval.resources import download_resource

UID = '004a1802-c546-4dcc-86ba-bf1080077017'


def fixture(root):
    episode = {'episode_id': 'E1', 'media_id': 'video/e1.mp4', 'source_dataset': 'Ego4D',
               'source_video_id': UID, 'source_start_seconds': 1., 'source_end_seconds': 2.}
    media = {**episode, 'duration_seconds': '1.000000', 'file_size_bytes': 1,
             'width': 64, 'height': 64, 'audio_sample_rate': '96000'}
    other = {'episode_id': 'E2', 'media_id': 'video/e2.mp4', 'source_dataset': 'EgoCom', 'file_size_bytes': 5}
    dump_jsonl(root / 'metadata/episodes.jsonl', [episode, other])
    dump_jsonl(root / 'metadata/media.jsonl', [media, other])
    return episode, media, other


class OfficialEgo4D(unittest.TestCase):
    def test_partitioned_download_resume_and_unchanged_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); episode, media, other = fixture(root)
            before = (root / 'metadata/media.jsonl').read_bytes()
            source = root / '.sources/Ego4D/v2/full_scale' / (UID + '.mp4')
            source.parent.mkdir(parents=True); source.write_bytes(b'original')
            calls = []
            def fetch(**kw):
                calls.append(kw)
                for f in kw.get('allow_patterns', []):
                    p = root / f; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'other')
                return str(root)
            def crop(src, target, ep, loc):
                self.assertEqual(src, source)
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'prepared')
                return {'file_size_bytes': 8, 'media_sha256': sha256(target)}
            with patch.dict(os.environ, {'EGO4D_SOURCE_DIR': str(root / '.sources/Ego4D')}), patch.object(ego4d, 'crop', side_effect=crop) as cutter:
                result = download_resource('benchmark', root, downloader=fetch)
                self.assertEqual((result['ego4d_sources'], result['ego4d_clips'], result['media_files']), (1, 1, 2))
                self.assertEqual(calls[0]['ignore_patterns'], ['benchmark/media/**'])
                self.assertEqual(calls[1]['allow_patterns'], ['benchmark/media/video/e2.mp4'])
                download_resource('benchmark', root, downloader=fetch)
                self.assertEqual(cutter.call_count, 1)
                # An interrupted or altered generated clip is repaired; successful clips are reused.
                (root / 'benchmark/media/video/e1.mp4').write_bytes(b'altered!')
                download_resource('benchmark', root, downloader=fetch)
                self.assertEqual(cutter.call_count, 2)
            self.assertEqual((root / 'metadata/media.jsonl').read_bytes(), before)
            local = read_jsonl(root / 'metadata/media.local.jsonl')[0]
            target = resolve_media(episode, root / 'benchmark/media', {'E1': local})
            self.assertEqual(target.read_bytes(), b'prepared')
            target.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
                resolve_media(episode, root / 'benchmark/media', {'E1': local})

    def test_credentials_and_selected_cli_never_expose_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            settings = {'EGO4D_ACCESS_KEY_ID': 'example-id', 'EGO4D_SECRET_ACCESS_KEY': 'example-secret',
                        'EGO4D_SESSION_TOKEN': 'example-session', 'EGO4D_PROFILE': ''}
            captured = []
            def run(cmd, **kw):
                self.assertNotIn('example-secret', ' '.join(cmd))
                self.assertEqual(Path(cmd[cmd.index('--video_uid_file')+1]).read_text().split(), [UID])
                self.assertEqual(cmd[cmd.index('--datasets')+1], 'full_scale')
                self.assertIn('--no-metadata', cmd)
                f = Path(kw['env']['AWS_SHARED_CREDENTIALS_FILE']); captured.append(f)
                self.assertEqual(f.stat().st_mode & 0o777, 0o600)
                cfg = configparser.ConfigParser(); cfg.read(f)
                self.assertEqual(cfg['psychemme']['aws_secret_access_key'], 'example-secret')
                self.assertEqual(cfg['psychemme']['aws_session_token'], 'example-session')
                p = root / 'v2/full_scale' / (UID+'.mp4'); p.parent.mkdir(parents=True); p.write_bytes(b'video')
                return subprocess.CompletedProcess(cmd, 0)
            with patch.dict(os.environ, settings), patch('shutil.which', return_value='ego4d'), patch('subprocess.run', side_effect=run) as caller:
                ego4d.fetch_sources([UID, UID], root)
                self.assertEqual(caller.call_count, 1)
                ego4d.fetch_sources([], root)
                self.assertEqual(caller.call_count, 1)
            self.assertFalse(captured[0].exists())
            settings['EGO4D_SECRET_ACCESS_KEY'] = ''
            with patch.dict(os.environ, settings), self.assertRaisesRegex(ValueError, 'both'):
                with ego4d.credentials(): pass

    @unittest.skipUnless(importlib.util.find_spec('ego4d'), 'Official Ego4D CLI required')
    def test_arguments_are_accepted_by_the_official_cli_offline(self):
        from ego4d.cli.config import config_from_args, validate_config
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def run(cmd, **kw):
                with patch.dict(os.environ, kw['env']):
                    cfg = validate_config(config_from_args(cmd[1:]))
                self.assertEqual(cfg.video_uids, {UID})
                self.assertEqual(cfg.datasets, {'full_scale'})
                self.assertFalse(cfg.metadata)
                self.assertEqual(cfg.aws_profile_name, 'psychemme')
                target = root / 'v2/full_scale' / (UID+'.mp4')
                target.parent.mkdir(parents=True); target.write_bytes(b'official-fixture')
                return subprocess.CompletedProcess(cmd, 0)
            with patch.dict(os.environ, {'EGO4D_ACCESS_KEY_ID':'example-id','EGO4D_SECRET_ACCESS_KEY':'example-secret','EGO4D_SESSION_TOKEN':'','EGO4D_PROFILE':''}), patch('shutil.which', return_value='ego4d'), patch('subprocess.run', side_effect=run):
                ego4d.fetch_sources([UID], root)

    def test_missing_credentials_stop_before_other_video_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); calls = []
            def fetch(**kw):
                calls.append(kw)
                return str(root)
            with patch.dict(os.environ, {'EGO4D_ACCESS_KEY_ID':'','EGO4D_SECRET_ACCESS_KEY':'','EGO4D_SESSION_TOKEN':'','EGO4D_PROFILE':'','EGO4D_SOURCE_DIR':str(root/'sources')}), patch('shutil.which', return_value='ego4d'):
                with self.assertRaisesRegex(ValueError, 'Apply at'):
                    download_resource('benchmark', root, downloader=fetch)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('allow_patterns', calls[0])
            self.assertFalse((root/'metadata/media.local.jsonl').exists())

    def test_expired_credentials_and_incomplete_download_stop(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'EGO4D_PROFILE':'test', 'EGO4D_ACCESS_KEY_ID':'', 'EGO4D_SECRET_ACCESS_KEY':'', 'EGO4D_SESSION_TOKEN':''}), patch('shutil.which', return_value='ego4d'):
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'secret-signed-url')):
                with self.assertRaisesRegex(ValueError, 'renew expired') as error:
                    ego4d.fetch_sources([UID], Path(d))
                self.assertNotIn('secret-signed-url', str(error.exception))
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    ego4d.fetch_sources([UID], Path(d))
            with patch('subprocess.run') as caller, self.assertRaises(ValueError):
                ego4d.fetch_sources(['not-a-uuid'], Path(d))
            caller.assert_not_called()

    def test_invalid_paths_and_mismatched_source_indexes_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); episode, media, other = fixture(root)
            for altered in [{**episode, 'media_id': '../escape.mp4'}, {**episode, 'source_end_seconds': 0}, {**episode, 'source_video_id': 'wrong'}]:
                dump_jsonl(root / 'metadata/episodes.jsonl', [altered, other])
                with self.assertRaises(ValueError): ego4d.plan(root)
            dump_jsonl(root / 'metadata/episodes.jsonl', [other])
            with self.assertRaises(ValueError): ego4d.plan(root)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_clip_and_generated_index_work_with_the_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); episode, media, _ = fixture(root)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=30:duration=3', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=3', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(source)], check=True)
            target = root / 'benchmark/media/video/e1.mp4'
            result = ego4d.crop(source, target, episode, media)
            info = ego4d.probe(target)
            self.assertAlmostEqual(float(info['format']['duration']), 1, places=3)
            self.assertEqual({s['codec_type'] for s in info['streams']}, {'video', 'audio'})
            self.assertEqual(resolve_media(episode, root / 'benchmark/media', {'E1': {**media, **result}}), target.resolve())
