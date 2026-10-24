import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from psychemme_eval import video


class VideoExample(unittest.TestCase):
    def args(self, clip, *extra):
        parser = argparse.ArgumentParser()
        video.add_arguments(parser)
        return parser.parse_args(['--video', str(clip), '--question', 'What happens?', *extra])

    def test_invalid_input_and_unpinned_remote_adapter_do_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(video, 'LocalQwen') as loader:
            clip = Path(directory) / 'video.mp4'
            with self.assertRaisesRegex(ValueError, 'not found'):
                video.run(self.args(clip))
            clip.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'adapter-revision'):
                video.run(self.args(clip, '--adapter', 'test/model'))
            with self.assertRaisesRegex(ValueError, 'must not be empty'):
                args = self.args(clip); args.question = ' '
                video.run(args)
            loader.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_real_frame_extraction_with_mock_model_and_saved_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); clip = root / 'video.mp4'
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'color=c=blue:s=32x32:r=4:d=2', '-c:v', 'mpeg4', str(clip)], check=True)
            transcript = root / 'asr.txt'; transcript.write_text('Hello there.')
            output = root / 'runs/answer.json'
            with patch.object(video, 'LocalQwen') as loader:
                def generate(frames, prompt):
                    self.assertEqual(len(frames), 2)
                    self.assertTrue(all(frame.is_file() for frame in frames))
                    self.assertIn('Transcript:\nHello there.', prompt)
                    return 'A blue video.', 'stop'
                loader.return_value.generate.side_effect = generate
                result = video.run(self.args(clip, '--asr', str(transcript), '--output', str(output)))
                loader.assert_called_once_with(video.BASE_MODEL, video.BASE_REVISION, None, None)
            self.assertEqual(result['status'], 'OK')
            self.assertEqual(json.loads(output.read_text())['response'], 'A blue video.')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                video.run(self.args(clip, '--output', str(output)))

    def test_incomplete_completion_is_not_reported_as_ok(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(video, 'frames_1fps', return_value=['frame']), patch.object(video, 'LocalQwen') as loader:
            clip = Path(directory) / 'video.mp4'; clip.write_bytes(b'fixture')
            loader.return_value.generate.return_value = ('truncated', 'length')
            self.assertEqual(video.run(self.args(clip))['status'], 'INCOMPLETE')
