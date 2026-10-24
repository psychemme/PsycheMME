from pathlib import Path
import tempfile
import unittest
from psychemme_eval.media import resolve_media


class ReleaseMedia(unittest.TestCase):
    def test_episode_media_binding_and_file_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'video').mkdir()
            (root / 'video/clip.mp4').write_bytes(b'fixture')
            item = {'episode_id': 'E1', 'media_id': 'video/clip.mp4'}
            locator = {'episode_id': 'E1', 'media_id': item['media_id'], 'file_size_bytes': 7}
            self.assertEqual(resolve_media(item, root, {'E1': locator}), (root / item['media_id']).resolve())
            for change in ({'file_size_bytes': 8}, {'media_id': 'video/other.mp4'}, {'file_size_bytes': None}):
                with self.assertRaises(ValueError):
                    resolve_media(item, root, {'E1': {**locator, **change}})
            with self.assertRaises(ValueError):
                resolve_media({**item, 'media_id': '../clip.mp4'}, root, {'E1': locator})
