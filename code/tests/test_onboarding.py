import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psychemme_eval.data import read_jsonl
from psychemme_eval.onboarding import demo, doctor, download, import_predictions
from psychemme_eval.scoring import score_predictions


class FirstRun(unittest.TestCase):
    def test_demo_and_custom_responses_preserve_population_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'demo'
            result = demo(root)
            self.assertEqual((result['raw_macro'], result['verified_macro']), (75.0, 71.18))
            self.assertEqual(result['model_calls'] + result['judge_calls'], 0)
            rows = read_jsonl(root / 'predictions.jsonl')
            responses = root / 'responses.jsonl'
            responses.write_text(''.join(json.dumps({'question_id': r['question_id'], 'response': r['prediction']}) + '\n' for r in rows))
            converted = root / 'imported.jsonl'
            status = import_predictions(root / 'inputs.jsonl', responses, 'my-model', converted)
            self.assertEqual(status['missing'], 1)
            self.assertEqual(read_jsonl(converted)[0]['finish_reason'], 'UNRECORDED')
            scored, summary = score_predictions(read_jsonl(root / 'inputs.jsonl'), read_jsonl(root / 'references_EVALUATOR_ONLY.jsonl'), read_jsonl(converted), read_jsonl(root / 'judgments.jsonl'))
            self.assertEqual(len(scored), 8)
            self.assertEqual(summary['raw_macro'], 75)
            with self.assertRaises(ValueError): demo(root)

    def test_import_rejects_foreign_duplicate_ids_and_retains_invalid_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'demo'; demo(root)
            responses = root / 'responses.jsonl'
            for content in ['{"question_id":"foreign","response":"{}"}\n', '{"question_id":"DEMO-G","response":"{}"}\n' * 2]:
                responses.write_text(content)
                with self.assertRaises(ValueError): import_predictions(root / 'inputs.jsonl', responses, 'model', root / 'unused.jsonl')
            pred = read_jsonl(root / 'predictions.jsonl')[0]
            responses.write_text(json.dumps({'question_id': pred['question_id'], 'response': pred['prediction'], 'finish_reason': 'length'}) + '\n')
            output = root / 'invalid.jsonl'
            result = import_predictions(root / 'inputs.jsonl', responses, 'model', output)
            self.assertEqual((result['invalid'], result['missing']), (1, 7))
            self.assertEqual(read_jsonl(output)[0]['parse_reason'], 'non_stop_finish_reason')

    def test_resource_downloads_select_separate_repositories(self):
        calls = []
        def fetch(**kwargs):
            calls.append(kwargs)
            return kwargs['local_dir']
        with tempfile.TemporaryDirectory() as directory:
            download(directory, 'replay', downloader=fetch)
            self.assertEqual(calls[-1]['repo_id'], 'psychemme/PsycheMME-logs')
            self.assertEqual(calls[-1]['revision'], 'v1.0-paper')
            download(directory, metadata_only=True, downloader=fetch)
            self.assertEqual(calls[-1]['repo_id'], 'psychemme/PsycheMME')
            self.assertEqual(calls[-1]['ignore_patterns'], ['benchmark/media/**'])
            download(directory, downloader=fetch)
            self.assertNotIn('ignore_patterns', calls[-1])

    def test_optional_model_packages_do_not_fail_core_doctor(self):
        with patch('importlib.metadata.version', side_effect=__import__('importlib.metadata', fromlist=['PackageNotFoundError']).PackageNotFoundError):
            self.assertEqual(doctor()['status'], 'PASS')
            self.assertEqual(doctor(mode='local')['status'], 'FAIL')

    def test_planned_resource_never_calls_download(self):
        from psychemme_eval.resources import download_resource
        with patch('builtins.print') as unused:
            with self.assertRaisesRegex(ValueError, 'planned'):
                download_resource('training', 'unused', downloader=lambda **kw: self.fail('network called'))
        self.assertFalse(Path('unused').exists())
