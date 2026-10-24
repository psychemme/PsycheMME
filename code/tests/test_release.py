"""Release interfaces: published media IDs and explicit memory transport."""
from pathlib import Path
import json
import re
import unittest
from psychemme_eval.memory_verification_context import build_memory_context
from psychemme_eval.protocol import enterprise_prompt


class ReleaseInterfaces(unittest.TestCase):
    def test_schema_accepts_published_media_names(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('model_input.schema.json', 'reference.schema.json'):
            schema = json.loads((root / 'schemas' / name).read_text())
            pattern = schema['properties']['media_id']['pattern']
            self.assertIsNotNone(re.fullmatch(pattern, 'video/237f29207e41a2c6.mp4'))
            for invalid in ('video/../clip.mp4', '/video/237f29207e41a2c6.mp4', 'video/clip.mov'):
                self.assertIsNone(re.fullmatch(pattern, invalid))

    def test_memory_enters_model_and_judge_without_reference_fields(self):
        item = {'question_id': 'M1', 'episode_id': 'E1', 'category': 'M',
                'question': 'How should the assistant respond?', 'task_type': 'single_choice',
                'options': ['Respect the preference.', 'Ignore it.'],
                'memory_context': '[M1] The user prefers a quiet environment.',
                'memory_dependency': 'DEPENDENCY_NOT_MODEL_INPUT'}
        gold = {**item, 'reference_answer': 'REFERENCE_NOT_MODEL_INPUT',
                'evidence': [{'content': 'EVIDENCE_NOT_MODEL_INPUT'}]}
        prompt = enterprise_prompt(item, item['options'], 'Transcript.', True)
        self.assertIn(item['memory_context'], prompt)
        for field in ('DEPENDENCY_NOT_MODEL_INPUT', 'REFERENCE_NOT_MODEL_INPUT', 'EVIDENCE_NOT_MODEL_INPUT'):
            self.assertNotIn(field, prompt)
        memory = build_memory_context(gold, item)
        self.assertEqual(memory['frozen_reference_memory'], item['memory_context'])
        self.assertEqual(memory['policy_memory_context'], item['memory_context'])
        absent = {**item, 'memory_context': None}
        self.assertEqual(build_memory_context({**gold, 'memory_context': None}, absent), {})
        self.assertNotIn('Public memory context:', enterprise_prompt(absent, absent['options'], '', True))
