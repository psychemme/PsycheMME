"""First-run setup, downloads and scoring examples."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from .data import CAPABILITIES, dump_json, dump_jsonl, read_jsonl, unique_index, validate_public
from .protocol import enterprise_parse
from .scoring import score_predictions


def download(output, bundle='benchmark', metadata_only=False, downloader=None):
    from .resources import download_resource
    name = 'logs' if bundle == 'replay' else 'benchmark'
    return download_resource(name, output, metadata_only, downloader)


def doctor(data_root=None, mode='core'):
    checks = [{'name': 'Python >=3.10', 'required': True, 'ok': sys.version_info >= (3, 10),
               'found': '.'.join(map(str, sys.version_info[:3]))}]
    for package, expected in [('torch', '2.11.0'), ('transformers', '5.15.1'), ('peft', '0.19.1'),
                              ('qwen-vl-utils', '0.0.14'), ('accelerate', None), ('Pillow', None)]:
        try:
            found = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found = None
        checks.append({'name': package, 'required': mode == 'local', 'found': found,
                       'ok': found is not None and (expected is None or found.split('+')[0] == expected),
                       'expected': expected or 'installed'})
    checks.append({'name': 'ffmpeg', 'required': mode == 'local', 'ok': shutil.which('ffmpeg') is not None})
    if data_root is not None:
        root = Path(data_root)
        required = ['dataset/metadata_only.jsonl', 'results/scoring_inputs.jsonl',
                    'results/per_question_scores.jsonl', 'results/paper_table_2.json']
        checks.append({'name': 'paper replay inputs', 'required': True,
                       'ok': all((root / name).is_file() for name in required)})
    return {'status': 'PASS' if all(c['ok'] for c in checks if c['required']) else 'FAIL',
            'mode': mode, 'checks': checks,
            'scope': 'Dependency checks for the selected runtime.'}


def demo(output):
    root = Path(output)
    if root.exists():
        raise ValueError('Demo output already exists. Choose a fresh --output directory.')
    inputs = []; references = []; predictions = []; judgments = []
    for cap in CAPABILITIES:
        qid = 'DEMO-' + cap
        item = {'question_id': qid, 'episode_id': 'SYNTHETIC', 'media_id': 'video/SYNTHETIC.mp4',
                'category': cap, 'task_type': 'single_choice', 'question': f'Synthetic {cap} contract example: which option matches the stated cue?',
                'options': ['The cue-supported option.', 'The unsupported alternative.'],
                'source_dataset': 'SYNTHETIC SOFTWARE DEMO', 'memory_context': None, 'memory_dependency': None}
        inputs.append(item)
        references.append({**item, 'answer_indices': [0], 'answers': [item['options'][0]],
                           'reference_answer': None, 'evidence': [{'content': 'Synthetic cue explicitly supports option A.'}]})
        if cap == 'R':  # A deliberately missing prediction remains in the denominator.
            continue
        predictions.append({'question_id': qid, 'model_id': 'synthetic-fixture', 'status': 'OK', 'format_valid': True,
                            'prediction': {'answer': 'B' if cap == 'N' else 'A',
                                           **{key: 'The synthetic cue supports this option.' if key == cap else None for key in CAPABILITIES}}})
        judgments.append({'question_id': qid, 'status': 'OK', 'basic_answer_score': 0 if cap == 'N' else 1,
                          'support': 'normal' if cap == 'T' else 'strong',
                          'logic': 'acceptable' if cap == 'T' else 'fully_coherent'})
    scored, summary = score_predictions(inputs, references, predictions, judgments)
    for name, rows in [('inputs', inputs), ('references_EVALUATOR_ONLY', references),
                       ('predictions', predictions), ('judgments', judgments), ('scores', scored)]:
        dump_jsonl(root / (name + '.jsonl'), rows)
    dump_json(root / 'scores.summary.json', summary)
    result = {'status': 'PASS', 'examples': 8, 'raw_macro': round(summary['raw_macro'], 2),
              'verified_macro': round(summary['verified_macro'], 2), 'model_calls': 0, 'judge_calls': 0,
              'scope': 'Synthetic examples of answer and rationale scoring.'}
    dump_json(root / 'DEMO.json', result)
    return result


def import_predictions(inputs, responses, model, output):
    public = validate_public(read_jsonl(inputs))
    supplied = unique_index(read_jsonl(responses))
    if supplied.keys() - public.keys():
        raise ValueError('Foreign response IDs; use the matching public input population.')
    if Path(output).exists():
        raise ValueError('Prediction output already exists. Choose a fresh --output file.')
    rows = []
    for qid, item in public.items():
        source = supplied.get(qid)
        record = {'question_id': qid, 'episode_id': item['episode_id'], 'model_id': model}
        if source is None:
            record.update(status='MISSING', format_valid=False, prediction={})
        else:
            raw = source.get('response')
            if not isinstance(raw, (str, dict)):
                raise ValueError(f'{qid}: response must be a string or a nine-key JSON object.')
            text = raw if isinstance(raw, str) else json.dumps(raw)
            parsed, valid, reason = enterprise_parse(text, item, set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(item['options'])]))
            finish = source.get('finish_reason', 'UNRECORDED')
            if finish not in {'stop', 'UNRECORDED'}:
                valid, reason = False, 'non_stop_finish_reason'
            record.update(status='OK' if valid else 'UNPARSEABLE', format_valid=valid,
                          prediction=parsed, prediction_raw=text, parse_reason=reason, finish_reason=finish)
        rows.append(record)
    dump_jsonl(output, rows)
    return {'records': len(rows), 'valid': sum(row['status'] == 'OK' for row in rows),
            'missing': sum(row['status'] == 'MISSING' for row in rows),
            'invalid': sum(row['status'] == 'UNPARSEABLE' for row in rows),
            'scope': 'Imported responses matched to the declared question population.'}
