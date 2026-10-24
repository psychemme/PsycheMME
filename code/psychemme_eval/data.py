"""Strict IDs and public/evaluator boundary; the loader never downloads media."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

CAPABILITIES = tuple("GWNTSAMR")
GOLD_FIELDS = {"answer", "answers", "answer_indices", "reference_answer", "evidence", "reasoning_chain"}


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at line {number}")
                records.append(row)
    return records


def unique_index(records, key="question_id"):
    result = {}
    for row in records:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_public(records):
    unique_index(records)
    for row in records:
        if GOLD_FIELDS.intersection(row):
            raise ValueError(f"Evaluator-only fields in public input {row['question_id']}")
        if row.get("category") not in CAPABILITIES:
            raise ValueError("Unknown capability")
        if row.get("task_type") not in {"single_choice", "multiple_select", "open"}:
            raise ValueError("Unknown task type")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Empty question")
        for field in ('memory_context', 'memory_dependency'):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f'{field} must be text or null')
        if not isinstance(row.get("options"), list) or not all(isinstance(x, str) for x in row['options']):
            raise ValueError("Invalid options")
        if row['task_type'] != 'open' and not 1 <= len(row['options']) <= 26:
            raise ValueError("Choice questions require 1 to 26 options")
        path = Path(row.get("media_id", ""))
        if path.is_absolute() or '..' in path.parts or path.suffix.lower() != '.mp4':
            raise ValueError("media_id must be a relative MP4 path without traversal")
        if not isinstance(row.get('episode_id'), str) or not row['episode_id']:
            raise ValueError('Missing episode ID')
    return unique_index(records)


def validate_pair(public, references):
    inputs = validate_public(public)
    truth = unique_index(references)
    if inputs.keys() != truth.keys():
        raise ValueError("Public/reference ID sets differ")
    for qid, row in inputs.items():
        gold = truth[qid]
        for field, value in row.items():
            if gold.get(field) != value:
                raise ValueError(f"Public/reference mismatch: {qid}/{field}")
        if row['task_type'] == 'open':
            if not isinstance(gold.get('reference_answer'), str) or not gold['reference_answer'].strip():
                raise ValueError(f"Missing open reference: {qid}")
        else:
            indices = gold.get('answer_indices')
            if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)):
                raise ValueError(f"Missing/duplicate answer indices: {qid}")
            if any(type(i) is not int or not 0 <= i < len(row['options']) for i in indices):
                raise ValueError(f"Invalid answer index: {qid}")
            if row['task_type'] == 'single_choice' and len(indices) != 1:
                raise ValueError(f"Invalid single-choice reference: {qid}")
    return inputs, truth


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def dump_jsonl(path, records):
    atomic_write(path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
