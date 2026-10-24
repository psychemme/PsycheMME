"""Fixed-population scoring and replay of archived categorical judge factors."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .data import CAPABILITIES, unique_index, validate_pair
from .protocol import enterprise_parse, informative_reasoning_slots, reasoning_coverage

DEFAULT_CONFIG = json.loads(Path(__file__).with_name('scoring_config.json').read_text())


def finite_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Score must be a finite number in [0,1]: {value!r}")
    return float(value)


def weighted_score(basic, coverage, support, logic, config=DEFAULT_CONFIG):
    basic = finite_score(basic)
    if coverage not in config['coverage_multiplier']:
        raise ValueError('Unknown rationale coverage label')
    if support not in config['support_multiplier'] or logic not in config['logic_multiplier']:
        raise ValueError('Unknown support/logic label')
    value = basic * config['coverage_multiplier'][coverage] * config['support_multiplier'][support] * config['logic_multiplier'][logic]
    if config['normalization']['enabled']:
        value /= config['normalization']['max_multiplier']
    return value


def aggregate(scored, population):
    expected = unique_index(population)
    actual = unique_index(scored)
    if expected.keys() != actual.keys() or not expected:
        raise ValueError('Scored population must exactly match the declared nonempty population')
    by = {cap: [] for cap in CAPABILITIES}
    for qid, row in actual.items():
        cap = expected[qid]['category']
        if cap not in by or cap != row['category']:
            raise ValueError('Capability drift')
        by[cap].append(row)
    summaries = {cap: {'n': len(values), **{name: 100 * sum(finite_score(x[name]) for x in values) / len(values) for name in ('raw', 'verified')}} for cap, values in by.items() if values}
    complete = len(summaries) == 8
    return {'n': len(actual), 'all_eight_capabilities_present': complete,
            'raw_macro': sum(x['raw'] for x in summaries.values()) / 8 if complete else None,
            'verified_macro': sum(x['verified'] for x in summaries.values()) / 8 if complete else None,
            'present_capability_macro_exploratory': {name: sum(v[name] for v in summaries.values()) / len(summaries) for name in ('raw', 'verified')},
            'capabilities': summaries, 'missing_capabilities': [c for c in CAPABILITIES if c not in summaries]}


def replay_factors(factors, population):
    """Recompute with persisted basic credit and labels, not persisted final scores."""
    expected = unique_index(population)
    grouped = {}
    for row in factors:
        grouped.setdefault(row['model'], []).append(row)
    outputs = []; summaries = []
    for name, records in grouped.items():
        indexed = unique_index(records)
        if indexed.keys() != expected.keys():
            raise ValueError(f'Factor population mismatch for {name}')
        scored = []
        for qid, row in indexed.items():
            if row['category'] != expected[qid]['category']:
                raise ValueError('Factor capability mismatch')
            basic = finite_score(row['basic_answer_score'])
            valid = row['inference_status'] == 'OK' and row['format_valid'] is True
            if not valid and basic != 0:
                raise ValueError('Invalid inference has nonzero basic credit')
            if not valid and row['reasoning_coverage'] != 'missing':
                raise ValueError('Invalid inference has rationale coverage')
            if (row['judge_status'] != 'OK' or not valid) and (row['support'] != 'insufficient' or row['logic'] != 'unevaluable'):
                raise ValueError('Failure must have zero verification factors')
            scored.append({'model': name, 'question_id': qid, 'category': row['category'], 'raw': basic,
                           'verified': weighted_score(basic, row['reasoning_coverage'], row['support'], row['logic'])})
        outputs.extend(scored)
        summaries.append({'model': name, **aggregate(scored, population)})
    return outputs, summaries


def score_predictions(public, references, predictions, judgments):
    """Compute choice correctness; open correctness requires a successful judge."""
    inputs, truth = validate_pair(public, references)
    pm = unique_index(predictions); jm = unique_index(judgments)
    if pm.keys() - inputs.keys() or jm.keys() - inputs.keys():
        raise ValueError('Foreign prediction/judgment ID')
    out = []
    for qid, item in inputs.items():
        pred = pm.get(qid, {}); judge = jm.get(qid, {})
        parsed, okay, reason = enterprise_parse(json.dumps(pred.get('prediction')), item, set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(item['options'])]))
        valid = pred.get('status') == 'OK' and pred.get('format_valid') is True and okay
        coverage = reasoning_coverage(parsed) if valid else 'missing'
        basic = 0.0
        if valid:
            answer = parsed['answer']; gold = truth[qid]
            if item['task_type'] == 'open':
                if judge.get('status') == 'OK':
                    basic = finite_score(judge['basic_answer_score'])
                    if basic not in {0, 0.5, 1}:
                        raise ValueError('Open score must be 0, 0.5 or 1')
            else:
                letters = {chr(65+i) for i in gold['answer_indices']}
                basic = float(set(answer) == letters) if item['task_type'] == 'multiple_select' else float(answer in letters)
        support, logic = 'insufficient', 'unevaluable'
        if valid and judge.get('status') == 'OK' and coverage == 'present':
            support, logic = judge['support'], judge['logic']
        out.append({'question_id': qid, 'category': item['category'], 'task_type': item['task_type'],
                    'raw': basic, 'verified': weighted_score(basic, coverage, support, logic),
                    'reasoning_coverage': coverage, 'support': support, 'logic': logic,
                    'informative_reasoning_slots': list(informative_reasoning_slots(parsed)) if valid else [],
                    'inference_status': pred.get('status', 'MISSING'), 'format_valid': valid,
                    'parse_reason': reason, 'judge_status': judge.get('status', 'MISSING')})
    return out, aggregate(out, public)
