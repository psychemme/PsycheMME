"""Recompute Table 2 from the complete lightweight per-question score export."""
import collections
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(l) for l in (root / 'results/per_question_scores.jsonl').read_text().splitlines()]
    expected = {r['model']: r for r in json.loads((root / 'results/paper_table_2.json').read_text())['models']}
    grouped = collections.defaultdict(list)
    for row in rows:
        assert all(isinstance(row[f], (int, float)) and math.isfinite(row[f]) and 0 <= row[f] <= 1.0000000001 for f in ('raw', 'verified'))
        grouped[row['model']].append(row)
    assert set(grouped) == set(expected)
    id_sets = []
    output = []
    for model in expected:
        records = grouped[model]
        ids = {r['question_id'] for r in records}
        assert len(records) == len(ids) == 1000
        id_sets.append(ids)
        by_cap = {c: [r for r in records if r['category'] == c] for c in 'GWNTSAMR'}
        assert all(by_cap.values())
        scores = [sum(sum(r[f] for r in v) / len(v) for v in by_cap.values()) / 8 * 100 for f in ('raw', 'verified')]
        scores += [sum(r['verified'] for r in v) / len(v) * 100 for v in by_cap.values()]
        formatted = [f'{score:.2f}' for score in scores]
        assert formatted == expected[model]['scores'], (model, formatted, expected[model]['scores'])
        output.append(dict(model=model, n=1000, scores=formatted))
    assert all(ids == id_sets[0] for ids in id_sets)
    print(json.dumps(dict(status='PASS', models=len(output), rows=len(rows), paper_cells_matched=len(output) * 10, table=output), indent=2))


if __name__ == '__main__':
    main()
