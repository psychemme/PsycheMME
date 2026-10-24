"""Prepare a small evaluation using the published question and episode IDs."""
from pathlib import Path
from .data import read_jsonl, dump_jsonl, unique_index, validate_pair
from .resources import catalog


def prepare(root, output, limit, fetcher=None):
    root, output = Path(root), Path(output)
    if limit < 1:
        raise ValueError('Sample size must be positive.')
    public = read_jsonl(root/'benchmark/public/test.jsonl')
    references = read_jsonl(root/'benchmark/ground_truth.jsonl')
    validate_pair(public, references)
    if limit > len(public):
        raise ValueError('Sample size exceeds the benchmark.')
    selected = []
    # Cover capabilities first, then fill in the original file order.
    for cap in 'GWNTSAMR':
        row = next((x for x in public if x['category'] == cap), None)
        if row and len(selected) < limit:
            selected.append(row)
    ids = {x['question_id'] for x in selected}
    selected += [x for x in public if x['question_id'] not in ids][:limit-len(selected)]
    ids = {x['question_id'] for x in selected}
    episodes = {x['episode_id'] for x in selected}
    asr = read_jsonl(root/'benchmark/asr.jsonl')
    sources = read_jsonl(root/'metadata/media.jsonl')
    for rows in (asr, sources):
        if episodes - unique_index(rows, 'episode_id').keys():
            raise ValueError('Sample episode mapping is incomplete.')
    from .checkpoint import bind_output, digest
    bind_output(output/'selection.jsonl', {'inputs':digest(public), 'references':digest(references), 'limit':limit}, True)
    files = {'inputs': selected, 'references': [x for x in references if x['question_id'] in ids],
             'asr': [x for x in asr if x['episode_id'] in episodes],
             'sources': [x for x in sources if x['episode_id'] in episodes]}
    for name, rows in files.items():
        dump_jsonl(output/(name+'.jsonl'), rows)
    if fetcher is None:
        from huggingface_hub import hf_hub_download
        fetcher = hf_hub_download
    resource = catalog()['resources']['benchmark']
    for media in sorted({x['media_id'] for x in selected}):
        fetcher(repo_id=resource['repo'], repo_type='dataset', revision=resource['revision'],
                filename='benchmark/media/'+media, local_dir=str(root))
    return {'questions':len(selected), 'episodes':len(episodes), 'directory':str(output)}
