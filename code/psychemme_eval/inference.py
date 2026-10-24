"""Portable inference adapters; only public inputs are accepted."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
import time
import sys
from .data import dump_json, dump_jsonl, sha256, unique_index, validate_public
from .media import frames_1fps, resolve_media
from .protocol import enterprise_prompt, enterprise_parse, transcript


def build_input(item, asr, media_root, cache, locators):
    # Whitelisted public record; no truth file is accepted by this function.
    media = resolve_media(item, media_root, locators)
    frames = frames_1fps(media, cache)
    entry = asr.get(item['episode_id'])
    if entry is None or entry.get('status') not in {'OK', 'NO_SPEECH'}:
        raise ValueError('Missing or failed ASR; no silent fallback')
    text = transcript(entry.get('segments', [])) if entry['status'] == 'OK' else ''
    return frames, enterprise_prompt(item, item['options'], text, True)


def request_api(endpoint, model, frames, prompt, key_env, timeout):
    content = [{'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode()}} for p in frames]
    content.append({'type':'text', 'text':prompt})
    body = {'model':model, 'messages':[{'role':'user', 'content':content}], 'temperature':0,
            'max_tokens':768, 'seed':42, 'chat_template_kwargs':{'enable_thinking':False}}
    headers = {'Content-Type':'application/json'}
    if key_env and os.environ.get(key_env):
        headers['Authorization'] = 'Bearer ' + os.environ[key_env]
    req = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions', json.dumps(body).encode(), headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    result = value['choices'][0]
    return result['message']['content'], result.get('finish_reason', 'UNRECORDED')


class LocalQwen:
    """Multimodal Qwen inference with Hugging Face and PEFT."""
    def __init__(self, base, revision, adapter=None, adapter_revision=None):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        self.torch = torch
        torch.manual_seed(42)
        self.processor = AutoProcessor.from_pretrained(base, revision=revision)
        self.processor.tokenizer.padding_side = 'left'
        model = Qwen3_5ForConditionalGeneration.from_pretrained(base, revision=revision, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='auto')
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision)
        self.model = model.eval()

    def generate(self, frames, prompt):
        from qwen_vl_utils import process_vision_info
        messages = [{'role':'user', 'content':[*[{'type':'image','image':str(p)} for p in frames], {'type':'text','text':prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images, videos, _ = process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size, return_video_kwargs=True)
        if videos:
            raise ValueError('Expected image frames only')
        inputs = self.processor(text=[text], images=images, return_tensors='pt', padding=True, do_sample_frames=False).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=768, do_sample=False, num_return_sequences=1)
        tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        end = next((i for i, token in enumerate(tokens) if token in eos), None)
        tokens = tokens if end is None else tokens[:end+1]
        return self.processor.decode(tokens, skip_special_tokens=True), 'stop' if end is not None else 'length'


def network_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def run(public, asr_rows, source_rows, args):
    from .checkpoint import run_lock
    with run_lock(args.output):
        return _run(public, asr_rows, source_rows, args)


def _run(public, asr_rows, source_rows, args):
    from .checkpoint import bind_output
    from .data import read_jsonl
    validate_public(public)
    asr = unique_index(asr_rows, 'episode_id')
    locators = unique_index(source_rows, 'episode_id')
    output = Path(args.output)
    selected = public[:args.limit] if args.limit else public
    if not selected:
        raise ValueError('Empty evaluation population')
    settings = {
        'backend': args.backend, 'model': args.model, 'base_revision': args.revision,
        'adapter': args.adapter, 'adapter_revision': args.adapter_revision,
        'endpoint': args.base_url if args.backend == 'api' else None,
        'inputs': selected, 'asr': asr_rows, 'sources': source_rows,
        'implementation': {name: sha256(Path(__file__).with_name(name)) for name in
                           ('inference.py', 'protocol.py', 'media.py')},
        'sampling': 'full-timeline 1 fps, uniform cap64', 'max_new_tokens': 768,
        'temperature': 0, 'seed': 42, 'thinking': False,
    }
    # Only the digest of input records is needed in the run manifest.
    from .checkpoint import digest
    for key in ('inputs', 'asr', 'sources'):
        settings[key] = digest(settings[key])
    manifest = bind_output(output, settings, getattr(args, 'resume', False))
    previous = unique_index(read_jsonl(output)) if output.exists() else {}
    ids = {x['question_id'] for x in selected}
    if previous.keys() - ids:
        raise ValueError('Checkpoint contains questions outside this run.')
    for row in previous.values():
        if row.get('status') not in {'OK', 'UNPARSEABLE', 'ERROR'}:
            raise ValueError('Checkpoint contains an unknown prediction status.')
    retry_errors = getattr(args, 'retry_errors', False)
    pending = [item for item in selected if item['question_id'] not in previous or
               (retry_errors and previous[item['question_id']].get('status') == 'ERROR')]
    local = None
    if pending and args.backend == 'local-qwen':
        if not args.revision:
            raise ValueError('Set --revision for the local base model.')
        if args.adapter and not args.adapter_revision:
            raise ValueError('Set --adapter-revision when using an adapter.')
        local = LocalQwen(args.model, args.revision, args.adapter, args.adapter_revision)
    elif pending and args.backend == 'api' and not args.base_url:
        raise ValueError('API backend requires --base-url')
    retries = getattr(args, 'retries', 2)
    for item in pending:
        qid = item['question_id']
        record = {'question_id': qid, 'model_id': args.model,
                  'episode_id': item['episode_id'], 'frame_count_used': 0}
        old = previous.get(qid)
        if old:
            record['previous_attempts'] = [*old.get('previous_attempts', []),
                                         {k: v for k, v in old.items() if k != 'previous_attempts'}]
        try:
            frames, prompt = build_input(item, asr, args.media_root, args.frame_cache, locators)
            for attempt in range(retries + 1):
                try:
                    raw, finish = local.generate(frames, prompt) if local else request_api(
                        args.base_url, args.model, frames, prompt, args.api_key_env, args.timeout)
                    break
                except Exception as exc:
                    if local or not network_error(exc) or attempt == retries:
                        raise
                    time.sleep(min(8, 2 ** attempt))
            parsed, valid, reason = enterprise_parse(raw, item, set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(item['options'])]))
            if finish != 'stop':
                valid, reason = False, 'non_stop_finish_reason'
            record.update(prediction=parsed, prediction_raw=raw, format_valid=valid,
                          status='OK' if valid else 'UNPARSEABLE', parse_reason=reason,
                          finish_reason=finish, frame_count_used=len(frames),
                          prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest())
        except Exception as exc:
            record.update(prediction={}, format_valid=False, status='ERROR',
                          error_type=type(exc).__name__, retryable=network_error(exc))
            if isinstance(exc, urllib.error.HTTPError):
                record['http_status'] = exc.code
        previous[qid] = record
        dump_jsonl(output, [previous[x['question_id']] for x in selected if x['question_id'] in previous])
        print(f"Inference {len(previous)}/{len(selected)}: {qid} {record['status']}", file=sys.stderr, flush=True)
        if record.get('http_status') in {401, 403}:
            break
    records = [previous[x['question_id']] for x in selected if x['question_id'] in previous]
    result = {'records': len(records), 'expected': len(selected),
              'successes': sum(r['status'] == 'OK' for r in records),
              'format_failures': sum(r['status'] == 'UNPARSEABLE' for r in records),
              'errors': sum(r['status'] == 'ERROR' for r in records),
              'pending': len(selected) - len(records), 'output': str(output)}
    result['failures'] = result['format_failures'] + result['errors']
    result['status'] = 'PASS' if not result['errors'] and not result['pending'] else 'INCOMPLETE'
    saved = json.loads(manifest.read_text()); saved['summary'] = result; dump_json(manifest, saved)
    return result
