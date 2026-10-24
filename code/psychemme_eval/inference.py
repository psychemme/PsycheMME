"""Portable inference adapters; only public inputs are accepted."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import urllib.request
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
    """Standard HF/PEFT backend; historical custom offload implementation omitted."""
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


def run(public, asr_rows, source_rows, args):
    validate_public(public)
    asr = unique_index(asr_rows, 'episode_id')
    locators = unique_index(source_rows, 'episode_id')
    output = Path(args.output)
    if output.exists() or output.with_suffix('.run.json').exists():
        raise ValueError('Output already exists: use a fresh run path')
    selected = public[:args.limit] if args.limit else public
    if not selected:
        raise ValueError('Empty evaluation population')
    local = None
    if args.backend == 'local-qwen':
        if not args.revision:
            raise ValueError('Local base model revision must be pinned')
        if args.adapter and not args.adapter_revision:
            raise ValueError('Adapter revision must be pinned')
        local = LocalQwen(args.model, args.revision, args.adapter, args.adapter_revision)
    elif not args.base_url:
        raise ValueError('API backend requires --base-url')
    records=[]
    for item in selected:
        record={'question_id':item['question_id'], 'model_id':args.model, 'episode_id':item['episode_id'], 'frame_count_used':0}
        try:
            frames, prompt = build_input(item, asr, args.media_root, args.frame_cache, locators)
            raw, finish = local.generate(frames,prompt) if local else request_api(args.base_url,args.model,frames,prompt,args.api_key_env,args.timeout)
            parsed, valid, reason = enterprise_parse(raw,item,set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(item['options'])]))
            if finish != 'stop':
                valid, reason = False, 'non_stop_finish_reason'
            record.update(prediction=parsed, prediction_raw=raw, format_valid=valid, status='OK' if valid else 'UNPARSEABLE', parse_reason=reason,
                          finish_reason=finish, frame_count_used=len(frames), prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest())
        except Exception as exc:
            # Do not copy API/HTTP exception text, headers, paths or response bodies into public logs.
            record.update(prediction={}, format_valid=False, status='ERROR', error_type=type(exc).__name__)
        records.append(record)
        dump_jsonl(output,records)
    dump_json(output.with_suffix('.run.json'),{
        'backend':args.backend,'model':args.model,'base_revision':args.revision,
        'adapter':args.adapter,'adapter_revision':args.adapter_revision,
        'n':len(selected),'input_ids':[p['question_id'] for p in selected],
        'sampling':'full-timeline 1 fps, uniform cap64','max_new_tokens':768,'temperature':0,'seed':42,
        'thinking':False,'format_guidance':False,'gold_inputs_read':False,
        'scope':'New evaluation of supplied release inputs; not an assertion of historical score equivalence',
    })
    return {'records':len(records),'successes':sum(r['status']=='OK' for r in records),'failures':sum(r['status']!='OK' for r in records)}
