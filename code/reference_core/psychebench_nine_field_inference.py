#!/usr/bin/env python3
"""Final-1000 inference only: emit answer plus eight concise rationale slots.

This is a narrow copy of the proven V3 inference path. It does not judge,
score, rank, filter, or mutate the frozen benchmark.
"""
from __future__ import annotations

import argparse, base64, concurrent.futures, json, os, random, re, shutil, subprocess, threading, time, urllib.error, urllib.request
from pathlib import Path
from PIL import Image, ImageOps

try:
    from .reasoning_verified_contract import (
        OUTPUT_FIELDS,
        PROTOCOL_VERSION,
        REASONING_FIELDS,
        informative_reasoning_slots,
    )
except ImportError:
    from reasoning_verified_contract import (
        OUTPUT_FIELDS,
        PROTOCOL_VERSION,
        REASONING_FIELDS,
        informative_reasoning_slots,
    )

FIELDS = list(OUTPUT_FIELDS)
IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VERSION = PROTOCOL_VERSION

def rows(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]

def load_env(path: str) -> None:
    if not path or not Path(path).is_file(): return
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k,v=line.split("=",1); os.environ.setdefault(k.removeprefix("export ").strip(),v.strip().strip("'\""))

def transcript(segments: list[dict] | list[str] | str) -> str:
    if isinstance(segments,str): return segments
    lines=[]
    for segment in segments or []:
        if isinstance(segment,dict):
            lines.append(f"[{float(segment.get('start',0)):.1f}-{float(segment.get('end',0)):.1f}s] {segment.get('text','')}")
        else:
            lines.append(str(segment))
    return "\n".join(lines)

def cached_frames(media: Path, cache_root: Path, count: int) -> list[Path]:
    target=cache_root/media.stem; expected=[target/f"{i:03d}.jpg" for i in range(count)]
    if all(x.is_file() and x.stat().st_size for x in expected): return expected
    target.mkdir(parents=True,exist_ok=True)
    dur=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(media)],text=True).strip())
    for i,p in enumerate(expected):
        if p.is_file() and p.stat().st_size: continue
        ts=dur*(i+0.5)/count
        tmp=p.with_name(f".{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
        try:
            subprocess.run(["ffmpeg","-loglevel","error","-ss",f"{ts:.6f}","-i",str(media),"-frames:v","1","-q:v","3","-y",str(tmp)],check=True)
            if not tmp.is_file() or not tmp.stat().st_size:
                subprocess.run(["ffmpeg","-loglevel","error","-sseof","-0.05","-i",str(media),"-frames:v","1","-q:v","3","-y",str(tmp)],check=True)
            if (not tmp.is_file() or not tmp.stat().st_size) and i > 0 and expected[i-1].is_file():
                shutil.copyfile(expected[i-1],tmp)
            if not tmp.is_file() or not tmp.stat().st_size:
                raise RuntimeError(f"frame_extract_no_output:{media}:frame={i}")
            os.replace(tmp,p)
        finally:
            tmp.unlink(missing_ok=True)
    return expected

def normalized_square_frames(frames: list[Path], size: int) -> list[Path]:
    """Pad frames to a fixed square for processors that require stable image tokens."""
    if not size: return frames
    normalized=[]
    for source in frames:
        target=source.parent/f"square_{size}"/source.name
        if not target.is_file() or not target.stat().st_size:
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
            try:
                with Image.open(source) as image:
                    image = ImageOps.contain(image.convert("RGB"),(size,size),method=Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB",(size,size),(0,0,0))
                    canvas.paste(image,((size-image.width)//2,(size-image.height)//2))
                    canvas.save(tmp,format="JPEG",quality=90,optimize=True)
                if not tmp.is_file() or not tmp.stat().st_size: raise RuntimeError(f"square_frame_no_output:{source}")
                os.replace(tmp,target)
            finally: tmp.unlink(missing_ok=True)
        normalized.append(target)
    return normalized

def _original_prompt(item: dict, shown: list[str], asr: str, include_transcript: bool = True) -> str:
    blocks=["Answer the benchmark question using the supplied video frames and timestamped transcript.",
            "Return exactly one JSON object with exactly these nine keys: answer, G, W, N, T, S, A, R, M.",
            "answer is the final answer: option ID(s) for closed questions, or concise natural language for Open.",
            "Each reasoning slot must be a very short key judgment (normally <= 20 words) or null when unused. No slot is mandatory, including the slot matching the question category.",
            "At least one reasoning field must be non-null and informative.",
            "Only fill fields that materially contribute to your answer; do not fill irrelevant fields.",
            "Do not output chain-of-thought, hidden reasoning, or any keys besides the nine listed keys."]
    if item.get("memory_context"): blocks.append("Public memory context:\n"+str(item["memory_context"]))
    if include_transcript:
        blocks.append("Timestamped transcript (may contain ASR errors):\n"+(asr or "[No speech was transcribed]"))
    blocks.append("Question:\n"+item["question"])
    if shown: blocks.append("Options:\n"+"\n".join(f"{IDS[i]}. {v}" for i,v in enumerate(shown)))
    return "\n\n".join(blocks)

def _multiple_prompt(item: dict, shown: list[str], asr: str, include_transcript: bool = True) -> str:
    task_type = item.get("task_type")
    task_instructions = []
    if task_type == "multiple_select":
        task_instructions = [
            "QUESTION TYPE: MULTIPLE-SELECT (select all that apply).",
            "More than one option may be correct. Evaluate every listed option and select ALL options supported by the supplied input; do not choose only one best option, even if the question uses singular wording.",
            'Set "answer" to a JSON array of option letters, for example "answer": ["A", "C"]. Use an array even if you select only one option.',
            "The example letters illustrate the format only; choose the actual letters from this question's options. Include each selected letter once and no option text in answer.",
        ]
    elif task_type == "single_choice":
        task_instructions = [
            "QUESTION TYPE: SINGLE-CHOICE (choose exactly one option).",
            'Set "answer" to exactly one option letter as a JSON string, for example "answer": "A". Do not return an array or multiple letters.',
            "The example letter illustrates the format only; choose the actual letter from this question's options.",
        ]
    blocks=["Answer the benchmark question using the supplied video frames and timestamped transcript.",
            "Return exactly one JSON object with exactly these nine keys: answer, G, W, N, T, S, A, R, M.",
            "answer is the final answer: option ID(s) for closed questions, or concise natural language for Open.",
            "Each reasoning slot must be a very short key judgment (normally <= 20 words) or null when unused. No slot is mandatory, including the slot matching the question category.",
            "At least one reasoning field must be non-null and informative.",
            "Only fill fields that materially contribute to your answer; do not fill irrelevant fields.",
            "Do not output chain-of-thought, hidden reasoning, or any keys besides the nine listed keys."]
    if item.get("memory_context"): blocks.append("Public memory context:\n"+str(item["memory_context"]))
    if include_transcript:
        blocks.append("Timestamped transcript (may contain ASR errors):\n"+(asr or "[No speech was transcribed]"))
    blocks.append("Question:\n"+item["question"])
    if shown: blocks.append("Options:\n"+"\n".join(f"{IDS[i]}. {v}" for i,v in enumerate(shown)))
    return "\n\n".join(task_instructions + blocks)

def prompt(*args, **kwargs):
    item = args[0] if args else kwargs["item"]
    fn = _multiple_prompt if item.get("task_type") == "multiple_select" else _original_prompt
    return fn(*args, **kwargs)


def parse(text: str, item: dict, valid: set[str]) -> tuple[dict, bool, str | None]:
    clean=re.sub(r"^```(?:json)?\s*|\s*```$","",text.strip(),flags=re.I)
    try: value=json.loads(clean)
    except json.JSONDecodeError:
        m=re.search(r"\{.*\}",clean,re.S)
        try: value=json.loads(m.group()) if m else {}
        except json.JSONDecodeError: value={}
    if not isinstance(value,dict) or set(value) != set(FIELDS): return {},False,"schema_keys"
    value={key:value.get(key) for key in FIELDS}
    # Small local VLMs often serialize JSON null as the literal string "null".
    # Normalize that harmless representation, but never synthesize a missing answer.
    for key in FIELDS[1:]:
        if isinstance(value.get(key), str) and value[key].strip().lower().rstrip(".") in {"null", "none", "n/a", "not applicable"}:
            value[key] = None
    for key in FIELDS[1:]:
        if value[key] is not None and (not isinstance(value[key],str) or not value[key].strip() or len(value[key].split())>40): return {},False,f"invalid_{key}"
    if not informative_reasoning_slots(value):
        return value,False,"missing_reasoning_coverage"
    answer=value["answer"]
    options=item.get("options") or []
    option_text_to_id={
        re.sub(r"\s+", " ", str(option).strip()).casefold(): IDS[index]
        for index, option in enumerate(options)
    }
    def option_id(part: object) -> object:
        if not isinstance(part, str):
            return part
        stripped=part.strip()
        upper=stripped.upper()
        if upper in valid:
            return upper
        return option_text_to_id.get(re.sub(r"\s+", " ", stripped).casefold(), part)
    if item.get("task_type")=="open":
        okay=isinstance(answer,str) and bool(answer.strip())
    elif item.get("task_type")=="multiple_select":
        parts=answer if isinstance(answer,list) else [answer]
        vals=[]
        for part in parts:
            part=option_id(part)
            if not isinstance(part,str):
                vals.append(part); continue
            text=part.strip().upper()
            if text in valid:
                vals.append(text); continue
            vals.extend(m.group(1) for m in re.finditer(r"(?:^|[,;]\s*)([A-Z])(?:\s*[.)\-]|\s*(?=,|;|$))",text) if m.group(1) in valid)
        okay=bool(vals) and all(isinstance(x,str) and x.upper() in valid for x in vals)
        value["answer"]=[str(x).upper() for x in vals]
    else:
        answer=option_id(answer)
        if isinstance(answer,str):
            match=re.match(r"^\s*([A-Z])(?:\s*[.):\-]|\s+)",answer.upper())
            if match and match.group(1) in valid:
                answer=match.group(1)
        okay=isinstance(answer,str) and answer.upper() in valid
        if okay: value["answer"]=answer.upper()
    return value,okay,None if okay else "invalid_answer"

def call(base_url: str, model: str, content: list[dict], timeout: int, required_reasoning_slot: str | None = None) -> tuple[str,float]:
    body={"model":model,"messages":[{"role":"user","content":content}],"temperature":0.0,"max_tokens":768,"chat_template_kwargs":{"enable_thinking":False}}
    is_local="127.0.0.1" in base_url or "localhost" in base_url
    if is_local:
        informative={"type":"string","minLength":1,"maxLength":120}
        slot={"anyOf":[informative,{"type":"null"}]}
        reasoning_properties={key:slot for key in FIELDS[1:]}
        if required_reasoning_slot in REASONING_FIELDS:
            reasoning_properties[required_reasoning_slot]=informative
        schema={"type":"object","properties":{"answer":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]},**reasoning_properties},"required":FIELDS,"additionalProperties":False}
        body["response_format"]={"type":"json_schema","json_schema":{"name":"psychebench_reasoning","strict":True,"schema":schema}}
    headers={"Content-Type":"application/json"}
    token=os.environ.get("QWEN_CLOUD_API_KEY")
    if token and not is_local:
        headers["Authorization"]=f"Bearer {token}"
    req=urllib.request.Request(base_url.rstrip("/")+"/chat/completions",json.dumps(body).encode(),headers,method="POST")
    start=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as resp: data=json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8","replace")[:600]
        raise RuntimeError(f"HTTP_{exc.code}:{detail}") from exc
    if not isinstance(data, dict) or not data.get("choices"):
        detail=json.dumps(data,ensure_ascii=False)[:600]
        raise RuntimeError(f"invalid_completion_response:{detail}")
    return str(data["choices"][0]["message"]["content"]),time.monotonic()-start

def main() -> None:
    p=argparse.ArgumentParser()
    for n in ("public-test","ground-truth","media-root","asr-jsonl","output","model-id","base-url"): p.add_argument("--"+n,required=True)
    p.add_argument("--frame-count",type=int,default=32); p.add_argument("--frame-cache",default="./cache/frames32")
    p.add_argument("--square-frame-size",type=int,default=0)
    p.add_argument("--concurrency",type=int,default=2); p.add_argument("--timeout",type=int,default=600); p.add_argument("--retries",type=int,default=4); p.add_argument("--limit",type=int,default=0); p.add_argument("--secrets-env",default="")
    p.add_argument("--input-condition",choices=("full","visual","text"),default="full")
    p.add_argument("--drop-memory-context",action="store_true")
    p.add_argument("--protocol-version",default="psychebench-final-1000-nine-field-inference-v1")
    args=p.parse_args(); load_env(args.secrets_env); version=args.protocol_version
    public=rows(args.public_test)[:args.limit or None]; truth={x["question_id"]:x for x in rows(args.ground_truth)}; asr={x["episode_id"]:transcript(x.get("segments",[])) for x in rows(args.asr_jsonl) if x.get("status")=="OK"}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); done=set()
    if out.is_file():
        for existing in rows(out):
            qid=existing.get("question_id")
            if existing.get("status") != "OK" or not qid:
                continue
            # Historical memory context is part of the public model input. An
            # older row produced before that context was recovered is not done.
            if truth.get(qid, {}).get("memory_context") and not existing.get("memory_context_supplied"):
                continue
            done.add((qid, existing.get("model_id"), existing.get("protocol_version"), existing.get("condition", "full")))
    jobs=[]
    for item in public:
        key=(item["question_id"],args.model_id,version,args.input_condition)
        if key not in done: jobs.append(item)
    lock=threading.Lock()
    def run(item:dict)->dict:
        qid=item["question_id"]; gt=truth[qid]; options=item.get("options") or []; shown=options; valid=set(IDS[:len(shown)])
        media=Path(args.media_root)/Path(item["media_id"]).name
        evidence=gt.get("evidence") or []
        if isinstance(evidence,str): evidence=[evidence]
        memory_context=None if args.drop_memory_context else gt.get("memory_context")
        common={"question_id":qid,"episode_id":item["episode_id"],"model_id":args.model_id,"condition":args.input_condition,"category":item["category"],"source_dataset":item["source_dataset"],"task_type":item["task_type"],"protocol_version":version,"prompt_version":version,"memory_context_supplied":bool(memory_context),"frame_normalization":f"square_pad_{args.square_frame_size}" if args.square_frame_size else "none","evidence":evidence,"evidence_ids":[str(e.get("evidence_id")) for e in evidence if isinstance(e,dict) and e.get("evidence_id")],"audio_context":{"asr_present":item["episode_id"] in asr,"asr_source":args.asr_jsonl if item["episode_id"] in asr else None,"audio_file":"unknown","iga_key":"unknown"}}
        if not media.is_file(): return {**common,"status":"ERROR","error":"missing_media"}
        frame_paths = cached_frames(media,Path(args.frame_cache),args.frame_count) if args.input_condition != "text" else []
        frame_paths = normalized_square_frames(frame_paths,args.square_frame_size)
        image_content=[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(x.read_bytes()).decode()}} for x in frame_paths]
        prompt_item={**item,"memory_context":memory_context}
        include_transcript = args.input_condition != "visual"
        text_content={"type":"text","text":prompt(prompt_item,shown,asr.get(item["episode_id"],""),include_transcript=include_transcript)}
        request_images=image_content
        error=None; raw=""; value={}; why=None; lat=0.0
        for attempt in range(1,args.retries+1):
            try:
                request_content=request_images+[text_content]
                if attempt > 1 and why:
                    if why == "missing_reasoning_coverage":
                        retry_instruction = (
                            "Previous output was answer-only because every reasoning field was null. "
                            "Return exactly the required nine-key JSON object and fill at least one active "
                            "reasoning field with a concise, informative judgment grounded in the supplied "
                            "frames or transcript. An answer with all G/W/N/T/S/A/R/M fields null is invalid. "
                            "Every key must be present exactly once: answer, G, W, N, T, S, A, R, M. "
                            "Use JSON null only for genuinely unused fields; never omit a slot. For example: "
                            "{\"answer\":\"A\",\"G\":\"brief grounded fact\",\"W\":null,\"N\":null,"
                            "\"T\":null,\"S\":null,\"A\":null,\"R\":null,\"M\":null}."
                        )
                    else:
                        retry_instruction = (
                            f"Previous output failed validation ({why}). Return exactly the required "
                            "nine-key JSON object. Every key must be present exactly once: "
                            "answer, G, W, N, T, S, A, R, M. Use JSON null for every unused "
                            "slot; never omit a slot. Keep each non-null reasoning field under 12 words. "
                            "Do not enumerate frame numbers or restate the transcript. For example: "
                            "{\"answer\":\"A\",\"G\":\"The person visibly steps away.\","
                            "\"W\":null,\"N\":null,\"T\":null,\"S\":null,\"A\":null,"
                            "\"R\":null,\"M\":null}."
                        )
                    previous = raw[:1200] if raw else "[no parseable output]"
                    request_content=request_content + [{
                        "type":"text",
                        "text":f"Previous invalid output:\n{previous}\n\n{retry_instruction}",
                    }]
                required_slot=item.get("category") if attempt > 1 and why == "missing_reasoning_coverage" else None
                raw,lat=call(args.base_url,args.model_id,request_content,args.timeout,required_slot); value,okay,why=parse(raw,item,valid)
                if okay:
                    return {**common,"condition":args.input_condition,"prediction":value,"prediction_raw":raw,"format_valid":True,"reasoning_coverage":"present","informative_reasoning_slots":list(informative_reasoning_slots(value)),"status":"OK","error":None,"attempt":attempt,"latency_seconds":round(lat,3),"frame_count_used":len(request_images),"transcript_supplied":include_transcript}
                error=why
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                context_overflow = (
                    "exceeds model's maximum context length" in error
                    or "max_tokens must be at least 1" in error
                    or "maximum context length" in error
                )
                if context_overflow and len(request_images)>1:
                    request_images=request_images[::4] if len(request_images)>=32 else request_images[::2]
                    continue
            time.sleep(min(20,2**(attempt-1)+random.random()))
        return {**common,"condition":args.input_condition,"prediction":value,"prediction_raw":raw,"format_valid":False,"reasoning_coverage":"present" if informative_reasoning_slots(value) else "missing","informative_reasoning_slots":list(informative_reasoning_slots(value)),"status":"UNPARSEABLE" if raw else "ERROR","error":error,"attempt":args.retries,"latency_seconds":round(lat,3),"frame_count_used":len(request_images),"transcript_supplied":include_transcript}
    with out.open("a",encoding="utf-8") as h, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        fs={pool.submit(run,x):x for x in jobs}
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            item=fs[f]
            try:
                r=f.result()
            except Exception as exc:
                r={"question_id":item["question_id"],"episode_id":item["episode_id"],"model_id":args.model_id,"condition":"full","category":item["category"],"source_dataset":item["source_dataset"],"task_type":item["task_type"],"protocol_version":version,"prompt_version":version,"status":"ERROR","format_valid":False,"error":f"{type(exc).__name__}: {exc}"}
            with lock: h.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"); h.flush(); os.fsync(h.fileno())
            print(json.dumps({"done":i,"total":len(jobs),"qid":r["question_id"],"status":r["status"]}),flush=True)

if __name__=="__main__": main()
