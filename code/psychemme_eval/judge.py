#!/usr/bin/env python3
"""Text-only Evidence Support and Logical Integrity judge."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, random, re, threading, time, urllib.error, urllib.request
from pathlib import Path

try:
    from .reasoning_verified_contract import REASONING_FIELDS, informative_reasoning_slots, reasoning_coverage
except ImportError:
    from reasoning_verified_contract import REASONING_FIELDS, informative_reasoning_slots, reasoning_coverage

try:
    from .memory_verification_context import build_memory_context, context_digest, SYSTEM_ADDITION, VERSION
except ImportError:
    from memory_verification_context import build_memory_context, context_digest, SYSTEM_ADDITION, VERSION

REASONING_SYSTEM = """You are a strict but fair text-only judge for Reasoning-Verified Evaluation.
You receive a question, options, gold/reference answer, frozen evidence, a model answer,
and nine-field concise reasoning. Do not compare against a Gold chain-of-thought: none exists.
Check (1) whether the reasoning is compatible with the evidence, allowing reasonable non-unique
psychological interpretations, and (2) whether the active slots support one another and the final
answer. Return exactly JSON with keys: support (strong|normal|weak|contradicted),
logic (fully_coherent|acceptable|weak|contradictory), support_reason, logic_reason. Keep each reason under 60 words.
Judge only the reasoning that the candidate explicitly wrote.
Do not infer, reconstruct, or supply reasoning that the candidate did not explicitly provide.
A correct answer does not imply that its rationale is supported or logically coherent.
Null fields contain no reasoning evidence and must never be treated as implicit justification.
Do not recount options, deliberate in the response, or add keys."""

OPEN_ANSWER_SYSTEM = """You are the relaxed semantic answer grader for an Open PsycheBench item.
Judge only the candidate final answer against the question, frozen evidence, and reference answer.
Do not receive or infer any candidate rationale. Return exactly JSON with keys answer_score and answer_reason.
answer_score must be 1 for a correct semantically equivalent answer, 0.5 for a materially correct but incomplete
answer, or 0 for a wrong, contradictory, unsupported, or non-answer response. Keep answer_reason under 60 words."""

def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(x) for x in stream if x.strip()]
def load_env(path):
    if not path or not Path(path).is_file(): return
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k,v=line.split("=",1); os.environ.setdefault(k.removeprefix("export ").strip(),v.strip().strip("'\""))
def extract(text):
    clean=re.sub(r"^```(?:json)?\s*|\s*```$","",text.strip(),flags=re.I)
    try:return json.loads(clean)
    except json.JSONDecodeError:
        m=re.search(r"\{.*\}",clean,re.S); return json.loads(m.group()) if m else {}
def call(args,payload,system=REASONING_SYSTEM):
    body={"model":args.judge_model,"messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}],"temperature":0.0,"max_tokens":768,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":False}}
    headers={"Content-Type":"application/json"}; key=os.environ.get(args.api_key_env,"")
    if key: headers["Authorization"]="Bearer "+key
    url=args.base_url.rstrip("/")+"/chat/completions"
    def request(request_body):
        req=urllib.request.Request(url,json.dumps(request_body).encode(),headers,method="POST")
        with urllib.request.urlopen(req,timeout=args.timeout) as resp:
            return str(json.loads(resp.read())["choices"][0]["message"]["content"])
    try:
        raw=request(body)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400,422,500}:
            raise
        body.pop("response_format",None)
        raw=request(body)
    return extract(raw),raw
def exact_basic(gt,pred):
    task=gt.get("task_type"); ans=(pred.get("prediction") or {}).get("answer")
    if task=="open": return None
    gold=[chr(65+i) for i in (gt.get("answer_indices") or [])]
    if task=="multiple_select": return 1.0 if sorted(set(ans or []))==sorted(gold) else 0.0
    return 1.0 if isinstance(ans,str) and ans.upper() in gold else 0.0
def main():
    p=argparse.ArgumentParser()
    for n in ("ground-truth","predictions","output","base-url","judge-model"):p.add_argument("--"+n,required=True)
    p.add_argument("--config",required=True);p.add_argument("--api-key-env",default="");p.add_argument("--secrets-env",default="");p.add_argument("--concurrency",type=int,default=4);p.add_argument("--timeout",type=int,default=300);p.add_argument("--retries",type=int,default=4)
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--public-inputs', default=''); p.add_argument('--memory-map', default='')
    a=p.parse_args();load_env(a.secrets_env); gt={x["question_id"]:x for x in read(a.ground_truth)}
    public={x['question_id']:x for x in read(a.public_inputs)} if a.public_inputs else gt
    memory_map={x['question_id']:x for x in read(a.memory_map)} if a.memory_map else None
    latest_predictions={}
    for row in read(a.predictions):
        if row.get("question_id"):
            latest_predictions[row["question_id"]]=row
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    previous=read(out) if out.is_file() else []
    done=set()
    assert len({row['question_id'] for row in previous}) == len(previous), 'duplicate judgment IDs'
    for old in previous:
        qid=old['question_id']
        assert qid in latest_predictions, 'foreign judgment in output'
        context=build_memory_context(gt[qid],public[qid],memory_map[qid] if memory_map is not None else None)
        if context:
            assert qid not in done, 'duplicate terminal memory judgment'
            assert old.get('judge_version')==VERSION and old.get('verification_memory_context_sha256')==context_digest(context), 'legacy or changed memory judgment: use a fresh output and preserve the old file'
            assert old.get('status') in {'OK','ERROR'}, 'nonterminal memory judgment'
            done.add(qid)
        elif old.get('status') in {'OK', 'ERROR'}:
            done.add(qid)
    if a.retry_errors:
        done -= {row['question_id'] for row in previous if row.get('status') == 'ERROR'}
    predictions=[x for qid,x in latest_predictions.items() if qid not in done];lock=threading.Lock()
    def run(pred):
        truth=gt[pred["question_id"]]; basic=exact_basic(truth,pred)
        intervention = memory_map[pred['question_id']] if memory_map is not None else None
        if pred.get('memory_intervention') in {'no_memory', 'wrong_memory'}:
            assert intervention is not None, 'intervention predictions require --memory-map'
            assert pred['memory_intervention'] == intervention['condition']
            assert pred.get('donor_question_id') == intervention.get('donor_question_id')
        memory = build_memory_context(truth, public[pred['question_id']], intervention)
        memory_meta = {"judge_version": VERSION, "verification_memory_context_sha256": context_digest(memory)} if memory else {}
        if pred.get("status") != "OK" or pred.get("format_valid") is not True:
            return {
                "question_id": pred["question_id"],
                "model_id": pred.get("model_id"),
                "status": "OK",
                "basic_answer_score": 0.0,
                "answer_reason": "Inference did not produce a valid benchmark response.",
                "answer_judge_raw": None,
                "answer_judge_attempt": 0,
                "reasoning_coverage": "missing",
                "informative_reasoning_slots": [],
                "support": "insufficient",
                "logic": "unevaluable",
                "support_reason": "No valid explicit rationale was produced.",
                "logic_reason": "Logical integrity cannot be evaluated for an invalid response.",
                "reasoning_judge_skipped": True,
                "reasoning_judge_raw": None,
                "judge_model": a.judge_model,
                "judge_version": "psychebench-reasoning-text-judge-v3-explicit-only",
                "attempt": 0,
                "inference_status": pred.get("status"),
                "inference_error": pred.get("error"),
                "error": None,
                **memory_meta,
            }
        prediction=pred.get("prediction") or {}; coverage=reasoning_coverage(prediction); slots=informative_reasoning_slots(prediction)
        answer_raw=None; answer_reason=None; error=None; answer_attempt=0
        if basic is None:
            answer_payload={"question":truth["question"],"frozen_evidence":truth.get("evidence") or [],"reference_answer":truth.get("reference_answer") or truth.get("answers"),"model_answer":prediction.get("answer"),"task_type":truth.get("task_type")}
            answer_payload.update(memory)
            for answer_attempt in range(1,a.retries+1):
                try:
                    answer_value,answer_raw=call(a,answer_payload,OPEN_ANSWER_SYSTEM + SYSTEM_ADDITION if memory else OPEN_ANSWER_SYSTEM)
                    if answer_value.get("answer_score") not in {0,0.5,1}: raise ValueError("invalid open answer judge schema")
                    basic=float(answer_value["answer_score"]); answer_reason=answer_value.get("answer_reason"); break
                except Exception as exc:
                    error=type(exc).__name__ + (f" (HTTP {exc.code})" if isinstance(exc, urllib.error.HTTPError) else ""); time.sleep(min(20,2**(answer_attempt-1)+random.random()))
            if basic is None:
                return {"question_id":pred["question_id"],"model_id":pred.get("model_id"),"status":"ERROR","stage":"open_answer_judge","error":error,**memory_meta}
        if coverage == "missing":
            return {"question_id":pred["question_id"],"model_id":pred.get("model_id"),"status":"OK","basic_answer_score":basic,"answer_reason":answer_reason,"answer_judge_raw":answer_raw,"answer_judge_attempt":answer_attempt,"reasoning_coverage":"missing","informative_reasoning_slots":[],"support":"insufficient","logic":"unevaluable","support_reason":"No explicit informative rationale was provided.","logic_reason":"Logical integrity cannot be evaluated without explicit rationale.","reasoning_judge_skipped":True,"reasoning_judge_raw":None,"judge_model":a.judge_model,"judge_version":"psychebench-reasoning-text-judge-v3-explicit-only","attempt":0,"error":None,**memory_meta}
        payload={"question":truth["question"],"options":truth.get("options") or [],"gold_reference_answer":truth.get("reference_answer") or truth.get("answers") or truth.get("answer_indices"),"frozen_evidence":truth.get("evidence") or [],"model_answer":prediction.get("answer"),"model_reasoning":{field:prediction.get(field) for field in REASONING_FIELDS},"informative_reasoning_slots":list(slots),"category":truth.get("category"),"task_type":truth.get("task_type")}
        payload.update(memory)
        error=None
        for attempt in range(1,a.retries+1):
            try:
                value,raw=call(a,payload,REASONING_SYSTEM + SYSTEM_ADDITION if memory else REASONING_SYSTEM); valid=value.get("support") in {"strong","normal","weak","contradicted"} and value.get("logic") in {"fully_coherent","acceptable","weak","contradictory"}
                if not valid: raise ValueError("invalid judge schema")
                return {"question_id":pred["question_id"],"model_id":pred.get("model_id"),"status":"OK","basic_answer_score":basic,"answer_reason":answer_reason,"answer_judge_raw":answer_raw,"answer_judge_attempt":answer_attempt,**value,"reasoning_coverage":"present","informative_reasoning_slots":list(slots),"reasoning_judge_skipped":False,"reasoning_judge_raw":raw,"judge_model":a.judge_model,"judge_version":"psychebench-reasoning-text-judge-v3-explicit-only","attempt":attempt,"error":None,**memory_meta}
            except Exception as exc: error=type(exc).__name__ + (f" (HTTP {exc.code})" if isinstance(exc, urllib.error.HTTPError) else ""); time.sleep(min(20,2**(attempt-1)+random.random()))
        return {"question_id":pred["question_id"],"model_id":pred.get("model_id"),"status":"ERROR","error":error,**memory_meta}
    try:
        from .data import dump_jsonl
    except ImportError:
        from data import dump_jsonl
    persisted = list(previous)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,x) for x in predictions]):
            r=f.result()
            with lock:
                old = next((row for row in persisted if row['question_id'] == r['question_id']), None)
                if old:
                    r['previous_attempts'] = [*old.get('previous_attempts', []),
                                              {k:v for k,v in old.items() if k != 'previous_attempts'}]
                    persisted = [row for row in persisted if row['question_id'] != r['question_id']]
                persisted.append(r)
                dump_jsonl(out, persisted)
            print(json.dumps({"qid":r["question_id"],"status":r["status"]}),flush=True)
if __name__=="__main__":main()
