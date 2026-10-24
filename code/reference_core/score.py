#!/usr/bin/env python3
"""Combine frozen basic Answer Score with configurable Support/Logic multipliers."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

try:
    from .reasoning_verified_contract import informative_reasoning_slots, reasoning_coverage
except ImportError:
    from reasoning_verified_contract import informative_reasoning_slots, reasoning_coverage

def read(path): return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--ground-truth",required=True); p.add_argument("--predictions",required=True); p.add_argument("--judgments",required=True); p.add_argument("--config",required=True); p.add_argument("--output",required=True); a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text()); gt={x["question_id"]:x for x in read(a.ground_truth)}; pred={x["question_id"]:x for x in read(a.predictions) if x.get("question_id")}; judges={x["question_id"]:x for x in read(a.judgments) if x.get("status")=="OK"}
 sm=cfg["support_multiplier"]; lm=cfg["logic_multiplier"]; cm=cfg["coverage_multiplier"]; maxm=float(cfg["normalization"]["max_multiplier"]); out=[]
 for qid,p in pred.items():
  t=gt[qid]; j=judges.get(qid,{}); basic=p.get("basic_answer_score")
  inference_valid=p.get("status")=="OK" and p.get("format_valid") is True
  if not inference_valid: basic=0.0
  if basic is None:
   # Open answers retain the existing relaxed Judge score when available.
   basic=p.get("open_score", j.get("basic_answer_score"))
  if basic is None: basic=0.0
  coverage=reasoning_coverage(p.get("prediction")) if inference_valid else "missing"; slots=informative_reasoning_slots(p.get("prediction")) if inference_valid else ()
  support=j.get("support"); logic=j.get("logic")
  if coverage == "missing": support="insufficient"; logic="unevaluable"
  raw=float(basic)*float(cm[coverage])*float(sm[support])*float(lm[logic])
  normalized=raw/maxm if cfg["normalization"].get("enabled") else raw
  out.append({"question_id":qid,"model_id":p.get("model_id"),"category":t.get("category"),"basic_answer_score":basic,"reasoning_coverage":coverage,"informative_reasoning_slots":list(slots),"support":support,"logic":logic,"final_score_raw":raw,"final_score_normalized":normalized,"judge_status":j.get("status","MISSING"),"inference_status":p.get("status"),"inference_valid":inference_valid})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out))
 by=collections.defaultdict(list)
 for x in out: by[(x["model_id"],x["category"])].append(x)
 summary={"items":len(out),"models":{},"reasoning_coverage_distribution":dict(collections.Counter(x.get("reasoning_coverage") for x in out)),"support_distribution":dict(collections.Counter(x.get("support") for x in out)),"logic_distribution":dict(collections.Counter(x.get("logic") for x in out))}
 for (m,c),vals in by.items(): summary["models"].setdefault(m,{})[c]={"n":len(vals),"raw_answer_score":sum(x["basic_answer_score"] for x in vals)/len(vals),"final_raw":sum(x["final_score_raw"] for x in vals)/len(vals),"final_normalized":sum(x["final_score_normalized"] for x in vals)/len(vals)}
 for m,cats in summary["models"].items():
  summary["models"][m]["macro"]={k:sum(v[k] for v in cats.values())/len(cats) for k in ("raw_answer_score","final_raw","final_normalized")}
 Path(a.output).with_suffix(".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
