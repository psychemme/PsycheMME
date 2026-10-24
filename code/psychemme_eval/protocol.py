"""Prompt/parser functions extracted unchanged from the final paper runtime."""

from __future__ import annotations

import json, re

from .reasoning_verified_contract import OUTPUT_FIELDS, informative_reasoning_slots, reasoning_coverage

FIELDS=list(OUTPUT_FIELDS)

IDS=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def transcript(segments: list[dict] | list[str] | str) -> str:
    if isinstance(segments,str): return segments
    lines=[]
    for segment in segments or []:
        if isinstance(segment,dict):
            lines.append(f"[{float(segment.get('start',0)):.1f}-{float(segment.get('end',0)):.1f}s] {segment.get('text','')}")
        else:
            lines.append(str(segment))
    return "\n".join(lines)

def prompt(item: dict, shown: list[str], asr: str, include_asr: bool = True) -> str:
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
    input_description = "the supplied video frames and timestamped transcript" if include_asr else "the supplied source video with its native audio"
    blocks=[f"Answer the benchmark question using {input_description}.",
            "Return exactly one JSON object with exactly these nine keys: answer, G, W, N, T, S, A, R, M.",
            "answer is the final answer: option ID(s) for closed questions, or concise natural language for Open.",
            "Each reasoning slot must be a very short key judgment (normally <= 20 words) or null when unused. No slot is mandatory, including the slot matching the question category.",
            "At least one reasoning field must be non-null and informative.",
            "Only fill fields that materially contribute to your answer; do not fill irrelevant fields.",
            "Do not output chain-of-thought, hidden reasoning, or any keys besides the nine listed keys."]
    if item.get("memory_context"): blocks.append("Public memory context:\n"+str(item["memory_context"]))
    if include_asr:
        blocks.append("Timestamped transcript (may contain ASR errors):\n"+(asr or "[No speech was transcribed]"))
    blocks.append("Question:\n"+item["question"])
    if shown: blocks.append("Options:\n"+"\n".join(f"{IDS[i]}. {v}" for i,v in enumerate(shown)))
    return "\n\n".join(task_instructions + blocks)

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

def enterprise_prompt(item: dict, options: list[str], asr_text: str, include_asr: bool) -> str:
    text = prompt(item, options, asr_text, include_asr=include_asr)
    text += (
        "\n\nSlot meanings: G=person/scene grounding; W=emotion/state and cause; "
        "N=need/request; T=response timing; S=spoken response strategy; "
        "A=physical/nonverbal action; R=risk/consent/boundary; "
        "M=relevant prior memory. Only fill a slot when that exact capability "
        "materially supports the answer; otherwise use null."
    )
    if item.get("task_type") == "open":
        text += "\nFor this Open question, answer must be a concise natural-language answer, never an option letter."
    return text

def enterprise_parse(raw: str, item: dict, valid: set[str]):
    value, okay, why = parse(raw, item, valid)
    if okay and item.get("task_type") == "open":
        answer = value.get("answer")
        if isinstance(answer, str) and re.fullmatch(r"[A-Z](?:[.)])?", answer.strip(), re.I):
            return value, False, "open_answer_is_option_label"
    return value, okay, why

from .reasoning_verified_contract import reasoning_coverage
