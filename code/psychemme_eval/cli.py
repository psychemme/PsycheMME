from __future__ import annotations
import argparse
import json
import math
import subprocess
from pathlib import Path
from .data import read_jsonl, validate_pair, validate_public, unique_index, dump_json, dump_jsonl
from .scoring import replay_factors, score_predictions


def main():
    parser = argparse.ArgumentParser(prog='psychemme')
    parser.add_argument('--debug', action='store_true', help='Show the Python traceback for input or environment errors')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Run a synthetic scoring example offline; no media, GPU or API key')
    demo.add_argument('--output', default='runs/demo')
    download = commands.add_parser('download', help='Download the benchmark or paper scoring records')
    download.add_argument('--output', help='Destination directory')
    download.add_argument('--bundle', choices=['benchmark', 'replay'], default='benchmark')
    download.add_argument('--metadata-only', action='store_true', help='Skip benchmark videos')
    doctor = commands.add_parser('doctor', help='Check dependencies and optionally pinned replay data')
    doctor.add_argument('--mode', choices=['core', 'local'], default='core'); doctor.add_argument('--data-root')
    convert = commands.add_parser('import-predictions', help='Convert your question_id/response JSONL to the evaluation format')
    for name in ('inputs', 'responses', 'model', 'output'): convert.add_argument('--'+name, required=True)
    check = commands.add_parser('validate', help='Check public/evaluator boundary, IDs and reference shape')
    check.add_argument('--inputs', required=True); check.add_argument('--references'); check.add_argument('--asr')
    replay = commands.add_parser('replay', help='Rebuild scores from archived basic credit and categorical judge factors')
    replay.add_argument('--root', type=Path, default=Path('.')); replay.add_argument('--output', required=True)
    score = commands.add_parser('score', help='Score new predictions with persisted judgments and fixed reference denominator')
    for name in ('inputs','references','predictions','judgments','output'): score.add_argument('--'+name, required=True)
    judge = commands.add_parser('judge', help='Call a user-specified text Judge with evaluator-only references')
    for name in ('inputs','references','predictions','base-url','judge-model','output'): judge.add_argument('--'+name, required=True)
    judge.add_argument('--api-key-env', default=''); judge.add_argument('--timeout', type=float, default=300)
    judge.add_argument('--retries', type=int, default=2); judge.add_argument('--concurrency', type=int, default=4)
    infer = commands.add_parser('infer', help='Run public inputs using local Qwen or an explicitly configured endpoint')
    for name in ('inputs','asr','sources','media-root','frame-cache','model','output'): infer.add_argument('--'+name, required=True)
    infer.add_argument('--backend', choices=['local-qwen','api'], default='local-qwen')
    infer.add_argument('--base-url'); infer.add_argument('--api-key-env', default=''); infer.add_argument('--timeout', type=float, default=900)
    infer.add_argument('--revision'); infer.add_argument('--adapter'); infer.add_argument('--adapter-revision')
    infer.add_argument('--limit', type=int, default=0)
    infer.add_argument('--resume', action='store_true', help='Continue a matching run')
    infer.add_argument('--retry-errors', action='store_true', help='Retry saved execution errors with --resume')
    infer.add_argument('--retries', type=int, default=2, help='Retries per transient request failure')
    judge.add_argument('--retry-errors', action='store_true', help='Retry saved Judge errors with --resume')
    judge.add_argument('--resume', action='store_true', help='Continue a matching Judge run')
    prepare = commands.add_parser('prepare', help='Download videos and prepare matching sample inputs')
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--limit', required=True, type=int)
    from .video import add_arguments
    video = commands.add_parser('video', help='Ask a question about your own local video using Qwen')
    add_arguments(video)
    args = parser.parse_args()
    try:
        return execute(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if args.debug:
            raise
        # Network failures can contain credential-bearing URLs; do not print them.
        import urllib.error
        message = 'Download failed. Check the connection and rerun; verified files are reused.' if isinstance(exc, urllib.error.URLError) else str(exc)
        parser.exit(2, f'psychemme: {message}\nSee docs/TROUBLESHOOTING.md or use --debug before the subcommand.\n')


def execute(args):
    if args.command in {'demo', 'download', 'doctor', 'import-predictions'}:
        from . import onboarding
        if args.command == 'demo': result = onboarding.demo(args.output)
        elif args.command == 'download':
            output = args.output or ('data/PsycheMME-logs' if args.bundle == 'replay' else 'data/PsycheMME')
            result = onboarding.download(output, args.bundle, args.metadata_only)
        elif args.command == 'doctor': result = onboarding.doctor(args.data_root, args.mode)
        else: result = onboarding.import_predictions(args.inputs, args.responses, args.model, args.output)
    elif args.command == 'prepare':
        from .subset import prepare
        result=prepare(args.root, args.output, args.limit)
    elif args.command == 'validate':
        public = read_jsonl(args.inputs)
        validate_pair(public,read_jsonl(args.references)) if args.references else validate_public(public)
        if args.asr:
            asr=unique_index(read_jsonl(args.asr),'episode_id')
            if {q['episode_id'] for q in public} - asr.keys():
                raise ValueError('Missing ASR records')
        result={'status':'PASS','questions':len(public),'episodes':len({r['episode_id'] for r in public}),'scope':'Validated question IDs, reference fields and ASR mappings'}
    elif args.command == 'replay':
        root=args.root
        factors=read_jsonl(root/'results/scoring_inputs.jsonl')
        population=read_jsonl(root/'dataset/metadata_only.jsonl')
        records,summaries=replay_factors(factors,population)
        saved={(r['model'],r['question_id']):r for r in read_jsonl(root/'results/per_question_scores.jsonl')}
        if set(saved)!={(r['model'],r['question_id']) for r in records}:
            raise ValueError('Archived row closure mismatch')
        for row in records:
            expected=saved[row['model'],row['question_id']]
            if any(not math.isclose(row[k],expected[k],rel_tol=0,abs_tol=1e-12) for k in ('raw','verified')):
                raise ValueError('Factor replay differs from archived score')
        paper=json.loads((root/'results/paper_table_2.json').read_text())
        table={r['model']:r['scores'] for r in paper['models']}
        cells=0
        for model in summaries:
            values=[model['raw_macro'],model['verified_macro'],*[model['capabilities'][c]['verified'] for c in 'GWNTSAMR']]
            if [f'{v:.2f}' for v in values]!=table[model['model']]:
                raise ValueError('Paper cell mismatch')
            cells+=len(values)
        result={'status':'PASS','models':len(summaries),'factor_rows':len(records),'paper_cells_matched':cells,'summaries':summaries,
                'scope':'Recomputed paper scores from released answer credit and rationale factors'}
        dump_json(args.output,result)
    elif args.command == 'video':
        from .video import run
        result = run(args)
    elif args.command == 'score':
        records,summary=score_predictions(read_jsonl(args.inputs),read_jsonl(args.references),read_jsonl(args.predictions),read_jsonl(args.judgments))
        dump_jsonl(args.output,records);dump_json(Path(args.output).with_suffix('.summary.json'),summary)
        result=summary
    elif args.command == 'judge':
        if args.retries < 1 or args.concurrency < 1 or args.timeout < 1:
            raise ValueError('Judge retries, concurrency and timeout must be positive.')
        import sys
        from . import judge as judge_module
        public=read_jsonl(args.inputs);validate_pair(public,read_jsonl(args.references))
        predictions=unique_index(read_jsonl(args.predictions))
        if predictions.keys()!={r['question_id'] for r in public}:
            raise ValueError('Judge requires exact prediction/public ID closure')
        argv=['psychemme-judge','--ground-truth',args.references,'--predictions',args.predictions,
              '--public-inputs',args.inputs,'--output',args.output,'--base-url',args.base_url,
              '--judge-model',args.judge_model,'--config',str(Path(__file__).with_name('scoring_config.json')),
              '--api-key-env',args.api_key_env,'--timeout',str(args.timeout),'--retries',str(args.retries),'--concurrency',str(args.concurrency)]
        # Original CLI accepts integral seconds.
        argv[argv.index('--timeout')+1]=str(int(args.timeout))
        from .checkpoint import run_lock, bind_output
        from .data import sha256
        settings = {name: sha256(getattr(args, name)) for name in ('inputs', 'references', 'predictions')}
        settings.update(endpoint=args.base_url, model=args.judge_model,
                        implementation=sha256(Path(judge_module.__file__)),
                        memory=sha256(Path(__file__).with_name('memory_verification_context.py')),
                        scoring=sha256(Path(__file__).with_name('scoring_config.json')))
        if args.retry_errors:
            if not args.resume: raise ValueError('--retry-errors requires --resume')
            argv.append('--retry-errors')
        with run_lock(args.output):
            bind_output(args.output, settings, args.resume)
            previous=sys.argv
            try:
                sys.argv=argv; judge_module.main()
            finally:
                sys.argv=previous
            rows=read_jsonl(args.output)
        errors=sum(r.get('status')!='OK' for r in rows)
        result={'judgments':len(rows), 'expected':len(public), 'errors':errors,
                'status':'PASS' if len(rows)==len(public) and not errors else 'INCOMPLETE'}
    else:
        if args.limit<0 or args.retries<0:raise ValueError('--limit and --retries must be nonnegative')
        if args.retry_errors and not args.resume:raise ValueError('--retry-errors requires --resume')
        from .inference import run
        result=run(read_jsonl(args.inputs),read_jsonl(args.asr),read_jsonl(args.sources),args)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'},ensure_ascii=False,allow_nan=False,indent=2))
    if args.command == 'doctor' and result['status'] != 'PASS':
        raise SystemExit(1)
    if args.command in {'infer', 'judge'} and result['status'] != 'PASS':
        raise SystemExit(1)
    if args.command == 'video' and result['status'] != 'OK':
        raise SystemExit(1)
