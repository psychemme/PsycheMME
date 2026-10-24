import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
from psychemme_eval import inference, cli
from psychemme_eval.data import read_jsonl, dump_jsonl
from psychemme_eval.onboarding import demo
from psychemme_eval.checkpoint import run_lock

class Recovery(unittest.TestCase):
    def setup_run(self, root):
        demo(root/'fixture')
        public=read_jsonl(root/'fixture/inputs.jsonl')[:2]
        args=argparse.Namespace(output=str(root/'predictions.jsonl'),  backend='api', model='test-model',
            base_url='http://local/v1', revision=None, adapter=None, adapter_revision=None,
            media_root=str(root),frame_cache=str(root/'frames'),api_key_env='',timeout=1,resume=True,retries=2,retry_errors=False)
        answer=json.dumps({'answer':'A','G':'A visible cue.', **{k:None for k in 'WNTSAMR'}})
        return public,args,answer

    def test_interruption_resumes_without_repeating_completed_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public,args,answer=self.setup_run(root)
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',side_effect=[(answer,'stop'),KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt): inference.run(public,[],[],args)
            self.assertEqual(len(read_jsonl(args.output)),1)
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',return_value=(answer,'stop')) as request:
                result=inference.run(public,[],[],args)
                self.assertEqual(request.call_count,1)
            self.assertEqual(result['successes'],2)
            args.model='changed'
            with self.assertRaisesRegex(ValueError,'settings or inputs changed'):inference.run(public,[],[],args)

    def test_finite_network_retry_and_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public,args,answer=self.setup_run(root);public=public[:1]
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',side_effect=TimeoutError()),patch.object(inference.time,'sleep'),contextlib.redirect_stderr(io.StringIO()):
                result=inference.run(public,[],[],args)
            self.assertEqual(result['status'],'INCOMPLETE')
            with patch.object(inference,'request_api') as request:
                inference.run(public,[],[],args);request.assert_not_called()
            args.retry_errors=True
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',return_value=(answer,'stop')):
                result=inference.run(public,[],[],args)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(len(read_jsonl(args.output)[0]['previous_attempts']),1)

    def test_invalid_answer_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public,args,_=self.setup_run(root);args.retry_errors=True
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',return_value=('bad-format','stop')) as request:
                inference.run(public,[],[],args);inference.run(public,[],[],args)
                self.assertEqual(request.call_count,2)
            self.assertEqual(len(read_jsonl(args.output)),2)

    def test_auth_failure_stops_before_remaining_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public,args,_=self.setup_run(root)
            error=urllib.error.HTTPError('http://unused',401,'Unauthorized',{},None)
            with patch.object(inference,'build_input',return_value=([], 'prompt')),patch.object(inference,'request_api',side_effect=error) as request:
                result=inference.run(public,[],[],args)
            self.assertEqual(request.call_count,1);self.assertEqual(result['pending'],1)
            self.assertEqual(read_jsonl(args.output)[0]['http_status'],401)

    def test_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with run_lock(Path(directory)/'output'):
                with self.assertRaisesRegex(ValueError,'another process'):
                    with run_lock(Path(directory)/'output'):pass

    def test_incomplete_infer_exits_nonzero(self):
        with patch.object(cli,'read_jsonl',return_value=[]),patch.object(inference,'run',return_value={'status':'INCOMPLETE'}),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.execute(argparse.Namespace(command='infer',retries=2,retry_errors=False,resume=True,inputs='x',asr='x',sources='x'))
            self.assertEqual(error.exception.code,1)

    def test_judge_interruption_resumes_persisted_rows(self):
        from psychemme_eval import judge
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); demo(root/'fixture')
            public=read_jsonl(root/'fixture/inputs.jsonl')[:2]
            truth=read_jsonl(root/'fixture/references_EVALUATOR_ONLY.jsonl')[:2]
            predictions=read_jsonl(root/'fixture/predictions.jsonl')[:2]
            for name,rows in [('inputs',public),('truth',truth),('predictions',predictions)]:
                dump_jsonl(root/(name+'.jsonl'),rows)
            argv=['psychemme','judge','--resume','--inputs',str(root/'inputs.jsonl'),
                  '--references',str(root/'truth.jsonl'),'--predictions',str(root/'predictions.jsonl'),
                  '--base-url','http://unused','--judge-model','fixture','--concurrency','1',
                  '--output',str(root/'judgments.jsonl')]
            answer=({'support':'strong','logic':'fully_coherent'},'{}')
            with patch.object(sys,'argv',argv),patch.object(judge,'call',side_effect=[answer,KeyboardInterrupt()]),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):cli.main()
            self.assertEqual(len(read_jsonl(root/'judgments.jsonl')),1)
            with patch.object(sys,'argv',argv),patch.object(judge,'call',return_value=answer) as call,contextlib.redirect_stdout(io.StringIO()):
                cli.main();self.assertEqual(call.call_count,1)
            self.assertEqual(len(read_jsonl(root/'judgments.jsonl')),2)
            with patch.object(sys,'argv',argv),patch.object(judge,'call') as call,contextlib.redirect_stdout(io.StringIO()):
                cli.main();call.assert_not_called()

    def test_judge_failed_request_can_be_recovered_without_repeating_successes(self):
        from psychemme_eval import judge
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);demo(root/'fixture')
            for filename,source in [('inputs','inputs'),('truth','references_EVALUATOR_ONLY'),('predictions','predictions')]:
                dump_jsonl(root/(filename+'.jsonl'),read_jsonl(root/'fixture'/(source+'.jsonl'))[:2])
            argv=['psychemme','judge','--resume','--retries','1','--concurrency','1',
                  '--inputs',str(root/'inputs.jsonl'),'--references',str(root/'truth.jsonl'),
                  '--predictions',str(root/'predictions.jsonl'),'--base-url','http://unused',
                  '--judge-model','fixture','--output',str(root/'judgments.jsonl')]
            answer=({'support':'strong','logic':'fully_coherent'},'{}')
            with patch.object(sys,'argv',argv),patch.object(judge,'call',side_effect=[answer,TimeoutError()]),patch.object(judge.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):cli.main()
            self.assertEqual([x['status'] for x in read_jsonl(root/'judgments.jsonl')],['OK','ERROR'])
            with patch.object(sys,'argv',argv+['--retry-errors']),patch.object(judge,'call',return_value=answer) as call,contextlib.redirect_stdout(io.StringIO()):
                cli.main();self.assertEqual(call.call_count,1)
            rows=read_jsonl(root/'judgments.jsonl')
            self.assertEqual([x['status'] for x in rows],['OK','OK'])
            self.assertEqual(rows[1]['previous_attempts'][0]['status'],'ERROR')
