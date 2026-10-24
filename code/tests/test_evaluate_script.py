"""Exercise the public shell entry point without remote downloads or API keys."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest


class EvaluateScript(unittest.TestCase):
    def test_sample_resume_environment_and_failure_stop(self):
        source=Path(__file__).resolve().parents[2]/'scripts/evaluate.sh'
        if not source.is_file():
            self.skipTest('Repository scripts are required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir();(root/'bin').mkdir()
            (root/'scripts/evaluate.sh').write_text(source.read_text())
            (root/'.env').write_text('MODEL_BASE_URL=http://unused/v1\nMODEL_ID=from-file\nJUDGE_BASE_URL=http://unused/v1\nJUDGE_MODEL=judge\nLIMIT=0\n')
            command=root/'bin/psychemme'
            command.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom unittest.mock import patch\nfrom psychemme_eval import cli\nwith patch.object(cli,"execute",return_value=None):cli.main()\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif sys.argv[1]==os.environ.get("FAIL_STAGE"):sys.exit(1)\n')
            command.chmod(0o755)
            ffmpeg=root/'bin/ffmpeg';ffmpeg.write_text('#!/bin/sh\nexit 0\n');ffmpeg.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                 'MODEL_ID':'from-shell','RUN_DIR':str(root/'run'),
                 'CALL_LOG':str(root/'calls.jsonl'),'RETRY_ERRORS':'1'}
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh'),'10'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([x[0] for x in calls],['download','prepare','infer','judge','score'])
            self.assertIn('--metadata-only',calls[0])
            self.assertEqual(calls[1][-1],'10')
            for row in calls[2:4]:
                self.assertIn('--resume',row);self.assertIn('--retry-errors',row)
            self.assertEqual(calls[2][calls[2].index('--model')+1],'from-shell')
            (root/'calls.jsonl').unlink()
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([x[0] for x in calls],['download','infer','judge','score'])
            self.assertNotIn('--metadata-only',calls[0])
            self.assertIn('benchmark',calls[0])
            env['FAIL_STAGE']='infer';(root/'calls.jsonl').unlink()
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh'),'10'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([x[0] for x in calls],['download','prepare','infer'])
