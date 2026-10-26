"""Exercise the public shell entry point without remote downloads or API keys."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest


class EvaluateScript(unittest.TestCase):
    def test_full_evaluation_resume_environment_and_failure_stop(self):
        source=Path(__file__).resolve().parents[2]/'scripts/evaluate.sh'
        if not source.is_file():
            self.skipTest('Repository scripts are required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir();(root/'bin').mkdir()
            (root/'scripts/evaluate.sh').write_text(source.read_text())
            (root/'.env').write_text('MODEL_BASE_URL=http://unused/v1\nMODEL_ID=from-file\nJUDGE_BASE_URL=http://unused/v1\nJUDGE_MODEL=judge\n')
            command=root/'bin/psychemme'
            command.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom unittest.mock import patch\nfrom psychemme_eval import cli\nwith patch.object(cli,"execute",return_value=None):cli.main()\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif sys.argv[1]==os.environ.get("FAIL_STAGE"):sys.exit(1)\n')
            command.chmod(0o755)
            ffmpeg=root/'bin/ffmpeg';ffmpeg.write_text('#!/bin/sh\nexit 0\n');ffmpeg.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                 'MODEL_ID':'from-shell','RUN_DIR':str(root/'run'),
                 'CALL_LOG':str(root/'calls.jsonl'),'RETRY_ERRORS':'1'}
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([x[0] for x in calls],['download','infer','judge','score'])
            self.assertNotIn('--metadata-only',calls[0])
            self.assertIn('benchmark',calls[0])
            self.assertTrue(calls[1][calls[1].index('--sources')+1].endswith('/metadata/media.local.jsonl'))
            for row in calls[1:3]:
                self.assertIn('--resume',row);self.assertIn('--retry-errors',row)
            self.assertEqual(calls[1][calls[1].index('--model')+1],'from-shell')
            env['FAIL_STAGE']='infer';(root/'calls.jsonl').unlink()
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([x[0] for x in calls],['download','infer'])
            (root/'calls.jsonl').unlink()
            result=subprocess.run(['bash',str(root/'scripts/evaluate.sh'),'10'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Usage:',result.stderr)
            self.assertFalse((root/'calls.jsonl').exists())

    def test_download_script_loads_official_credentials_and_preserves_shell_overrides(self):
        source=Path(__file__).resolve().parents[2]/'scripts/download.sh'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir();(root/'bin').mkdir()
            (root/'scripts/download.sh').write_text(source.read_text())
            (root/'.env').write_text('EGO4D_ACCESS_KEY_ID=example-id\nEGO4D_SECRET_ACCESS_KEY=example-secret\nEGO4D_PROFILE=from-file\n')
            command=root/'bin/psychemme'
            command.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom unittest.mock import patch\nfrom psychemme_eval import cli\nwith patch.object(cli,"execute",return_value=None):cli.main()\nassert os.environ["EGO4D_ACCESS_KEY_ID"]=="example-id"\nassert os.environ["EGO4D_SECRET_ACCESS_KEY"]=="example-secret"\nassert os.environ["EGO4D_PROFILE"]=="from-shell"\nwith open(os.environ["CALL_LOG"],"w") as f:json.dump(sys.argv[1:],f)\n')
            command.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                 'EGO4D_PROFILE':'from-shell','DATA_DIR':str(root/'benchmark'),
                 'CALL_LOG':str(root/'calls.json')}
            result=subprocess.run(['bash',str(root/'scripts/download.sh'),'--metadata-only'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=json.loads((root/'calls.json').read_text())
            self.assertEqual(args[:3],['download','--bundle','benchmark'])
            self.assertIn('--metadata-only',args)
            self.assertEqual(args[args.index('--output')+1],str(root/'benchmark'))
            self.assertNotIn('example-secret',result.stdout+result.stderr)
