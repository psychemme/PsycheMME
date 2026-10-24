"""CPU-only contract smoke test; synthetic media and deterministic local endpoints."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from psychemme_eval.cli import main
from psychemme_eval.data import dump_jsonl, read_jsonl, sha256
from psychemme_eval.inference import run
from psychemme_eval.media import frames_1fps
from psychemme_eval.scoring import score_predictions


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for the real frame-extraction contract test')
class PipelineSmoke(unittest.TestCase):
    def test_full_timeline_cap_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);media=root/'long_fixture.mp4'
            subprocess.run(['ffmpeg','-loglevel','error','-f','lavfi','-i','testsrc=size=64x64:rate=2:duration=75','-c:v','mpeg4','-y',str(media)],check=True)
            frames=frames_1fps(media,root/'cache')
            self.assertEqual(len(frames),64)
            self.assertEqual(frames[0].name,'000000.jpg')
            self.assertEqual(frames[-1].name,'000074.jpg')
            self.assertEqual(frames,frames_1fps(media,root/'cache'))
            frames[-1].write_bytes(b'corrupted')
            with self.assertRaises(ValueError):frames_1fps(media,root/'cache')

    def test_media_inference_judge_score(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));captured.append(body)
                if isinstance(body['messages'][-1]['content'],list):
                    answer={'answer':'A','G':'The synthetic scene is red.',**{key:None for key in 'WNTSARM'}}
                else:
                    answer={'support':'strong','logic':'fully_coherent','support_reason':'Synthetic fixture.','logic_reason':'Consistent fixture.'}
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]}).encode())
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);media=root/'video/E0001.mp4';media.parent.mkdir()
                subprocess.run(['ffmpeg','-loglevel','error','-f','lavfi','-i','color=c=red:s=64x64:r=2:d=3','-c:v','mpeg4','-y',str(media)],check=True)
                public=[{'question_id':'Q0001','episode_id':'E0001','media_id':'video/E0001.mp4','category':'G','task_type':'single_choice','question':'What color is visible?','options':['Red','Blue'],'source_dataset':'synthetic','memory_context':None,'memory_dependency':None}]
                truth=[{**public[0],'answer_indices':[0],'answers':['Red'],'reference_answer':None,'evidence':[{'content':'Red synthetic frame.','time':[0,3]}]}]
                asr=[{'episode_id':'E0001','status':'NO_SPEECH','segments':[]}]
                sources=[{'episode_id':'E0001','expected_frozen_media_sha256':sha256(media)}]
                args=argparse.Namespace(output=str(root/'predictions.jsonl'),limit=0,backend='api',model='synthetic-model',revision=None,adapter=None,adapter_revision=None,base_url=f'http://127.0.0.1:{server.server_port}/v1',media_root=str(root),frame_cache=str(root/'cache'),api_key_env='',timeout=5)
                self.assertEqual(run(public,asr,sources,args)['successes'],1)
                self.assertEqual(read_jsonl(args.output)[0]['frame_count_used'],3)
                for name,value in [('inputs',public),('truth',truth)]:dump_jsonl(root/(name+'.jsonl'),value)
                argv=['psychemme','judge','--inputs',str(root/'inputs.jsonl'),'--references',str(root/'truth.jsonl'),'--predictions',args.output,'--output',str(root/'judgments.jsonl'),'--base-url',args.base_url,'--judge-model','synthetic-judge']
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
                scores,summary=score_predictions(public,truth,read_jsonl(args.output),read_jsonl(root/'judgments.jsonl'))
                self.assertEqual(scores[0]['raw'],1);self.assertEqual(scores[0]['verified'],1)
                self.assertEqual(summary['n'],1)
                self.assertNotIn('frozen_evidence',json.dumps(captured[0]))
                self.assertIn('frozen_evidence',captured[1]['messages'][-1]['content'])
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__=='__main__':unittest.main()
