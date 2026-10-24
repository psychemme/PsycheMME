"""Scientific contracts: fixed denominators, no truth leakage, exact scoring."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from psychemme_eval.data import unique_index, validate_public, read_jsonl
from psychemme_eval.protocol import enterprise_parse, enterprise_prompt
from psychemme_eval.scoring import weighted_score, score_predictions, replay_factors, aggregate
from psychemme_eval.inference import request_api


def item(qid='Q0001', kind='single_choice', cap='G'):
    return {'question_id':qid,'episode_id':'E0001','media_id':'video/E0001.mp4','category':cap,
            'task_type':kind,'question':'Which response fits?','options':[] if kind=='open' else ['Wait','Speak','Leave'],
            'source_dataset':'synthetic','memory_context':None,'memory_dependency':None}


def truth(public, indices=None):
    return {**public,'answer_indices':indices or ([0] if public['task_type']!='open' else []),
            'answers':[], 'reference_answer':'Wait for consent.' if public['task_type']=='open' else None,'evidence':[]}


def prediction(qid='Q0001', answer='A'):
    return {'question_id':qid,'status':'OK','format_valid':True,
            'prediction':{'answer':answer,'G':'The participant has not finished speaking.',**{k:None for k in 'WNTSARM'}}}


def judgment(qid='Q0001', basic=1):
    return {'question_id':qid,'status':'OK','basic_answer_score':basic,'support':'strong','logic':'fully_coherent'}


class ScientificContract(unittest.TestCase):
    def test_no_private_reference_in_public_inputs(self):
        with self.assertRaises(ValueError):validate_public([{**item(),'reference_answer':'private'}])

    def test_duplicates_and_foreign_ids_rejected(self):
        with self.assertRaises(ValueError):unique_index([item(),item()])
        with self.assertRaises(ValueError):score_predictions([item()],[truth(item())],[prediction('FOREIGN')],[])

    def test_fixed_denominator_includes_missing_prediction(self):
        public=[item('Q0001'),item('Q0002')]
        records,summary=score_predictions(public,[truth(x) for x in public],[prediction()],[judgment()])
        self.assertEqual([r['raw'] for r in records],[1,0])
        self.assertEqual(summary['n'],2)
        self.assertEqual(summary['capabilities']['G']['raw'],50)
        self.assertIsNone(summary['raw_macro'])

    def test_multiselect_exact_set_not_partial_credit(self):
        public=item(kind='multiple_select')
        records,_=score_predictions([public],[truth(public,[0,2])],[prediction(answer=['A'])],[judgment()])
        self.assertEqual(records[0]['raw'],0)
        records,_=score_predictions([public],[truth(public,[0,2])],[prediction(answer=['C','A'])],[judgment()])
        self.assertEqual(records[0]['raw'],1)

    def test_open_credit_requires_persisted_judge(self):
        public=item(kind='open');pred=prediction(answer='Wait for consent.')
        records,_=score_predictions([public],[truth(public)],[pred],[judgment(basic=.5)])
        self.assertEqual(records[0]['raw'],.5)
        records,_=score_predictions([public],[truth(public)],[pred],[])
        self.assertEqual(records[0]['raw'],0)

    def test_invalid_schema_or_missing_rationale_zero(self):
        pred=prediction();pred['prediction']['G']=None
        records,_=score_predictions([item()],[truth(item())],[pred],[judgment()])
        self.assertEqual(records[0]['raw'],0)
        self.assertEqual(records[0]['verified'],0)

    def test_failed_judge_zero_verified(self):
        records,_=score_predictions([item()],[truth(item())],[prediction()],[{'question_id':'Q0001','status':'ERROR'}])
        self.assertEqual(records[0]['verified'],0)

    def test_rubric_weights_and_finite_values(self):
        self.assertEqual(weighted_score(1,'present','strong','fully_coherent'),1)
        self.assertAlmostEqual(weighted_score(.5,'present','weak','acceptable'),.5*.8/1.44)
        self.assertEqual(weighted_score(1,'missing','insufficient','unevaluable'),0)
        for invalid in [float('nan'),float('inf'),2,True]:
            with self.assertRaises(ValueError):weighted_score(invalid,'present','strong','fully_coherent')

    def test_task_type_prompt_and_open_option_rejection(self):
        self.assertIn('SINGLE-CHOICE',enterprise_prompt(item(),['Wait','Speak','Leave'],'',True))
        self.assertIn('MULTIPLE-SELECT',enterprise_prompt(item(kind='multiple_select'),['Wait','Speak','Leave'],'',True))
        parsed,valid,why=enterprise_parse(json.dumps(prediction()['prediction']),item(kind='open'),set())
        self.assertFalse(valid);self.assertEqual(why,'open_answer_is_option_label')

    def test_api_request_has_expected_public_prompt_and_parameters(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}).encode())
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            prompt=enterprise_prompt(item(),item()['options'],'Public speech only.',True)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'frame.jpg';path.write_bytes(b'synthetic-transport-fixture')
                request_api(f'http://127.0.0.1:{server.server_port}/v1','fixture-model',[path],prompt,'',5)
            body=captured[0]
            self.assertEqual(body['messages'][0]['content'][-1]['text'],prompt)
            self.assertEqual(body['temperature'],0);self.assertEqual(body['seed'],42);self.assertEqual(body['max_tokens'],768)
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertNotIn('reference_answer',json.dumps(body))
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__=='__main__':unittest.main()
