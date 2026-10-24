"""Check the 18 released traces against references and archived numeric results."""
from pathlib import Path
import argparse
import json
import math
from psychemme_eval.data import read_jsonl
from psychemme_eval.scoring import score_predictions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.'));args=parser.parse_args()
    root=args.root
    public={r['question_id']:r for r in read_jsonl(root/'dataset/text_subset/model_inputs.jsonl')}
    truth={r['question_id']:r for r in read_jsonl(root/'dataset/text_subset/references_EVALUATOR_ONLY.jsonl')}
    details=read_jsonl(root/'results/detailed_outputs_subset.jsonl')
    expected={(r['model'],r['question_id']):r for r in read_jsonl(root/'results/per_question_scores.jsonl')}
    checked=0
    for model in sorted({r['model'] for r in details}):
        selected=[r for r in details if r['model']==model]
        ids=[r['question_id'] for r in selected]
        preds=[{'question_id':r['question_id'],'status':r['inference_status'],'format_valid':r['format_valid'],'prediction':r['prediction']} for r in selected]
        judges=[{'question_id':r['question_id'],**r['judgment']} for r in selected]
        actual,_=score_predictions([public[q] for q in ids],[truth[q] for q in ids],preds,judges)
        for row in actual:
            saved=expected[model,row['question_id']]
            assert all(math.isclose(row[k],saved[k],abs_tol=1e-12,rel_tol=0) for k in ['raw','verified']),(model,row['question_id'])
            checked+=1
    print(json.dumps({'status':'PASS','detailed_records_checked':checked,'scope':'Choice correctness from released gold; open credit uses persisted judge; all 18 raw/verified scores match archive'}))


if __name__=='__main__':main()
