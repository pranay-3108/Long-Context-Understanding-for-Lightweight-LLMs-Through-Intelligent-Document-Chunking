import argparse, json
from pathlib import Path
from collections import Counter
import re, string

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "bert" / "models"

def norm(text):
    text = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def f1(pred, gold):
    p=norm(pred).split(); g=norm(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    c=sum((Counter(p)&Counter(g)).values())
    if not c: return 0.0
    return round(2*(c/len(p))*(c/len(g))/((c/len(p))+(c/len(g))),4)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--mode', choices=['fixed_chunk_7k','direct'], required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[8,16,24,32,44])
    args=ap.parse_args()
    root=RESULT_ROOT/args.label/args.mode
    rows=[]
    for size in args.sizes:
        p=root/f'{size}k/answers.json'
        if not p.exists():
            print('MISSING', p); continue
        data=json.loads(p.read_text(encoding='utf-8'))
        for r in data.get('answers',[]):
            score=f1(r.get('answer',''), r.get('ground_truth_answer',''))
            rows.append({'stage_k':size,'question_id':r.get('question_id'),'category':r.get('category',''),'answer_f1':score,'model_answer':r.get('answer',''),'ground_truth_answer':r.get('ground_truth_answer','')})
    out=root/'comparison.json'
    out.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    summary=[]
    for size in args.sizes:
        vals=[r['answer_f1'] for r in rows if r['stage_k']==size]
        summary.append({'stage_k':size,'question_count':len(vals),'average_answer_f1':round(sum(vals)/len(vals),4) if vals else 0})
    (root/'comparison_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for x in summary: print(f"{x['stage_k']}k: {x['question_count']} answers, avg Answer F1={x['average_answer_f1']}")

if __name__=='__main__': main()
