import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / 'papers' / 'bert'
QUESTION_FILE = ROOT / 'benchmark' / 'bert_questions.txt'
BENCHMARK_FILE = ROOT / 'benchmark' / 'bert_benchmark.json'
OUT_ROOT = ROOT / 'results' / 'bert' / 'models'
SIZES = [8, 16, 24, 32, 44]


def ollama_generate(model: str, prompt: str):
    payload = json.dumps({'model': model, 'prompt': prompt, 'stream': False}).encode('utf-8')
    req = urllib.request.Request('http://localhost:11434/api/generate', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read().decode('utf-8'))
    return obj.get('response', '')


def read_questions():
    # Use the canonical JSON benchmark because it contains the exact 55
    # questions and their ground-truth answers. The txt file uses Q01.
    data = json.loads(BENCHMARK_FILE.read_text(encoding='utf-8'))
    rows = []
    for row in data.get("records", []):
        rows.append({
            "question_id": row.get("question_id"),
            "question": row.get("question", ""),
            "category": row.get("category", ""),
            "ground_truth_answer": row.get("ground_truth_answer", ""),
        })
    return rows


def build_context(text, mode):
    if mode == 'direct':
        return text
    # Fixed 7k chunking: summarize chunks first, then ask questions against the aggregate.
    chunks = [text[i:i+7000] for i in range(0, len(text), 7000)]
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        prompt = f'''You are preparing a faithful research-paper evidence summary for later question answering.\n\nRules:\n- Use ONLY the supplied chunk.\n- Preserve technical claims, definitions, equations, experimental results, and numbers.\n- Do not invent missing information.\n- If something is absent, do not add it.\n- Keep important details, not generic prose.\n\nCHUNK {idx}:\n{chunk}\n\nSUMMARY:'''
        summaries.append(ollama_generate(CURRENT_MODEL, prompt))
    return '\n\n--- CHUNK SUMMARY ---\n\n'.join(summaries)


def main():
    global CURRENT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Ollama model name')
    ap.add_argument('--label', required=True, help='Output folder label')
    ap.add_argument('--mode', choices=['direct', 'fixed'], default='direct')
    ap.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    args = ap.parse_args()
    CURRENT_MODEL = args.model

    questions = read_questions()
    if len(questions) != 55:
        raise SystemExit(f'Expected 55 canonical questions, found {len(questions)}. Check benchmark/bert_benchmark.json.')

    for size in args.sizes:
        paper_path = PAPER_DIR / f'bert_{size}k.txt'
        text = paper_path.read_text(encoding='utf-8')
        folder = OUT_ROOT / args.label / ('fixed_chunk_7k' if args.mode == 'fixed' else 'direct') / f'{size}k'
        folder.mkdir(parents=True, exist_ok=True)
        start = time.time()
        context = build_context(text, args.mode)
        answers = []
        for i, row in enumerate(questions, 1):
            q = row['question']
            prompt = f'''Answer the question using ONLY the research-paper context below.\n\nDo not use outside knowledge. Do not guess. If the available context does not contain enough information, say exactly: "The provided paper context does not contain enough information to answer this."\n\nPAPER CONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nANSWER:'''
            t0 = time.time()
            try:
                ans = ollama_generate(args.model, prompt)
                error = None
            except Exception as e:
                ans = ''
                error = str(e)
            answers.append({'question_id': row.get('question_id', i), 'question': q, 'category': row.get('category', ''), 'ground_truth_answer': row.get('ground_truth_answer', ''), 'answer': ans, 'error': error, 'time_sec': round(time.time()-t0, 3)})
            print(f'{args.label} {args.mode} {size}k: {i}/{len(questions)}')
        meta = {
            'model': args.model,
            'model_label': args.label,
            'mode': args.mode,
            'paper': str(paper_path.relative_to(ROOT)),
            'paper_characters': len(text),
            'chunk_size_characters': 7000 if args.mode == 'fixed' else None,
            'question_count': len(answers),
            'total_time_sec': round(time.time()-start, 3),
            'answers': answers,
        }
        (folder / 'answers.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        (folder / 'answers.txt').write_text('\n\n'.join(f"{x['question_id']}\n{x['answer']}" for x in answers), encoding='utf-8')
        print('SAVED', folder)


if __name__ == '__main__':
    main()
