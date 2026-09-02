from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = Path("/mnt/data/0167d5d0-9d33-4e4c-b119-61a4a32e807e.json")
if not source.exists():
    raise SystemExit(f"Combined result file not found: {source}")

items = json.loads(source.read_text(encoding="utf-8"))
by_stage = {}
for item in items:
    by_stage.setdefault(int(item["stage_k"]), []).append(item)

base = ROOT / "results" / "bert" / "models" / "granite3_3_2b" / "fixed_chunk_7k"
base.mkdir(parents=True, exist_ok=True)

for stage, answers in sorted(by_stage.items()):
    stage_dir = base / f"{stage}k"
    stage_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "model": "granite3.3:2b",
        "model_label": "granite3_3_2b",
        "method_display": "Fixed 7k chunking",
        "mode": "fixed",
        "paper": f"papers/bert/bert_{stage}k.txt",
        "paper_characters": stage * 1000,
        "chunk_size_characters": 7000,
        "stage_k": stage,
        "question_count": len(answers),
        "answers": [
            {
                "question_id": x["question_id"],
                "question": x.get("question", ""),
                "category": x.get("category", ""),
                "ground_truth_answer": x.get("ground_truth_answer", ""),
                "answer": x.get("model_answer", ""),
                "answer_f1": x.get("answer_f1"),
            }
            for x in answers
        ],
        "source": "User-provided combined BERT stage results",
    }
    (stage_dir / "result.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Imported {len(items)} answers into {len(by_stage)} BERT stages.")
