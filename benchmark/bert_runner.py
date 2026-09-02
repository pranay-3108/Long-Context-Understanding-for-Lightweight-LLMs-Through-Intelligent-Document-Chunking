from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "papers" / "bert"
QUESTION_PATH = ROOT / "benchmark" / "bert_benchmark.json"
OUT_DIR = ROOT / "results" / "bert_runs"

SIZES = [8000, 16000, 24000, 32000, 44000]
MODEL_DEFAULT = "granite4:tiny-h"
CHUNK_SIZE = 7000


def load_questions() -> list[dict[str, Any]]:
    data = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    return data["records"]


def ollama_chat(model: str, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Ollama returned a non-zero exit code")
    return proc.stdout.strip(), elapsed


def stage_text(full_text: str, size: int) -> str:
    return full_text[:size]


def fixed_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def make_direct_prompt(context: str, question: str) -> str:
    return f"""You are answering a research-paper question.
Use ONLY the paper context below.
Do not use outside knowledge.
If the provided context does not contain enough information, say that explicitly.
Answer the question directly and accurately.

PAPER CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def make_summary_prompt(chunk: str, index: int, total: int) -> str:
    return f"""Summarize chunk {index} of {total} from a research paper for later question answering.
Preserve factual details, definitions, methodology, experimental findings, numerical values, equations, comparisons, limitations, and conclusions.
Do not invent information. Keep named entities and numbers exact.

CHUNK:
{chunk}

SUMMARY:
"""


def make_aggregate_prompt(summaries: list[str]) -> str:
    joined = "\n\n===== NEXT CHUNK SUMMARY =====\n\n".join(summaries)
    return f"""Combine the following chunk summaries into one faithful research-paper knowledge record.
Preserve important technical details, equations, numerical results, experimental comparisons, caveats, and relationships across sections.
Do not invent information or add outside knowledge.

CHUNK SUMMARIES:
{joined}

AGGREGATED PAPER RECORD:
"""


def read_paper(size: int) -> str:
    candidate = PAPER_DIR / f"bert_{size//1000}k.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    fallback = PAPER_DIR / "bert_full.txt"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing {candidate} and {fallback}. Run prepare_bert.py first.")
    text = fallback.read_text(encoding="utf-8")
    return stage_text(text, size)


def run_stage(model: str, size: int, mode: str) -> dict[str, Any]:
    text = read_paper(size)
    records = []
    total_time = 0.0
    chunk_count = 0
    summary_time = 0.0
    aggregation_time = 0.0

    if mode == "direct":
        answer_context = text
    else:
        chunks = fixed_chunks(text)
        chunk_count = len(chunks)
        summaries: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            summary, elapsed = ollama_chat(model, make_summary_prompt(chunk, i, len(chunks)))
            summaries.append(summary)
            summary_time += elapsed
        answer_context, aggregation_time = ollama_chat(model, make_aggregate_prompt(summaries))
        total_time += summary_time + aggregation_time

    for row in records:
        started = time.perf_counter()
        answer, elapsed = ollama_chat(model, make_direct_prompt(answer_context, row["question"]))
        if mode == "direct":
            total_time += elapsed
        records_out = {
            "question_id": row["question_id"],
            "question": row["question"],
            "category": row.get("category", ""),
            "ground_truth_answer": row["ground_truth_answer"],
            "model_answer": answer,
            "model": model,
            "parameters": "7B" if "7b" in model.lower() or "tiny-h" in model.lower() else "unknown",
            "paper": "BERT",
            "paper_length_chars": len(text),
            "mode": mode,
            "answer_time_sec": elapsed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        records_out["stage_answer_time_sec"] = time.perf_counter() - started
        records.append(records_out)

    result = {
        "paper": "BERT",
        "model": model,
        "mode": mode,
        "paper_length_chars": len(text),
        "requested_stage_chars": size,
        "chunk_size_chars": CHUNK_SIZE if mode == "fixed_chunk" else None,
        "chunk_count": chunk_count,
        "chunk_summary_time_sec": round(summary_time, 3),
        "aggregation_time_sec": round(aggregation_time, 3),
        "answer_time_sec": round(sum(r["answer_time_sec"] for r in records), 3),
        "total_time_sec": round(total_time, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BERT 55-question understanding benchmark through Ollama.")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--mode", choices=["direct", "fixed_chunk"], default="direct")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        if size not in SIZES:
            raise SystemExit(f"Unsupported size {size}. Use one of: {SIZES}")
        out = OUT_DIR / f"bert_{args.mode}_{size//1000}k.json"
        if out.exists() and not args.force:
            print(f"SKIP existing: {out}")
            continue
        print(f"RUN {args.mode} {size} chars with {args.model}")
        result = run_stage(args.model, size, args.mode)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"SAVED {out}")


if __name__ == "__main__":
    main()
