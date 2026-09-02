from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import ollama
import psutil

from .evaluation_result import EvaluationResult

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers" / "qasper"
METADATA_DIR = PAPERS_DIR / "metadata"
EVALUATIONS_DIR = ROOT / "benchmark" / "evaluations"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "granite": {
        "ollama_model": "granite3.3:2b",
        "chunk_size": 7000,
        "options": {"num_ctx": 16384, "temperature": 0},
    },
    "qwen": {
        "ollama_model": "qwen2.5:3b",
        "chunk_size": 7000,
        "options": {"num_ctx": 16384, "temperature": 0},
    },
    "deepseek": {
        "ollama_model": "deepseek-r1:1.5b",
        "chunk_size": 7000,
        "options": {"num_ctx": 16384, "temperature": 0},
    },
}

SUPPORTED_MODES = {"direct", "chunk"}


def _normalize_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized not in MODEL_CONFIGS:
        supported = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unsupported model '{model}'. Supported models: {supported}")
    return normalized


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")
    return normalized


def _parse_stringified_structure(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue

    return value


def _coerce_evidence_list(value: Any) -> List[str]:
    parsed = _parse_stringified_structure(value)

    if parsed is None:
        return []
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed if str(item).strip()]
    if isinstance(parsed, dict):
        items: List[str] = []
        for key in ("highlighted_evidence", "evidence", "extractive_spans", "free_form_answer"):
            nested = parsed.get(key)
            items.extend(_coerce_evidence_list(nested))
        return items

    return [str(parsed)]


def _extract_gold_answer(raw_gold_answer: Any) -> str:
    parsed = _parse_stringified_structure(raw_gold_answer)

    if isinstance(parsed, dict):
        free_form_answer = str(parsed.get("free_form_answer", "")).strip()
        if free_form_answer:
            return free_form_answer

        extractive_spans = _coerce_evidence_list(parsed.get("extractive_spans"))
        if extractive_spans:
            return " ".join(extractive_spans)

        yes_no = parsed.get("yes_no")
        if yes_no is not None:
            return str(yes_no)

        unanswerable = parsed.get("unanswerable")
        if unanswerable is True:
            return "unanswerable"

    if isinstance(parsed, (list, tuple)):
        return " ".join(str(item) for item in parsed if str(item).strip()).strip()

    return str(parsed).strip()


def _extract_gold_evidence(raw_gold_answer: Any, fallback_evidence: Any) -> List[str]:
    parsed = _parse_stringified_structure(raw_gold_answer)
    if isinstance(parsed, dict):
        highlighted = _coerce_evidence_list(parsed.get("highlighted_evidence"))
        if highlighted:
            return highlighted

        evidence = _coerce_evidence_list(parsed.get("evidence"))
        if evidence:
            return evidence

    return _coerce_evidence_list(fallback_evidence)


def load_qasper_paper(paper_id: str) -> Dict[str, Any]:
    paper_path = PAPERS_DIR / f"{paper_id}.txt"
    metadata_path = METADATA_DIR / f"{paper_id}.json"

    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paper_text = paper_path.read_text(encoding="utf-8")

    return {
        "paper_name": paper_id,
        "paper_id": metadata.get("paper_id", paper_id),
        "paper_path": paper_path,
        "metadata_path": metadata_path,
        "paper_text": paper_text,
        "metadata": metadata,
    }


def _resolve_question_payload(paper: Dict[str, Any], question_id: int) -> Dict[str, Any]:
    metadata = paper["metadata"]
    questions = metadata.get("questions", [])
    gold_answers = metadata.get("gold_answers", [])
    evidence_list = metadata.get("evidence", [])

    if question_id < 1 or question_id > len(questions):
        raise IndexError(f"Question id {question_id} is out of range for {paper['paper_name']}")

    index = question_id - 1
    raw_gold_answer = gold_answers[index] if index < len(gold_answers) else ""
    fallback_evidence = evidence_list[index] if index < len(evidence_list) else evidence_list

    return {
        "question_id": question_id,
        "question": questions[index],
        "gold_answer": _extract_gold_answer(raw_gold_answer),
        "evidence": _extract_gold_evidence(raw_gold_answer, fallback_evidence),
    }


def _get_ollama_ram_mb() -> float:
    highest = 0.0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"]
            if name and "ollama" in name.lower():
                highest = max(highest, proc.info["memory_info"].rss / (1024 * 1024))
        except Exception:
            continue
    return round(highest, 2)


def _chat(model_key: str, prompt: str) -> tuple[str, float, float]:
    config = MODEL_CONFIGS[model_key]
    ram_before = _get_ollama_ram_mb()
    start = time.time()
    response = ollama.chat(
        model=config["ollama_model"],
        options=config["options"],
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = round(time.time() - start, 2)
    ram_after = _get_ollama_ram_mb()
    ram_usage = round(max(ram_before, ram_after), 2)
    return response["message"]["content"].strip(), elapsed, ram_usage


def _build_direct_prompt(question: str, paper_text: str) -> str:
    return (
        "Read the research paper carefully and answer the question using evidence from the paper.\n\n"
        f"Question: {question}\n\n"
        "Answer in 2-5 sentences. If the paper does not contain the answer, say that clearly.\n\n"
        f"Document:\n{paper_text}\n"
    )


def _build_chunk_summary_prompt(question: str, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return (
        "You are reading one chunk of a research paper in order to answer a question.\n\n"
        f"Question: {question}\n"
        f"Chunk: {chunk_index} of {total_chunks}\n\n"
        "Extract only information from this chunk that helps answer the question.\n"
        "If the chunk appears irrelevant, say \"No relevant evidence in this chunk.\"\n\n"
        f"Paper Chunk:\n{chunk_text}\n"
    )


def _build_chunk_aggregation_prompt(question: str, chunk_notes: Iterable[str]) -> str:
    joined_notes = "\n\n".join(chunk_notes)
    return (
        "The following notes were extracted from different chunks of the same research paper.\n"
        "Synthesize them into one final answer to the question.\n\n"
        f"Question: {question}\n\n"
        "Answer in 2-5 sentences. Use only the provided notes. If they are insufficient, say so clearly.\n\n"
        f"Chunk Notes:\n{joined_notes}\n"
    )


def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
    return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)] or [text]


def _run_direct_answer(model_key: str, paper_text: str, question: str) -> tuple[str, float, float]:
    prompt = _build_direct_prompt(question, paper_text)
    return _chat(model_key, prompt)


def _run_chunk_answer(model_key: str, paper_text: str, question: str) -> tuple[str, float, float]:
    config = MODEL_CONFIGS[model_key]
    chunks = _split_into_chunks(paper_text, config["chunk_size"])

    chunk_notes: List[str] = []
    total_time = 0.0
    peak_ram = 0.0

    for index, chunk_text in enumerate(chunks, start=1):
        prompt = _build_chunk_summary_prompt(question, chunk_text, index, len(chunks))
        chunk_answer, elapsed, ram_usage = _chat(model_key, prompt)
        chunk_notes.append(f"Chunk {index}:\n{chunk_answer}")
        total_time += elapsed
        peak_ram = max(peak_ram, ram_usage)

    final_prompt = _build_chunk_aggregation_prompt(question, chunk_notes)
    final_answer, elapsed, ram_usage = _chat(model_key, final_prompt)
    total_time += elapsed
    peak_ram = max(peak_ram, ram_usage)

    return final_answer, round(total_time, 2), round(peak_ram, 2)


def evaluate_question(
    paper: Dict[str, Any] | str,
    question: Dict[str, Any] | str,
    model: str,
    mode: str,
    *,
    paper_id: str | None = None,
    question_id: int | None = None,
    gold_answer: str | None = None,
    evidence: List[str] | None = None,
) -> EvaluationResult:
    model_key = _normalize_model(model)
    mode_key = _normalize_mode(mode)

    if isinstance(paper, str):
        paper_payload = load_qasper_paper(paper)
    else:
        paper_payload = paper

    if isinstance(question, dict):
        question_text = str(question["question"])
        resolved_question_id = int(question.get("question_id", question_id or 0))
        resolved_gold_answer = str(question.get("gold_answer", gold_answer or ""))
        resolved_evidence = list(question.get("evidence", evidence or []))
    else:
        question_text = question
        resolved_question_id = 0 if question_id is None else question_id
        resolved_gold_answer = "" if gold_answer is None else gold_answer
        resolved_evidence = [] if evidence is None else evidence

    resolved_paper_id = paper_id or paper_payload["paper_name"]
    paper_text = paper_payload["paper_text"]

    if mode_key == "direct":
        model_answer, execution_time, ram_usage = _run_direct_answer(model_key, paper_text, question_text)
    else:
        model_answer, execution_time, ram_usage = _run_chunk_answer(model_key, paper_text, question_text)

    return EvaluationResult(
        paper_id=resolved_paper_id,
        question_id=resolved_question_id,
        question=question_text,
        gold_answer=resolved_gold_answer,
        model_answer=model_answer,
        evidence=resolved_evidence,
        paper_length=len(paper_text),
        execution_time=execution_time,
        ram_usage=ram_usage,
        mode=mode_key,
        model=model_key,
        metadata={
            "paper_name": paper_payload["paper_name"],
            "source_paper_id": paper_payload["paper_id"],
            "ollama_model": MODEL_CONFIGS[model_key]["ollama_model"],
        },
    )


def save_evaluation_result(
    result: EvaluationResult,
    *,
    output_path: Path | None = None,
    save_latest_alias: bool = False,
) -> Path:
    target_path = output_path
    if target_path is None:
        target_dir = EVALUATIONS_DIR / result.model / result.mode
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{result.paper_id}_question_{result.question_id:04d}_evaluation.json"
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if save_latest_alias:
        latest_path = target_path.parent / "evaluation.json"
        latest_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return target_path


def run_qasper_evaluation(paper_name: str, question_id: int, model: str, mode: str) -> EvaluationResult:
    paper = load_qasper_paper(paper_name)
    question_payload = _resolve_question_payload(paper, question_id)

    return evaluate_question(paper, question_payload, model, mode, paper_id=paper["paper_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one QASPER evaluation and save it as JSON.")
    parser.add_argument("--paper", default="paper_0001", help="Paper filename stem, for example paper_0001.")
    parser.add_argument("--question-id", type=int, default=1, help="1-based question index.")
    parser.add_argument("--model", default="granite", choices=sorted(MODEL_CONFIGS), help="Model backend.")
    parser.add_argument("--mode", default="direct", choices=sorted(SUPPORTED_MODES), help="Evaluation mode.")
    parser.add_argument(
        "--output",
        help="Optional explicit JSON output path. Defaults to benchmark/evaluations/<model>/<mode>/...",
    )
    parser.add_argument(
        "--save-latest-alias",
        action="store_true",
        help="Also save benchmark/evaluations/<model>/<mode>/evaluation.json for quick inspection.",
    )
    args = parser.parse_args()

    result = run_qasper_evaluation(args.paper, args.question_id, args.model, args.mode)
    saved_path = save_evaluation_result(
        result,
        output_path=Path(args.output) if args.output else None,
        save_latest_alias=args.save_latest_alias,
    )

    print(f"Saved evaluation to {saved_path}")
    if args.save_latest_alias:
        print(f"Saved latest evaluation alias to {saved_path.parent / 'evaluation.json'}")


if __name__ == "__main__":
    main()
