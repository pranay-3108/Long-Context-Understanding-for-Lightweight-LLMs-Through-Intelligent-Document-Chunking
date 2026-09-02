from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from benchmark.answer_f1 import compute_answer_f1
from benchmark.evaluation import (
    _extract_gold_answer,
    _extract_gold_evidence,
    _get_ollama_ram_mb,
    load_qasper_paper,
)
from benchmark.evaluation_result import EvaluationResult
from models.granite.aggregate_paper import aggregate_summary_files
from models.granite.chunk_paper import chunk_text_to_files
from models.granite.run_qasper_single import run_qasper_direct_question
from models.granite.summarize_chunk import summarize_chunk_files

ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = ROOT / "papers" / "qasper" / "metadata"
EVALUATIONS_DIR = ROOT / "benchmark" / "evaluations" / "granite"
RESULTS_DIR = ROOT / "results"
VALIDATION_REPORT_PATH = ROOT / "validation_report.md"
DIRECT_TIMEOUT_SECONDS = 300
CHUNK_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_RETRIES = 2
PLACEHOLDER_PATTERNS = (
    "placeholder",
    "lorem ipsum",
    "dummy answer",
    "todo",
    "tbd",
)


def _list_selected_papers(limit: int = 5) -> list[str]:
    return [path.stem for path in sorted(METADATA_DIR.glob("paper_*.json"))[:limit]]


def _read_question_payload(paper_name: str, question_id: int) -> dict[str, Any]:
    metadata_path = METADATA_DIR / f"{paper_name}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    questions = metadata.get("questions", [])
    gold_answers = metadata.get("gold_answers", [])
    evidence_items = metadata.get("evidence", [])

    index = question_id - 1
    raw_gold_answer = gold_answers[index] if index < len(gold_answers) else ""
    fallback_evidence = evidence_items[index] if index < len(evidence_items) else evidence_items

    return {
        "question_id": question_id,
        "question": questions[index],
        "gold_answer": _extract_gold_answer(raw_gold_answer),
        "evidence": _extract_gold_evidence(raw_gold_answer, fallback_evidence),
    }


def _contains_placeholder(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(pattern in normalized for pattern in PLACEHOLDER_PATTERNS)


def _normalize_for_copy_check(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _looks_like_context_limit_error(message: str) -> bool:
    lowered = message.lower()
    return "context" in lowered or "token" in lowered or "length" in lowered


def _run_with_timeout(callable_obj: Callable[[], Any], timeout_seconds: int) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callable_obj)
        return future.result(timeout=timeout_seconds)


def _versioned_output_path(base_dir: Path, paper_name: str) -> Path:
    base_path = base_dir / f"{paper_name}.json"
    if not base_path.exists():
        return base_path

    version = 2
    while True:
        candidate = base_dir / f"{paper_name}_v{version}.json"
        if not candidate.exists():
            return candidate
        version += 1


def _load_existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_paper_output(base_dir: Path, paper_name: str, expected_questions: int) -> tuple[Path, dict[str, Any] | None]:
    primary_path = base_dir / f"{paper_name}.json"
    existing_payload = _load_existing_payload(primary_path)
    if existing_payload is not None and len(existing_payload.get("results", [])) < expected_questions:
        return primary_path, existing_payload
    return _versioned_output_path(base_dir, paper_name), None


def _versioned_summary_csv_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_path = RESULTS_DIR / "granite_5paper_summary.csv"
    if not base_path.exists():
        return base_path

    version = 2
    while True:
        candidate = RESULTS_DIR / f"granite_5paper_summary_v{version}.csv"
        if not candidate.exists():
            return candidate
        version += 1


def _safe_json_check(path: Path) -> tuple[bool, str | None]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _status_from_answer(model_answer: str, gold_answer: str) -> str:
    if model_answer is None or not str(model_answer).strip():
        return "FAILED_EMPTY_RESPONSE"
    if _contains_placeholder(model_answer):
        return "FAILED_PLACEHOLDER_RESPONSE"
    if gold_answer and _normalize_for_copy_check(model_answer) == _normalize_for_copy_check(gold_answer):
        return "FAILED_COPIED_GOLD"
    return "SUCCESS"


def _build_result(
    *,
    paper_name: str,
    question_payload: dict[str, Any],
    mode: str,
    model_answer: str,
    execution_time: float | None,
    ram_usage: float | None,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        paper_id=paper_name,
        question_id=question_payload["question_id"],
        question=question_payload["question"],
        gold_answer=question_payload["gold_answer"],
        model_answer=model_answer,
        evidence=question_payload["evidence"],
        paper_length=len(load_qasper_paper(paper_name)["paper_text"]),
        execution_time=execution_time,
        ram_usage=ram_usage,
        mode=mode,
        model="granite",
        status=status,
        metadata=metadata or {},
    )


def _run_direct_question(paper_name: str, question_payload: dict[str, Any], paper_text: str) -> tuple[EvaluationResult, int]:
    retries = 0
    last_ram = None

    while True:
        try:
            ram_before = _get_ollama_ram_mb()
            start = time.time()
            answer = _run_with_timeout(
                lambda: run_qasper_direct_question(paper_text, question_payload["question"]),
                DIRECT_TIMEOUT_SECONDS,
            )
            elapsed = round(time.time() - start, 2)
            ram_after = _get_ollama_ram_mb()
            last_ram = round(max(ram_before, ram_after), 2)
            status = _status_from_answer(answer, question_payload["gold_answer"])
            result = _build_result(
                paper_name=paper_name,
                question_payload=question_payload,
                mode="direct",
                model_answer=answer or "",
                execution_time=elapsed,
                ram_usage=last_ram,
                status=status,
            )
            result.metadata = {
                **(result.metadata or {}),
                "retries": retries,
                "json_saved": False,
                "used_chunk_pipeline": False,
            }
            return result, retries
        except TimeoutError:
            retries += 1
            if retries > MAX_TIMEOUT_RETRIES:
                result = _build_result(
                    paper_name=paper_name,
                    question_payload=question_payload,
                    mode="direct",
                    model_answer="",
                    execution_time=None,
                    ram_usage=last_ram,
                    status="FAILED_TIMEOUT",
                    metadata={"retries": retries, "used_chunk_pipeline": False},
                )
                return result, retries
        except Exception as exc:
            status = "DIRECT_CONTEXT_LIMIT" if _looks_like_context_limit_error(str(exc)) else "FAILED_JSON"
            result = _build_result(
                paper_name=paper_name,
                question_payload=question_payload,
                mode="direct",
                model_answer="",
                execution_time=None,
                ram_usage=last_ram,
                status=status,
                metadata={"error": str(exc), "retries": retries, "used_chunk_pipeline": False},
            )
            return result, retries


def _run_chunk_question(
    paper_name: str,
    question_payload: dict[str, Any],
    paper_text: str,
    *,
    run_tag: str,
) -> tuple[EvaluationResult, int]:
    retries = 0
    artifact_name = f"{paper_name}__q{question_payload['question_id']:04d}__{run_tag}"

    while True:
        try:
            ram_before = _get_ollama_ram_mb()
            start = time.time()

            chunk_files = chunk_text_to_files(artifact_name, paper_text)
            summary_files, summarize_time, chunk_failures = summarize_chunk_files(
                artifact_name,
                question=question_payload["question"],
            )

            if not summary_files:
                result = _build_result(
                    paper_name=paper_name,
                    question_payload=question_payload,
                    mode="chunk",
                    model_answer="",
                    execution_time=round(time.time() - start, 2),
                    ram_usage=_get_ollama_ram_mb(),
                    status="FAILED_CHUNK",
                    metadata={
                        "chunk_failures": chunk_failures,
                        "retries": retries,
                        "used_chunk_pipeline": True,
                        "chunk_files_created": len(chunk_files),
                    },
                )
                return result, retries

            answer, aggregate_time, output_paths = _run_with_timeout(
                lambda: aggregate_summary_files(artifact_name, question=question_payload["question"]),
                CHUNK_TIMEOUT_SECONDS,
            )
            elapsed = round(time.time() - start, 2)
            ram_after = _get_ollama_ram_mb()
            peak_ram = round(max(ram_before, ram_after), 2)
            status = _status_from_answer(answer, question_payload["gold_answer"])
            result = _build_result(
                paper_name=paper_name,
                question_payload=question_payload,
                mode="chunk",
                model_answer=answer or "",
                execution_time=elapsed,
                ram_usage=peak_ram,
                status=status,
                metadata={
                    "retries": retries,
                    "used_chunk_pipeline": True,
                    "chunk_failures": chunk_failures,
                    "chunk_files_created": len(chunk_files),
                    "summary_files_created": len(summary_files),
                    "summarize_time": summarize_time,
                    "aggregate_time": aggregate_time,
                    "pipeline_outputs": [str(path) for path in output_paths],
                },
            )
            return result, retries
        except TimeoutError:
            retries += 1
            if retries > MAX_TIMEOUT_RETRIES:
                result = _build_result(
                    paper_name=paper_name,
                    question_payload=question_payload,
                    mode="chunk",
                    model_answer="",
                    execution_time=None,
                    ram_usage=_get_ollama_ram_mb(),
                    status="FAILED_TIMEOUT",
                    metadata={"retries": retries, "used_chunk_pipeline": True},
                )
                return result, retries
        except Exception as exc:
            result = _build_result(
                paper_name=paper_name,
                question_payload=question_payload,
                mode="chunk",
                model_answer="",
                execution_time=None,
                ram_usage=_get_ollama_ram_mb(),
                status="FAILED_CHUNK",
                metadata={"error": str(exc), "retries": retries, "used_chunk_pipeline": True},
            )
            return result, retries


def _paper_payload(mode: str, paper_name: str, paper_text: str) -> dict[str, Any]:
    return {
        "paper_id": paper_name,
        "model": "granite",
        "mode": mode,
        "paper_length_chars": len(paper_text),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": [],
        "summary": {},
    }


def _completed_question_ids(payload: dict[str, Any]) -> set[int]:
    return {int(item["question_id"]) for item in payload.get("results", [])}


def _save_paper_payload(path: Path, payload: dict[str, Any]) -> tuple[bool, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return _safe_json_check(path)


def _summarize_mode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    scored = [item["answer_f1"] for item in results if item["status"] == "SUCCESS" and item.get("answer_f1") is not None]
    times = [item["execution_time"] for item in results if item.get("execution_time") is not None]
    ram_values = [item["ram_usage"] for item in results if item.get("ram_usage") is not None]
    return {
        "questions_total": len(results),
        "questions_scored": len(scored),
        "average_answer_f1": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "total_time": round(sum(times), 2) if times else 0.0,
        "peak_ram": round(max(ram_values), 2) if ram_values else 0.0,
        "failures": len([item for item in results if item["status"] != "SUCCESS"]),
        "skipped": len([item for item in results if item["status"] == "SKIPPED_NO_GOLD"]),
    }


def run_validation() -> dict[str, Any]:
    selected_papers = _list_selected_papers(limit=5)
    run_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    aggregate_stats = {
        "papers": len(selected_papers),
        "questions": 0,
        "failures": 0,
        "retries": 0,
        "skipped": 0,
        "direct_f1_values": [],
        "chunk_f1_values": [],
        "direct_times": [],
        "chunk_times": [],
        "known_issues": set(),
    }
    summary_rows: list[dict[str, Any]] = []

    try:
        for paper_name in selected_papers:
            paper = load_qasper_paper(paper_name)
            paper_text = paper["paper_text"]
            metadata = paper["metadata"]
            question_count = len(metadata.get("questions", []))
            aggregate_stats["questions"] += question_count

            direct_path, existing_direct = _resolve_paper_output(EVALUATIONS_DIR / "direct", paper_name, question_count)
            chunk_path, existing_chunk = _resolve_paper_output(EVALUATIONS_DIR / "chunk", paper_name, question_count)
            direct_payload = existing_direct or _paper_payload("direct", paper_name, paper_text)
            chunk_payload = existing_chunk or _paper_payload("chunk", paper_name, paper_text)
            completed_direct = _completed_question_ids(direct_payload)
            completed_chunk = _completed_question_ids(chunk_payload)

            for question_id in range(1, question_count + 1):
                question_payload = _read_question_payload(paper_name, question_id)
                direct_done = question_id in completed_direct
                chunk_done = question_id in completed_chunk

                if direct_done and chunk_done:
                    continue

                if not question_payload["gold_answer"].strip():
                    skipped = _build_result(
                        paper_name=paper_name,
                        question_payload=question_payload,
                        mode="direct",
                        model_answer="",
                        execution_time=None,
                        ram_usage=None,
                        status="SKIPPED_NO_GOLD",
                        metadata={"used_chunk_pipeline": False},
                    ).to_dict()
                    skipped["answer_f1"] = None
                    if not direct_done:
                        direct_payload["results"].append(skipped)
                        completed_direct.add(question_id)

                    skipped_chunk = _build_result(
                        paper_name=paper_name,
                        question_payload=question_payload,
                        mode="chunk",
                        model_answer="",
                        execution_time=None,
                        ram_usage=None,
                        status="SKIPPED_NO_GOLD",
                        metadata={"used_chunk_pipeline": True},
                    ).to_dict()
                    skipped_chunk["answer_f1"] = None
                    if not chunk_done:
                        chunk_payload["results"].append(skipped_chunk)
                        completed_chunk.add(question_id)
                    aggregate_stats["skipped"] += int(not direct_done) + int(not chunk_done)
                    direct_payload["summary"] = _summarize_mode_payload(direct_payload)
                    chunk_payload["summary"] = _summarize_mode_payload(chunk_payload)
                    _save_paper_payload(direct_path, direct_payload)
                    _save_paper_payload(chunk_path, chunk_payload)
                    continue

                if not direct_done:
                    direct_result, direct_retries = _run_direct_question(paper_name, question_payload, paper_text)
                    aggregate_stats["retries"] += direct_retries
                    direct_dict = direct_result.to_dict()
                    direct_dict["answer_f1"] = (
                        round(compute_answer_f1(direct_result.model_answer, direct_result.gold_answer), 4)
                        if direct_result.status == "SUCCESS"
                        else None
                    )
                    direct_payload["results"].append(direct_dict)
                    completed_direct.add(question_id)
                    direct_payload["summary"] = _summarize_mode_payload(direct_payload)
                    valid_json, json_error = _save_paper_payload(direct_path, direct_payload)
                    direct_dict["metadata"] = {**(direct_dict.get("metadata") or {}), "json_saved": valid_json}
                    if not valid_json:
                        direct_dict["status"] = "FAILED_JSON"
                        direct_dict["metadata"]["json_error"] = json_error

                if not chunk_done:
                    chunk_result, chunk_retries = _run_chunk_question(
                        paper_name,
                        question_payload,
                        paper_text,
                        run_tag=run_tag,
                    )
                    aggregate_stats["retries"] += chunk_retries
                    chunk_dict = chunk_result.to_dict()
                    chunk_dict["answer_f1"] = (
                        round(compute_answer_f1(chunk_result.model_answer, chunk_result.gold_answer), 4)
                        if chunk_result.status == "SUCCESS"
                        else None
                    )
                    chunk_payload["results"].append(chunk_dict)
                    completed_chunk.add(question_id)
                    chunk_payload["summary"] = _summarize_mode_payload(chunk_payload)
                    valid_json, json_error = _save_paper_payload(chunk_path, chunk_payload)
                    chunk_dict["metadata"] = {**(chunk_dict.get("metadata") or {}), "json_saved": valid_json}
                    if not valid_json:
                        chunk_dict["status"] = "FAILED_JSON"
                        chunk_dict["metadata"]["json_error"] = json_error

            direct_payload["summary"] = _summarize_mode_payload(direct_payload)
            chunk_payload["summary"] = _summarize_mode_payload(chunk_payload)
            _save_paper_payload(direct_path, direct_payload)
            _save_paper_payload(chunk_path, chunk_payload)

            direct_statuses = [item["status"] for item in direct_payload["results"]]
            chunk_statuses = [item["status"] for item in chunk_payload["results"]]
            aggregate_stats["failures"] += len([status for status in direct_statuses + chunk_statuses if status not in ("SUCCESS", "SKIPPED_NO_GOLD")])
            aggregate_stats["direct_f1_values"].extend([item["answer_f1"] for item in direct_payload["results"] if item["answer_f1"] is not None])
            aggregate_stats["chunk_f1_values"].extend([item["answer_f1"] for item in chunk_payload["results"] if item["answer_f1"] is not None])
            aggregate_stats["direct_times"].append(direct_payload["summary"]["total_time"])
            aggregate_stats["chunk_times"].append(chunk_payload["summary"]["total_time"])

            for item in direct_payload["results"] + chunk_payload["results"]:
                if item["status"] not in ("SUCCESS", "SKIPPED_NO_GOLD"):
                    aggregate_stats["known_issues"].add(item["status"])

            summary_rows.append(
                {
                    "Paper ID": paper_name,
                    "Questions": question_count,
                    "Characters": len(paper_text),
                    "Direct F1": direct_payload["summary"]["average_answer_f1"],
                    "Chunk F1": chunk_payload["summary"]["average_answer_f1"],
                    "Direct Time": direct_payload["summary"]["total_time"],
                    "Chunk Time": chunk_payload["summary"]["total_time"],
                    "RAM": max(direct_payload["summary"]["peak_ram"], chunk_payload["summary"]["peak_ram"]),
                    "Model": "granite",
                }
            )
    except KeyboardInterrupt:
        aggregate_stats["known_issues"].add("INTERRUPTED_PARTIAL_PROGRESS_SAVED")

    csv_path = _versioned_summary_csv_path()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Paper ID", "Questions", "Characters", "Direct F1", "Chunk F1", "Direct Time", "Chunk Time", "RAM", "Model"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "selected_papers": selected_papers,
        "summary_csv": str(csv_path),
        "papers_completed": len(summary_rows),
        "direct_json_count": len(list((EVALUATIONS_DIR / "direct").glob("paper_*.json"))),
        "chunk_json_count": len(list((EVALUATIONS_DIR / "chunk").glob("paper_*.json"))),
        "average_direct_f1": round(sum(aggregate_stats["direct_f1_values"]) / len(aggregate_stats["direct_f1_values"]), 4) if aggregate_stats["direct_f1_values"] else 0.0,
        "average_chunk_f1": round(sum(aggregate_stats["chunk_f1_values"]) / len(aggregate_stats["chunk_f1_values"]), 4) if aggregate_stats["chunk_f1_values"] else 0.0,
        "average_direct_time": round(sum(aggregate_stats["direct_times"]) / len(aggregate_stats["direct_times"]), 2) if aggregate_stats["direct_times"] else 0.0,
        "average_chunk_time": round(sum(aggregate_stats["chunk_times"]) / len(aggregate_stats["chunk_times"]), 2) if aggregate_stats["chunk_times"] else 0.0,
        "failures": aggregate_stats["failures"],
        "retries": aggregate_stats["retries"],
        "skipped_questions": aggregate_stats["skipped"],
        "known_issues": sorted(aggregate_stats["known_issues"]),
    }
    _write_validation_report(report)
    return report


def _write_validation_report(report: dict[str, Any]) -> None:
    question_total = 0
    for paper_name in report["selected_papers"]:
        direct_matches = sorted((EVALUATIONS_DIR / "direct").glob(f"{paper_name}*.json"))
        if direct_matches:
            payload = json.loads(direct_matches[-1].read_text(encoding="utf-8"))
            question_total += len(payload.get("results", []))

    lines = [
        "# Granite Validation Report",
        "",
        f"- Number of papers: {report['papers_completed']}",
        f"- Number of questions: {question_total}",
        f"- Failures: {report['failures']}",
        f"- Retries: {report['retries']}",
        f"- Skipped questions: {report['skipped_questions']}",
        f"- Average Direct F1: {report['average_direct_f1']}",
        f"- Average Chunk F1: {report['average_chunk_f1']}",
        f"- Average Direct Time: {report['average_direct_time']} sec",
        f"- Average Chunk Time: {report['average_chunk_time']} sec",
        "",
        "## Known issues",
    ]

    if report["known_issues"]:
        lines.extend([f"- {issue}" for issue in report["known_issues"]])
    else:
        lines.append("- None observed during the 5-paper Granite validation run.")

    lines.extend(
        [
            "",
            "## Recommendations before scaling",
            "- Review all failed or skipped questions before expanding to more papers.",
            "- Keep direct and chunk outputs versioned so reruns do not overwrite validation artifacts.",
            "- Scale Granite beyond 5 papers only after confirming the F1 distribution looks reasonable and no placeholder or copied-gold failures appear.",
            "- Delay Qwen and DeepSeek until this Granite validation report is approved.",
        ]
    )

    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report = run_validation()
    print(json.dumps(report, indent=2))
