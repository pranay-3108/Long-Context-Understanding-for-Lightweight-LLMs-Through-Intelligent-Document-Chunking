from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    paper_id: str
    question_id: int
    question: str
    gold_answer: str
    model_answer: str
    evidence: List[str]
    paper_length: int
    execution_time: Optional[float] = None
    ram_usage: Optional[float] = None
    mode: str = "direct"
    model: str = "unknown"
    status: str = "SUCCESS"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        payload["paper_length_chars"] = self.paper_length
        payload["ram_mb"] = self.ram_usage
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            paper_id=payload["paper_id"],
            question_id=payload["question_id"],
            question=payload["question"],
            gold_answer=payload["gold_answer"],
            model_answer=payload["model_answer"],
            evidence=list(payload.get("evidence", [])),
            paper_length=payload.get("paper_length", payload.get("paper_length_chars", 0)),
            execution_time=payload.get("execution_time"),
            ram_usage=payload.get("ram_usage", payload.get("ram_mb")),
            mode=payload.get("mode", "direct"),
            model=payload.get("model", "unknown"),
            status=payload.get("status", "SUCCESS"),
            metadata=payload.get("metadata"),
        )
