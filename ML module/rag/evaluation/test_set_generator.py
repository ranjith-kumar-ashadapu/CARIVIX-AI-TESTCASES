"""Generate test cases for AI and RAG evaluation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvaluationCase:
    """One evaluation case used to assess AI retrieval or grounding."""

    query: str
    expected_answer: Optional[str] = None
    document_type: str = "text"
    category: str = "context-based"
    expected_source: Optional[str] = None
    context: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TestSetGenerator:
    """Generate a small JSON/CSV test set for AI and RAG evaluation."""

    def __init__(self, cases: Optional[Iterable[EvaluationCase]] = None) -> None:
        self.cases = list(cases or [])

    @staticmethod
    def default_cases() -> List[EvaluationCase]:
        return [
            EvaluationCase(
                query="What is CARIVIX AI?",
                expected_answer="CARIVIX AI is a comprehensive AI platform for economic analysis.",
                document_type="text",
                category="direct-factual",
                expected_source="sample_about_carivix.txt",
                context="CARIVIX AI is a comprehensive AI platform for economic analysis.",
                notes="Direct factual question.",
            ),
            EvaluationCase(
                query="Which machine learning algorithms are supported?",
                expected_answer="The platform supports Random Forest and XGBoost.",
                document_type="text",
                category="context-based",
                expected_source="sample_about_carivix.txt",
                context="The platform supports multiple ML algorithms including Random Forest and XGBoost.",
                notes="Context-based answer.",
            ),
            EvaluationCase(
                query="How does CARIVIX handle model experiment tracking?",
                expected_answer="Experiment tracking is done with MLflow.",
                document_type="text",
                category="multi-document",
                expected_source="sample_about_carivix.txt",
                context="Experiment tracking is done with MLflow.",
                notes="Question can be answered from multiple document fragments.",
            ),
            EvaluationCase(
                query="What is the projected revenue growth next quarter?",
                expected_answer=None,
                document_type="report",
                category="insufficient-context",
                expected_source=None,
                context=None,
                notes="Should decline when the context lacks an answer.",
            ),
            EvaluationCase(
                query="Which source supports the statement about experiment tracking?",
                expected_answer="MLflow.",
                document_type="text",
                category="source-grounding",
                expected_source="sample_about_carivix.txt",
                context="Experiment tracking is done with MLflow.",
                notes="Ground the answer to the correct source.",
            ),
        ]

    def add_case(self, case: EvaluationCase) -> None:
        self.cases.append(case)

    def generate(self) -> List[Dict[str, Any]]:
        return [case.to_dict() for case in self.cases or self.default_cases()]

    def save_json(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.generate(), handle, indent=2)
        return output_path

    def save_csv(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        rows = self.generate()
        fieldnames = [
            "query",
            "expected_answer",
            "document_type",
            "category",
            "expected_source",
            "context",
            "notes",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return output_path
