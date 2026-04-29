"""Data models for bias-audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BiasType(Enum):
    LEADING_QUESTION = "leading_question"
    LOADED_QUESTION = "loaded_question"
    DOUBLE_BARRELED = "double_barreled"
    CLOSED_QUESTION = "closed_question"
    CONFIRMATION_REINFORCEMENT = "confirmation_reinforcement"
    AGREEMENT_INJECTION = "agreement_injection"
    HYPOTHETICAL_GROUNDING = "hypothetical_grounding"
    ANSWER_SUGGESTION = "answer_suggestion"
    SOCIAL_DESIRABILITY = "social_desirability"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]


@dataclass
class TranscriptTurn:
    turn_index: int
    speaker: str
    text: str


@dataclass
class BiasInstance:
    turn_index: int
    speaker: str
    text: str
    bias_type: BiasType
    severity: Severity
    excerpt: str
    explanation: str
    llm_rationale: Optional[str] = None
    suggested_alternative: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "text": self.text,
            "bias_type": self.bias_type.value,
            "severity": self.severity.value,
            "excerpt": self.excerpt,
            "explanation": self.explanation,
            "llm_rationale": self.llm_rationale,
            "suggested_alternative": self.suggested_alternative,
        }


@dataclass
class TranscriptReport:
    session_id: str
    total_interviewer_turns: int
    biased_turns: int
    bias_rate: float
    instances: list[BiasInstance] = field(default_factory=list)
    bias_type_counts: dict = field(default_factory=dict)
    severity_counts: dict = field(default_factory=dict)
    most_common_bias: Optional[str] = None
    narrative_summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_interviewer_turns": self.total_interviewer_turns,
            "biased_turns": self.biased_turns,
            "bias_rate": round(self.bias_rate, 4),
            "bias_type_counts": self.bias_type_counts,
            "severity_counts": self.severity_counts,
            "most_common_bias": self.most_common_bias,
            "narrative_summary": self.narrative_summary,
            "instances": [i.to_dict() for i in self.instances],
        }


@dataclass
class CorpusReport:
    total_sessions: int
    total_interviewer_turns: int
    total_biased_turns: int
    overall_bias_rate: float
    per_session: list[TranscriptReport] = field(default_factory=list)
    corpus_bias_type_counts: dict = field(default_factory=dict)
    most_prevalent_bias: Optional[str] = None
    worst_sessions: list[str] = field(default_factory=list)
    corpus_narrative: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_sessions": self.total_sessions,
            "total_interviewer_turns": self.total_interviewer_turns,
            "total_biased_turns": self.total_biased_turns,
            "overall_bias_rate": round(self.overall_bias_rate, 4),
            "corpus_bias_type_counts": self.corpus_bias_type_counts,
            "most_prevalent_bias": self.most_prevalent_bias,
            "worst_sessions": self.worst_sessions,
            "corpus_narrative": self.corpus_narrative,
            "per_session": [r.to_dict() for r in self.per_session],
        }
