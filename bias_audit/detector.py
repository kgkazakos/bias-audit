"""Rule-based interviewer bias detection engine."""

from __future__ import annotations

import re
from typing import Optional

from bias_audit.models import BiasInstance, BiasType, Severity, TranscriptTurn

# ---------------------------------------------------------------------------
# Pattern registry
# Each entry: list of regex patterns, default severity, human explanation
# ---------------------------------------------------------------------------

_REGISTRY: dict[BiasType, dict] = {
    BiasType.LEADING_QUESTION: {
        "patterns": [
            r"\bdon'?t you (think|feel|agree|find|believe|see)\b",
            r"\bwouldn'?t you (say|agree|think|consider)\b",
            r"\bisn'?t it (true|right|fair|correct|obvious)\b",
            r"\byou would agree\b",
            r"\byou'?d agree\b",
            r"\bsurely you\b",
            r"\bobviously\b.{0,60}\?",
            r"\bclearly\b.{0,60}\?",
            r"\baren'?t you (finding|seeing|feeling)\b",
        ],
        "severity": Severity.HIGH,
        "explanation": (
            "Leading question that presupposes or steers the participant toward "
            "a particular answer, compromising response validity."
        ),
    },
    BiasType.LOADED_QUESTION: {
        "patterns": [
            r"\bwhy did you (struggle|fail|have trouble|have difficulty|have a hard time)\b",
            r"\bwhat made it (so )?(hard|difficult|frustrating|confusing|painful)\b",
            r"\bwhen did you (stop|give up|quit|abandon)\b",
            r"\bwhy (couldn'?t|can'?t) you\b",
            r"\bwhat went wrong\b",
            r"\bwhat (problems|issues|difficulties) did you have\b",
            r"\bwhy (was|is|were|are) (it|that|this) (so )?(hard|difficult|frustrating)\b",
        ],
        "severity": Severity.HIGH,
        "explanation": (
            "Loaded question containing an embedded negative assumption about the "
            "participant's experience that the participant has not volunteered."
        ),
    },
    BiasType.DOUBLE_BARRELED: {
        "patterns": [
            r"\b(how|what|why|when|where|who)\b.{10,}\band\b\s+(how|what|why|when|where|who)\b",
            r"\band also\b.{3,}\?",
            r"\bas well as\b.{3,}\?",
            r"\band (at the same time|simultaneously)\b.{3,}\?",
        ],
        "severity": Severity.MEDIUM,
        "explanation": (
            "Double-barreled question that asks about two distinct things at once, "
            "making it impossible to know which part the participant is answering."
        ),
    },
    BiasType.CLOSED_QUESTION: {
        "patterns": [
            r"^(is|are|was|were|do|does|did|can|could|will|would|have|has|had|should)\s",
        ],
        "severity": Severity.LOW,
        "explanation": (
            "Closed yes/no question that constrains participant response depth; "
            "open-ended probes typically yield richer qualitative data."
        ),
    },
    BiasType.CONFIRMATION_REINFORCEMENT: {
        "patterns": [
            r"\bso (basically|essentially|you'?re saying|what you'?re saying is)\b",
            r"\bthat'?s (what I thought|what I expected|what I assumed|what we expected)\b",
            r"\bjust (as|like) (I|we) (expected|thought|assumed|suspected)\b",
            r"\bso (you agree|you think|you believe|you feel) that\b",
            r"[,\s](right|correct|yeah)\?$",
            r"\bconfirm(s|ing)?\b.{0,40}\b(thought|hypothesis|assumption|hunch)\b",
        ],
        "severity": Severity.MEDIUM,
        "explanation": (
            "Confirmation-seeking language that surfaces the interviewer's prior "
            "hypothesis and may pressure participants to validate it."
        ),
    },
    BiasType.AGREEMENT_INJECTION: {
        "patterns": [
            r"^(exactly|right|of course|absolutely|definitely|totally|perfect|great|fantastic|wonderful)[\.!\,]?\s*$",
            r"^(yes yes|yeah yeah|mm-?hmm\.?\s*$|uh-?huh\.?\s*$)",
            r"^(exactly|of course|absolutely|definitely|right)[\.!,]\s+(so|and|now|tell)",
        ],
        "severity": Severity.LOW,
        "explanation": (
            "Enthusiastic agreement injected mid-session signals the interviewer's "
            "approval of specific responses, priming participants toward similar answers."
        ),
    },
    BiasType.HYPOTHETICAL_GROUNDING: {
        "patterns": [
            r"\bif you (were|had|could|would)\b",
            r"\bimagine (if|you|that)\b",
            r"\bhow would you\b",
            r"\bwould you (say|describe|call|consider|characterise|characterize)\b",
            r"\bcould you imagine\b",
            r"\blet'?s say (that |you |hypothetically )\b",
            r"\bsuppose (you|that|for a moment)\b",
        ],
        "severity": Severity.MEDIUM,
        "explanation": (
            "Hypothetical framing that asks participants to speculate about imagined "
            "behaviour rather than report actual past experience — a core validity risk."
        ),
    },
    BiasType.ANSWER_SUGGESTION: {
        "patterns": [
            r"\bwas it (because|due to|related to|caused by)\b",
            r"\bdid you feel (frustrated|confused|lost|overwhelmed|annoyed|happy|satisfied|relieved)\b",
            r"\bwas (that|it|this) (frustrating|confusing|difficult|easy|helpful|annoying|satisfying)\b",
            r"\b(frustrated|confused|overwhelmed|annoyed)\?.*\bor\b",
            r"\bso it (felt|seemed|sounded) like\b",
            r"\bwas the (problem|issue|difficulty|challenge)\b",
        ],
        "severity": Severity.HIGH,
        "explanation": (
            "Suggests specific emotions, causes, or framings before the participant "
            "has volunteered them, anchoring responses and distorting data."
        ),
    },
    BiasType.SOCIAL_DESIRABILITY: {
        "patterns": [
            r"\bmost (people|users|participants|folks|individuals)\b",
            r"\ba lot of (people|users|participants)\b",
            r"\beveryone (else|we'?ve spoken to|I'?ve spoken to|we talk(ed)? to)\b",
            r"\bstudies (show|suggest|indicate|have found)\b",
            r"\bresearch (shows|suggests|indicates|has shown|has found)\b",
            r"\bthe (norm|standard|typical|average|usual)\b.{0,20}(is|are)\b",
            r"\bpeople (usually|typically|generally|normally)\b",
        ],
        "severity": Severity.HIGH,
        "explanation": (
            "Social desirability framing invokes norms or other people's behaviour "
            "before asking the participant, creating pressure to conform."
        ),
    },
}

# Human-readable labels
BIAS_LABELS: dict[BiasType, str] = {
    BiasType.LEADING_QUESTION: "Leading Question",
    BiasType.LOADED_QUESTION: "Loaded Question",
    BiasType.DOUBLE_BARRELED: "Double-Barreled Question",
    BiasType.CLOSED_QUESTION: "Closed Question",
    BiasType.CONFIRMATION_REINFORCEMENT: "Confirmation Reinforcement",
    BiasType.AGREEMENT_INJECTION: "Agreement Injection",
    BiasType.HYPOTHETICAL_GROUNDING: "Hypothetical Grounding",
    BiasType.ANSWER_SUGGESTION: "Answer Suggestion",
    BiasType.SOCIAL_DESIRABILITY: "Social Desirability Framing",
}


def _extract_excerpt(text: str, match: re.Match, context: int = 30) -> str:
    start = max(0, match.start() - context)
    end = min(len(text), match.end() + context)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt = excerpt + "…"
    return excerpt


def detect_bias_in_turn(
    turn: TranscriptTurn,
    min_severity: Optional[Severity] = None,
) -> list[BiasInstance]:
    """Detect all bias types present in a single interviewer turn."""
    instances: list[BiasInstance] = []
    text_lower = turn.text.lower().strip()

    for bias_type, config in _REGISTRY.items():
        severity: Severity = config["severity"]

        if min_severity is not None and severity.rank < min_severity.rank:
            continue

        for pattern in config["patterns"]:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                excerpt = _extract_excerpt(turn.text, match)
                instances.append(
                    BiasInstance(
                        turn_index=turn.turn_index,
                        speaker=turn.speaker,
                        text=turn.text,
                        bias_type=bias_type,
                        severity=severity,
                        excerpt=excerpt,
                        explanation=config["explanation"],
                    )
                )
                break  # one instance per bias type per turn

    return instances


def detect_bias_in_transcript(
    turns: list[TranscriptTurn],
    interviewer_label: str = "Interviewer",
    min_severity: Optional[Severity] = None,
) -> tuple[list[BiasInstance], int]:
    """
    Detect bias across all interviewer turns in a transcript.

    Returns (instances, total_interviewer_turns).
    """
    all_instances: list[BiasInstance] = []
    interviewer_turns = 0

    for turn in turns:
        if turn.speaker.strip().lower() == interviewer_label.strip().lower():
            interviewer_turns += 1
            instances = detect_bias_in_turn(turn, min_severity=min_severity)
            all_instances.extend(instances)

    return all_instances, interviewer_turns
