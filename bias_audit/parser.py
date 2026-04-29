"""Transcript parsing — JSON and plain text formats."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bias_audit.models import TranscriptTurn

SUPPORTED_EXTENSIONS = {".json", ".txt", ".md"}

# Common interviewer speaker labels (checked case-insensitively)
_INTERVIEWER_LABELS = {
    "interviewer", "i", "moderator", "facilitator", "researcher",
    "host", "interviewer 1", "interviewer1",
}


def parse_json_transcript(path: Path) -> list[TranscriptTurn]:
    """
    Parse a JSON transcript.

    Accepted shapes:
      - List of {"speaker": "...", "text": "..."} objects
      - List of {"role": "...", "content": "..."} objects
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"JSON transcript must be a list, got {type(raw).__name__}")

    turns: list[TranscriptTurn] = []
    for idx, item in enumerate(raw):
        speaker = item.get("speaker") or item.get("role") or "Unknown"
        text = item.get("text") or item.get("content") or ""
        if text.strip():
            turns.append(TranscriptTurn(turn_index=idx, speaker=str(speaker), text=str(text)))
    return turns


def parse_text_transcript(path: Path) -> list[TranscriptTurn]:
    """
    Parse a plain-text transcript with 'Speaker: text' lines.

    Handles multi-line turns — a new speaker line starts a new turn.
    """
    content = path.read_text(encoding="utf-8")
    speaker_line = re.compile(r"^([A-Za-z][A-Za-z\s\-]*?):\s+(.+)$")

    turns: list[TranscriptTurn] = []
    current_speaker: str | None = None
    current_parts: list[str] = []
    turn_index = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = speaker_line.match(line)
        if m:
            # Save accumulated turn
            if current_speaker and current_parts:
                turns.append(
                    TranscriptTurn(
                        turn_index=turn_index,
                        speaker=current_speaker,
                        text=" ".join(current_parts).strip(),
                    )
                )
                turn_index += 1
            current_speaker = m.group(1).strip()
            current_parts = [m.group(2).strip()]
        elif current_speaker:
            # Multi-line continuation
            current_parts.append(line)

    # Flush final turn
    if current_speaker and current_parts:
        turns.append(
            TranscriptTurn(
                turn_index=turn_index,
                speaker=current_speaker,
                text=" ".join(current_parts).strip(),
            )
        )

    return turns


def parse_transcript(path: Path) -> list[TranscriptTurn]:
    """Auto-detect format and parse a transcript file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if path.suffix == ".json":
        return parse_json_transcript(path)
    return parse_text_transcript(path)


def infer_interviewer_label(turns: list[TranscriptTurn]) -> str:
    """
    Infer which speaker label belongs to the interviewer.

    Priority:
    1. Exact match against known interviewer labels
    2. Speaker with the minority of turns (interviewers usually speak less)
    """
    speakers = [t.speaker for t in turns]
    for label in turns:
        if label.speaker.strip().lower() in _INTERVIEWER_LABELS:
            return label.speaker.strip()

    counts = Counter(speakers)
    if not counts:
        return "Interviewer"
    # Return least-frequent speaker as a heuristic
    return counts.most_common()[-1][0]
