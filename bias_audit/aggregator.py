"""Aggregation — transcript and corpus reports from raw bias instances."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from bias_audit.models import BiasInstance, CorpusReport, TranscriptReport


def build_transcript_report(
    session_id: str,
    instances: list[BiasInstance],
    total_interviewer_turns: int,
) -> TranscriptReport:
    """Build a per-session summary from a list of bias instances."""
    biased_turn_indices = {i.turn_index for i in instances}
    biased_turns = len(biased_turn_indices)
    bias_rate = biased_turns / total_interviewer_turns if total_interviewer_turns > 0 else 0.0

    bias_type_counts = dict(Counter(i.bias_type.value for i in instances))
    severity_counts = dict(Counter(i.severity.value for i in instances))

    type_counter = Counter(i.bias_type.value for i in instances)
    most_common_bias = type_counter.most_common(1)[0][0] if type_counter else None

    return TranscriptReport(
        session_id=session_id,
        total_interviewer_turns=total_interviewer_turns,
        biased_turns=biased_turns,
        bias_rate=bias_rate,
        instances=instances,
        bias_type_counts=bias_type_counts,
        severity_counts=severity_counts,
        most_common_bias=most_common_bias,
    )


def build_corpus_report(session_reports: list[TranscriptReport]) -> CorpusReport:
    """Aggregate per-session reports into a corpus-level summary."""
    total_sessions = len(session_reports)
    total_interviewer_turns = sum(r.total_interviewer_turns for r in session_reports)
    total_biased_turns = sum(r.biased_turns for r in session_reports)
    overall_bias_rate = (
        total_biased_turns / total_interviewer_turns if total_interviewer_turns > 0 else 0.0
    )

    all_instances = [inst for r in session_reports for inst in r.instances]
    corpus_type_counts = dict(Counter(i.bias_type.value for i in all_instances))
    type_counter = Counter(i.bias_type.value for i in all_instances)
    most_prevalent_bias = type_counter.most_common(1)[0][0] if type_counter else None

    sorted_by_rate = sorted(session_reports, key=lambda r: r.bias_rate, reverse=True)
    worst_sessions = [r.session_id for r in sorted_by_rate[:3] if r.bias_rate > 0]

    return CorpusReport(
        total_sessions=total_sessions,
        total_interviewer_turns=total_interviewer_turns,
        total_biased_turns=total_biased_turns,
        overall_bias_rate=overall_bias_rate,
        per_session=session_reports,
        corpus_bias_type_counts=corpus_type_counts,
        most_prevalent_bias=most_prevalent_bias,
        worst_sessions=worst_sessions,
    )


def session_id_from_path(path: Path) -> str:
    """Derive a clean session ID from a file path."""
    return path.stem
