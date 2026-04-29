"""bias-audit — detect and categorise interviewer bias patterns across transcript corpora."""

from bias_audit.aggregator import build_corpus_report, build_transcript_report
from bias_audit.detector import detect_bias_in_transcript, detect_bias_in_turn
from bias_audit.models import (
    BiasInstance,
    BiasType,
    CorpusReport,
    Severity,
    TranscriptReport,
    TranscriptTurn,
)
from bias_audit.parser import parse_transcript

__version__ = "0.1.0"

__all__ = [
    "BiasType",
    "Severity",
    "TranscriptTurn",
    "BiasInstance",
    "TranscriptReport",
    "CorpusReport",
    "parse_transcript",
    "detect_bias_in_turn",
    "detect_bias_in_transcript",
    "build_transcript_report",
    "build_corpus_report",
]
