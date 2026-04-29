"""basic_usage.py — programmatic API example for bias-audit."""

from pathlib import Path
from bias_audit import (
    parse_transcript,
    detect_bias_in_transcript,
    build_transcript_report,
    build_corpus_report,
)
from bias_audit.models import Severity

# ---------------------------------------------------------------------------
# Single transcript analysis
# ---------------------------------------------------------------------------

turns = parse_transcript(Path("tests/fixtures/sample_transcript.json"))

instances, total_turns = detect_bias_in_transcript(
    turns,
    interviewer_label="Interviewer",
    min_severity=Severity.MEDIUM,
)

report = build_transcript_report(
    session_id="sample",
    instances=instances,
    total_interviewer_turns=total_turns,
)

print(f"Session     : {report.session_id}")
print(f"Bias rate   : {report.bias_rate:.1%}")
print(f"Most common : {report.most_common_bias}")
print(f"Instances   : {len(report.instances)}")

for inst in report.instances:
    print(f"  [{inst.severity.value.upper()}] {inst.bias_type.value}")
    print(f"    \"{inst.excerpt}\"")
    print(f"    → {inst.explanation}")

# ---------------------------------------------------------------------------
# Corpus-level analysis
# ---------------------------------------------------------------------------

corpus_dir = Path("tests/fixtures/corpus")
session_reports = []

for f in sorted(corpus_dir.glob("*.json")):
    t = parse_transcript(f)
    insts, total = detect_bias_in_transcript(t, "Interviewer")
    r = build_transcript_report(f.stem, insts, total)
    session_reports.append(r)

corpus = build_corpus_report(session_reports)

print(f"\nCorpus summary")
print(f"  Sessions       : {corpus.total_sessions}")
print(f"  Overall rate   : {corpus.overall_bias_rate:.1%}")
print(f"  Top bias type  : {corpus.most_prevalent_bias}")
print(f"  Worst sessions : {corpus.worst_sessions}")
