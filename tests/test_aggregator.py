"""Tests for bias_audit.aggregator."""


from pathlib import Path

from bias_audit.aggregator import (
    build_corpus_report,
    build_transcript_report,
    session_id_from_path,
)
from bias_audit.models import BiasInstance, BiasType, Severity


def _make_instance(turn_index: int = 0, bias_type: BiasType = BiasType.LEADING_QUESTION,
                   severity: Severity = Severity.HIGH) -> BiasInstance:
    return BiasInstance(
        turn_index=turn_index,
        speaker="Interviewer",
        text="Don't you think this is great?",
        bias_type=bias_type,
        severity=severity,
        excerpt="Don't you think this is great?",
        explanation="Leading question.",
    )


class TestBuildTranscriptReport:
    def test_empty_instances(self):
        report = build_transcript_report("s1", [], total_interviewer_turns=10)
        assert report.biased_turns == 0
        assert report.bias_rate == 0.0
        assert report.most_common_bias is None

    def test_bias_rate_calculation(self):
        instances = [_make_instance(0), _make_instance(1)]
        report = build_transcript_report("s1", instances, total_interviewer_turns=10)
        assert report.bias_rate == 0.2

    def test_biased_turns_counts_unique_indices(self):
        # Two instances on the same turn should count as 1 biased turn
        instances = [
            _make_instance(0, BiasType.LEADING_QUESTION),
            _make_instance(0, BiasType.LOADED_QUESTION),
        ]
        report = build_transcript_report("s1", instances, total_interviewer_turns=5)
        assert report.biased_turns == 1

    def test_bias_type_counts(self):
        instances = [
            _make_instance(0, BiasType.LEADING_QUESTION),
            _make_instance(1, BiasType.LEADING_QUESTION),
            _make_instance(2, BiasType.CLOSED_QUESTION),
        ]
        report = build_transcript_report("s1", instances, total_interviewer_turns=10)
        assert report.bias_type_counts["leading_question"] == 2
        assert report.bias_type_counts["closed_question"] == 1

    def test_severity_counts(self):
        instances = [
            _make_instance(0, severity=Severity.HIGH),
            _make_instance(1, severity=Severity.HIGH),
            _make_instance(2, severity=Severity.LOW),
        ]
        report = build_transcript_report("s1", instances, total_interviewer_turns=10)
        assert report.severity_counts["high"] == 2
        assert report.severity_counts["low"] == 1

    def test_most_common_bias(self):
        instances = [
            _make_instance(0, BiasType.LEADING_QUESTION),
            _make_instance(1, BiasType.LEADING_QUESTION),
            _make_instance(2, BiasType.CLOSED_QUESTION),
        ]
        report = build_transcript_report("s1", instances, total_interviewer_turns=10)
        assert report.most_common_bias == "leading_question"

    def test_zero_interviewer_turns_rate(self):
        report = build_transcript_report("s1", [], total_interviewer_turns=0)
        assert report.bias_rate == 0.0

    def test_session_id_preserved(self):
        report = build_transcript_report("my_session", [], total_interviewer_turns=5)
        assert report.session_id == "my_session"

    def test_instances_stored(self):
        instances = [_make_instance(0), _make_instance(1)]
        report = build_transcript_report("s1", instances, total_interviewer_turns=5)
        assert len(report.instances) == 2


class TestBuildCorpusReport:
    def _make_report(self, session_id: str, biased: int, total: int) -> object:
        instances = [_make_instance(i) for i in range(biased)]
        return build_transcript_report(session_id, instances, total_interviewer_turns=total)

    def test_total_sessions(self):
        reports = [self._make_report(f"s{i}", 1, 5) for i in range(3)]
        corpus = build_corpus_report(reports)
        assert corpus.total_sessions == 3

    def test_total_interviewer_turns(self):
        reports = [self._make_report(f"s{i}", 1, 5) for i in range(3)]
        corpus = build_corpus_report(reports)
        assert corpus.total_interviewer_turns == 15

    def test_total_biased_turns(self):
        reports = [self._make_report(f"s{i}", 2, 10) for i in range(3)]
        corpus = build_corpus_report(reports)
        assert corpus.total_biased_turns == 6

    def test_overall_bias_rate(self):
        reports = [self._make_report(f"s{i}", 2, 10) for i in range(3)]
        corpus = build_corpus_report(reports)
        assert abs(corpus.overall_bias_rate - 0.2) < 1e-9

    def test_worst_sessions_sorted(self):
        r1 = self._make_report("low_bias", 1, 10)   # 0.1
        r2 = self._make_report("high_bias", 8, 10)  # 0.8
        r3 = self._make_report("mid_bias", 3, 10)   # 0.3
        corpus = build_corpus_report([r1, r2, r3])
        assert corpus.worst_sessions[0] == "high_bias"

    def test_corpus_bias_type_counts_aggregated(self):
        instances_a = [_make_instance(0, BiasType.LEADING_QUESTION)]
        instances_b = [_make_instance(0, BiasType.LEADING_QUESTION),
                       _make_instance(1, BiasType.CLOSED_QUESTION)]
        r1 = build_transcript_report("a", instances_a, 5)
        r2 = build_transcript_report("b", instances_b, 5)
        corpus = build_corpus_report([r1, r2])
        assert corpus.corpus_bias_type_counts["leading_question"] == 2
        assert corpus.corpus_bias_type_counts["closed_question"] == 1

    def test_most_prevalent_bias(self):
        instances = [_make_instance(i, BiasType.SOCIAL_DESIRABILITY) for i in range(3)]
        report = build_transcript_report("s1", instances, 10)
        corpus = build_corpus_report([report])
        assert corpus.most_prevalent_bias == "social_desirability"

    def test_zero_sessions(self):
        corpus = build_corpus_report([])
        assert corpus.total_sessions == 0
        assert corpus.overall_bias_rate == 0.0
        assert corpus.most_prevalent_bias is None

    def test_per_session_preserved(self):
        reports = [self._make_report(f"s{i}", 1, 5) for i in range(2)]
        corpus = build_corpus_report(reports)
        assert len(corpus.per_session) == 2


class TestSessionIdFromPath:
    def test_simple_path(self):
        assert session_id_from_path(Path("session1.json")) == "session1"

    def test_nested_path(self):
        assert session_id_from_path(Path("/some/dir/interview_02.txt")) == "interview_02"

    def test_no_extension(self):
        assert session_id_from_path(Path("mysession")) == "mysession"
