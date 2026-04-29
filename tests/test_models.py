"""Tests for bias_audit.models."""


from bias_audit.models import (
    BiasInstance,
    BiasType,
    CorpusReport,
    Severity,
    TranscriptReport,
    TranscriptTurn,
)


class TestBiasType:
    def test_all_expected_values_exist(self):
        expected = {
            "leading_question", "loaded_question", "double_barreled",
            "closed_question", "confirmation_reinforcement", "agreement_injection",
            "hypothetical_grounding", "answer_suggestion", "social_desirability",
        }
        actual = {bt.value for bt in BiasType}
        assert actual == expected

    def test_enum_count(self):
        assert len(BiasType) == 9

    def test_enum_access_by_name(self):
        assert BiasType["LEADING_QUESTION"] == BiasType.LEADING_QUESTION

    def test_enum_access_by_value(self):
        assert BiasType("leading_question") == BiasType.LEADING_QUESTION


class TestSeverity:
    def test_rank_ordering(self):
        assert Severity.LOW.rank < Severity.MEDIUM.rank < Severity.HIGH.rank

    def test_rank_values(self):
        assert Severity.LOW.rank == 0
        assert Severity.MEDIUM.rank == 1
        assert Severity.HIGH.rank == 2

    def test_all_severities(self):
        assert {s.value for s in Severity} == {"low", "medium", "high"}


class TestTranscriptTurn:
    def test_basic_creation(self):
        turn = TranscriptTurn(turn_index=0, speaker="Interviewer", text="Hello?")
        assert turn.turn_index == 0
        assert turn.speaker == "Interviewer"
        assert turn.text == "Hello?"

    def test_zero_index_allowed(self):
        turn = TranscriptTurn(turn_index=0, speaker="I", text="test")
        assert turn.turn_index == 0


class TestBiasInstance:
    def _make(self, **kwargs) -> BiasInstance:
        defaults = dict(
            turn_index=1,
            speaker="Interviewer",
            text="Don't you think this is great?",
            bias_type=BiasType.LEADING_QUESTION,
            severity=Severity.HIGH,
            excerpt="Don't you think this is great?",
            explanation="Leading question.",
        )
        defaults.update(kwargs)
        return BiasInstance(**defaults)

    def test_creation(self):
        inst = self._make()
        assert inst.bias_type == BiasType.LEADING_QUESTION
        assert inst.severity == Severity.HIGH

    def test_optional_fields_default_none(self):
        inst = self._make()
        assert inst.llm_rationale is None
        assert inst.suggested_alternative is None

    def test_to_dict_keys(self):
        inst = self._make()
        d = inst.to_dict()
        for key in ["turn_index", "speaker", "text", "bias_type", "severity",
                    "excerpt", "explanation", "llm_rationale", "suggested_alternative"]:
            assert key in d

    def test_to_dict_enum_serialisation(self):
        inst = self._make()
        d = inst.to_dict()
        assert d["bias_type"] == "leading_question"
        assert d["severity"] == "high"

    def test_to_dict_with_llm_fields(self):
        inst = self._make(llm_rationale="test rationale", suggested_alternative="Better question?")
        d = inst.to_dict()
        assert d["llm_rationale"] == "test rationale"
        assert d["suggested_alternative"] == "Better question?"


class TestTranscriptReport:
    def test_basic_creation(self):
        report = TranscriptReport(
            session_id="s1",
            total_interviewer_turns=10,
            biased_turns=3,
            bias_rate=0.3,
        )
        assert report.session_id == "s1"
        assert report.bias_rate == 0.3

    def test_default_empty_lists(self):
        report = TranscriptReport(
            session_id="s1", total_interviewer_turns=0, biased_turns=0, bias_rate=0.0
        )
        assert report.instances == []
        assert report.bias_type_counts == {}
        assert report.severity_counts == {}

    def test_to_dict_rounds_bias_rate(self):
        report = TranscriptReport(
            session_id="s1", total_interviewer_turns=3, biased_turns=1, bias_rate=1 / 3
        )
        d = report.to_dict()
        assert d["bias_rate"] == round(1 / 3, 4)


class TestCorpusReport:
    def test_basic_creation(self):
        corpus = CorpusReport(
            total_sessions=3,
            total_interviewer_turns=30,
            total_biased_turns=9,
            overall_bias_rate=0.3,
        )
        assert corpus.total_sessions == 3
        assert corpus.overall_bias_rate == 0.3

    def test_default_empty_collections(self):
        corpus = CorpusReport(
            total_sessions=0, total_interviewer_turns=0,
            total_biased_turns=0, overall_bias_rate=0.0
        )
        assert corpus.per_session == []
        assert corpus.corpus_bias_type_counts == {}
        assert corpus.worst_sessions == []

    def test_to_dict_includes_per_session(self):
        r = TranscriptReport(
            session_id="s1", total_interviewer_turns=5, biased_turns=1, bias_rate=0.2
        )
        corpus = CorpusReport(
            total_sessions=1, total_interviewer_turns=5,
            total_biased_turns=1, overall_bias_rate=0.2,
            per_session=[r],
        )
        d = corpus.to_dict()
        assert len(d["per_session"]) == 1
        assert d["per_session"][0]["session_id"] == "s1"
