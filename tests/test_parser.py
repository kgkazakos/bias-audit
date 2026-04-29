"""Tests for bias_audit.parser."""

import json
from pathlib import Path

import pytest

from bias_audit.models import TranscriptTurn
from bias_audit.parser import (
    infer_interviewer_label,
    parse_json_transcript,
    parse_text_transcript,
    parse_transcript,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseJsonTranscript:
    def test_loads_sample_transcript(self):
        turns = parse_json_transcript(FIXTURES / "sample_transcript.json")
        assert len(turns) > 0

    def test_loads_clean_transcript(self):
        turns = parse_json_transcript(FIXTURES / "clean_transcript.json")
        assert len(turns) > 0

    def test_turn_types(self):
        turns = parse_json_transcript(FIXTURES / "sample_transcript.json")
        for t in turns:
            assert isinstance(t, TranscriptTurn)

    def test_speakers_preserved(self):
        turns = parse_json_transcript(FIXTURES / "sample_transcript.json")
        speakers = {t.speaker for t in turns}
        assert "Interviewer" in speakers
        assert "Participant" in speakers

    def test_turn_indices_sequential(self):
        turns = parse_json_transcript(FIXTURES / "sample_transcript.json")
        for i, t in enumerate(turns):
            assert t.turn_index == i

    def test_role_field_fallback(self, tmp_path):
        data = [{"role": "Interviewer", "content": "Tell me more."}]
        p = tmp_path / "t.json"
        p.write_text(json.dumps(data))
        turns = parse_json_transcript(p)
        assert len(turns) == 1
        assert turns[0].speaker == "Interviewer"
        assert turns[0].text == "Tell me more."

    def test_empty_text_skipped(self, tmp_path):
        data = [
            {"speaker": "Interviewer", "text": ""},
            {"speaker": "Participant", "text": "  "},
            {"speaker": "Interviewer", "text": "Good."},
        ]
        p = tmp_path / "t.json"
        p.write_text(json.dumps(data))
        turns = parse_json_transcript(p)
        assert len(turns) == 1

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json")
        with pytest.raises(Exception):
            parse_json_transcript(p)

    def test_non_list_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"key": "value"}))
        with pytest.raises(ValueError):
            parse_json_transcript(p)


class TestParseTextTranscript:
    def _write_txt(self, tmp_path, content: str) -> Path:
        p = tmp_path / "transcript.txt"
        p.write_text(content)
        return p

    def test_basic_parse(self, tmp_path):
        content = "Interviewer: Tell me about your experience.\nParticipant: It was fine.\n"
        p = self._write_txt(tmp_path, content)
        turns = parse_text_transcript(p)
        assert len(turns) == 2

    def test_speakers_extracted(self, tmp_path):
        content = "Interviewer: Hello.\nParticipant: Hi.\n"
        p = self._write_txt(tmp_path, content)
        turns = parse_text_transcript(p)
        assert turns[0].speaker == "Interviewer"
        assert turns[1].speaker == "Participant"

    def test_text_extracted(self, tmp_path):
        content = "Interviewer: Can you walk me through that?\nParticipant: Sure.\n"
        p = self._write_txt(tmp_path, content)
        turns = parse_text_transcript(p)
        assert "walk me through" in turns[0].text

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        turns = parse_text_transcript(p)
        assert turns == []


class TestParseTranscript:
    def test_json_dispatch(self):
        turns = parse_transcript(FIXTURES / "sample_transcript.json")
        assert len(turns) > 0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_transcript(Path("/nonexistent/file.json"))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "file.pdf"
        p.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_transcript(p)

    def test_corpus_sessions_parse(self):
        corpus_dir = FIXTURES / "corpus"
        for f in corpus_dir.glob("*.json"):
            turns = parse_transcript(f)
            assert len(turns) > 0


class TestInferInterviewerLabel:
    def test_recognises_interviewer_label(self):
        turns = [
            TranscriptTurn(0, "Interviewer", "Hello."),
            TranscriptTurn(1, "Participant", "Hi."),
            TranscriptTurn(2, "Interviewer", "Tell me more."),
            TranscriptTurn(3, "Participant", "Sure."),
        ]
        label = infer_interviewer_label(turns)
        assert label == "Interviewer"

    def test_recognises_moderator(self):
        turns = [
            TranscriptTurn(0, "Moderator", "Welcome."),
            TranscriptTurn(1, "Participant", "Thanks."),
        ]
        label = infer_interviewer_label(turns)
        assert label == "Moderator"

    def test_fallback_for_unknown_labels(self):
        turns = [
            TranscriptTurn(0, "Alex", "Question?"),
            TranscriptTurn(1, "Alex", "Question 2?"),
            TranscriptTurn(2, "Jordan", "Answer."),
            TranscriptTurn(3, "Jordan", "Answer 2."),
            TranscriptTurn(4, "Jordan", "Answer 3."),
        ]
        # Fallback: least frequent speaker = Alex
        label = infer_interviewer_label(turns)
        assert label == "Alex"

    def test_empty_turns_returns_default(self):
        label = infer_interviewer_label([])
        assert label == "Interviewer"
