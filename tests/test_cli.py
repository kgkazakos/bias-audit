"""Tests for bias_audit CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bias_audit.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    def test_analyze_returns_zero_exit(self, runner):
        result = runner.invoke(cli, ["analyze", str(FIXTURES / "sample_transcript.json")])
        assert result.exit_code == 0

    def test_analyze_clean_transcript(self, runner):
        result = runner.invoke(cli, ["analyze", str(FIXTURES / "clean_transcript.json")])
        assert result.exit_code == 0

    def test_analyze_json_output_parseable(self, runner):
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "session_id" in data
        assert "bias_rate" in data
        assert "instances" in data

    def test_analyze_json_bias_rate_in_range(self, runner):
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json"],
        )
        data = json.loads(result.output)
        assert 0.0 <= data["bias_rate"] <= 1.0

    def test_analyze_detects_bias_in_sample(self, runner):
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json"],
        )
        data = json.loads(result.output)
        assert data["biased_turns"] > 0

    def test_analyze_clean_has_zero_bias(self, runner):
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "clean_transcript.json"), "--format", "json",
             "--min-severity", "medium"],
        )
        data = json.loads(result.output)
        assert data["biased_turns"] == 0

    def test_analyze_custom_interviewer_label(self, runner):
        result = runner.invoke(
            cli,
            [
                "analyze",
                str(FIXTURES / "sample_transcript.json"),
                "--interviewer", "Interviewer",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_min_severity_high_reduces_instances(self, runner):
        low_result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json",
             "--min-severity", "low"],
        )
        high_result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json",
             "--min-severity", "high"],
        )
        low_data = json.loads(low_result.output)
        high_data = json.loads(high_result.output)
        assert len(low_data["instances"]) >= len(high_data["instances"])

    def test_analyze_output_to_file(self, runner, tmp_path):
        out = tmp_path / "report.json"
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"),
             "--format", "json", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "session_id" in data

    def test_analyze_nonexistent_file(self, runner):
        result = runner.invoke(cli, ["analyze", "/no/such/file.json"])
        assert result.exit_code != 0

    def test_analyze_instances_contain_required_fields(self, runner):
        result = runner.invoke(
            cli,
            ["analyze", str(FIXTURES / "sample_transcript.json"), "--format", "json"],
        )
        data = json.loads(result.output)
        if data["instances"]:
            inst = data["instances"][0]
            for field in ["turn_index", "bias_type", "severity", "excerpt", "explanation"]:
                assert field in inst


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


class TestBatchCommand:
    def test_batch_returns_zero_exit(self, runner):
        result = runner.invoke(cli, ["batch", str(FIXTURES / "corpus")])
        assert result.exit_code == 0

    def test_batch_json_output_parseable(self, runner):
        result = runner.invoke(
            cli, ["batch", str(FIXTURES / "corpus"), "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_sessions" in data
        assert "overall_bias_rate" in data

    def test_batch_session_count(self, runner):
        result = runner.invoke(
            cli, ["batch", str(FIXTURES / "corpus"), "--format", "json"]
        )
        data = json.loads(result.output)
        assert data["total_sessions"] == 3

    def test_batch_per_session_in_output(self, runner):
        result = runner.invoke(
            cli, ["batch", str(FIXTURES / "corpus"), "--format", "json"]
        )
        data = json.loads(result.output)
        assert len(data["per_session"]) == 3

    def test_batch_output_to_file(self, runner, tmp_path):
        out = tmp_path / "corpus.json"
        result = runner.invoke(
            cli,
            ["batch", str(FIXTURES / "corpus"), "--format", "json", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()

    def test_batch_nonexistent_dir(self, runner):
        result = runner.invoke(cli, ["batch", "/no/such/directory"])
        assert result.exit_code != 0

    def test_batch_min_severity_medium(self, runner):
        result = runner.invoke(
            cli,
            ["batch", str(FIXTURES / "corpus"), "--format", "json", "--min-severity", "medium"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "overall_bias_rate" in data


# ---------------------------------------------------------------------------
# list-types command
# ---------------------------------------------------------------------------


class TestListTypesCommand:
    def test_list_types_exits_zero(self, runner):
        result = runner.invoke(cli, ["list-types"])
        assert result.exit_code == 0

    def test_list_types_shows_all_nine(self, runner):
        result = runner.invoke(cli, ["list-types"])
        # Check a selection of expected bias type names appear
        assert "Leading Question" in result.output
        assert "Loaded Question" in result.output
        assert "Double-Barreled" in result.output
        assert "Social Desirability" in result.output

    def test_list_types_shows_severity(self, runner):
        result = runner.invoke(cli, ["list-types"])
        assert "HIGH" in result.output
        assert "MEDIUM" in result.output
        assert "LOW" in result.output


# ---------------------------------------------------------------------------
# version / help
# ---------------------------------------------------------------------------


class TestMisc:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_analyze_help(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_batch_help(self, runner):
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
