"""Click CLI for bias-audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from bias_audit.aggregator import (
    build_corpus_report,
    build_transcript_report,
    session_id_from_path,
)
from bias_audit.detector import BIAS_LABELS, detect_bias_in_transcript
from bias_audit.models import Severity, TranscriptReport
from bias_audit.parser import infer_interviewer_label, parse_transcript

_SEVERITY_MAP = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}
_SEVERITY_COLOURS = {
    Severity.LOW: "yellow",
    Severity.MEDIUM: "magenta",
    Severity.HIGH: "red",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_transcript_report(report: TranscriptReport) -> None:
    click.echo()
    click.secho(f"Session: {report.session_id}", bold=True)
    click.echo(
        f"  Interviewer turns : {report.total_interviewer_turns}"
    )
    click.echo(f"  Biased turns      : {report.biased_turns}")
    click.secho(
        f"  Bias rate         : {report.bias_rate:.1%}",
        fg="red" if report.bias_rate > 0.3 else "yellow" if report.bias_rate > 0.1 else "green",
    )
    if report.most_common_bias:
        click.echo(f"  Most common bias  : {report.most_common_bias}")
    if report.narrative_summary:
        click.echo(f"\n  Narrative: {report.narrative_summary}")

    if report.instances:
        click.echo()
        click.secho("  Detected instances:", bold=True)
        for inst in report.instances:
            colour = _SEVERITY_COLOURS.get(inst.severity, "white")
            click.secho(
                f"    [Turn {inst.turn_index:>3}] [{inst.severity.value.upper():>6}] "
                f"{BIAS_LABELS.get(inst.bias_type, inst.bias_type.value)}",
                fg=colour,
            )
            click.echo(f'      Excerpt : "{inst.excerpt}"')
            click.echo(f"      Reason  : {inst.explanation}")
            if inst.llm_rationale:
                click.echo(f"      Rationale : {inst.llm_rationale}")
            if inst.suggested_alternative:
                click.secho(
                    f"      Better  : {inst.suggested_alternative}", fg="cyan"
                )
    else:
        click.secho("  No bias instances detected.", fg="green")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="bias-audit")
def cli() -> None:
    """bias-audit — detect and categorise interviewer bias in transcript corpora."""


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("transcript", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--interviewer",
    default=None,
    help="Speaker label for the interviewer (auto-detected if omitted).",
)
@click.option(
    "--min-severity",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default="low",
    show_default=True,
    help="Minimum severity level to report.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write output to file instead of stdout.",
)
@click.option(
    "--llm-backend",
    default=None,
    help="LLM backend for enrichment: openai | anthropic | gemini | custom LiteLLM string.",
)
def analyze(
    transcript: Path,
    interviewer: Optional[str],
    min_severity: str,
    output_format: str,
    output: Optional[Path],
    llm_backend: Optional[str],
) -> None:
    """Analyse a single transcript for interviewer bias."""
    try:
        turns = parse_transcript(transcript)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)

    if not turns:
        click.secho("No turns parsed from transcript.", fg="yellow", err=True)
        sys.exit(1)

    label = interviewer or infer_interviewer_label(turns)
    min_sev = _SEVERITY_MAP[min_severity.lower()]

    instances, total_turns = detect_bias_in_transcript(
        turns, interviewer_label=label, min_severity=min_sev
    )

    if llm_backend:
        from bias_audit.llm import enrich_instances

        instances = enrich_instances(instances, llm_backend)

    report = build_transcript_report(
        session_id=session_id_from_path(transcript),
        instances=instances,
        total_interviewer_turns=total_turns,
    )

    if output_format == "json":
        result = json.dumps(report.to_dict(), indent=2)
        if output:
            output.write_text(result, encoding="utf-8")
            click.echo(f"Report written to {output}")
        else:
            click.echo(result)
    else:
        if output:
            # Redirect text output to file
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _print_transcript_report(report)
            output.write_text(buf.getvalue(), encoding="utf-8")
            click.echo(f"Report written to {output}")
        else:
            _print_transcript_report(report)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--interviewer", default=None, help="Speaker label for the interviewer.")
@click.option(
    "--min-severity",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default="medium",
    show_default=True,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--llm-backend", default=None)
def batch(
    directory: Path,
    interviewer: Optional[str],
    min_severity: str,
    output_format: str,
    output: Optional[Path],
    llm_backend: Optional[str],
) -> None:
    """Analyse all transcripts in a directory and produce a corpus report."""
    extensions = {".json", ".txt", ".md"}
    files = sorted(p for p in directory.iterdir() if p.suffix in extensions)

    if not files:
        click.secho("No supported transcript files found.", fg="yellow", err=True)
        sys.exit(1)

    min_sev = _SEVERITY_MAP[min_severity.lower()]
    session_reports: list[TranscriptReport] = []

    for f in files:
        try:
            turns = parse_transcript(f)
        except Exception as exc:
            click.secho(f"  Skipping {f.name}: {exc}", fg="yellow", err=True)
            continue

        if not turns:
            continue

        label = interviewer or infer_interviewer_label(turns)
        instances, total_turns = detect_bias_in_transcript(
            turns, interviewer_label=label, min_severity=min_sev
        )

        if llm_backend:
            from bias_audit.llm import enrich_instances

            instances = enrich_instances(instances, llm_backend)

        report = build_transcript_report(
            session_id=session_id_from_path(f),
            instances=instances,
            total_interviewer_turns=total_turns,
        )
        session_reports.append(report)
        if output_format == "text":
            click.secho(f"  ✓ {f.name} — bias rate {report.bias_rate:.1%}", fg="green", err=True)

    if not session_reports:
        click.secho("No sessions processed.", fg="red", err=True)
        sys.exit(1)

    corpus = build_corpus_report(session_reports)

    if llm_backend:
        from bias_audit.llm import enrich_corpus_report

        corpus = enrich_corpus_report(corpus, llm_backend)

    if output_format == "json":
        result = json.dumps(corpus.to_dict(), indent=2)
        if output:
            output.write_text(result, encoding="utf-8")
            click.echo(f"Corpus report written to {output}")
        else:
            click.echo(result)
    else:
        click.echo()
        click.secho("═" * 60, bold=True)
        click.secho("  CORPUS BIAS REPORT", bold=True)
        click.secho("═" * 60, bold=True)
        click.echo(f"  Sessions        : {corpus.total_sessions}")
        click.echo(f"  Interviewer turns: {corpus.total_interviewer_turns}")
        click.echo(f"  Biased turns    : {corpus.total_biased_turns}")
        rate_colour = (
            "red" if corpus.overall_bias_rate > 0.3
            else "yellow" if corpus.overall_bias_rate > 0.1
            else "green"
        )
        click.secho(
            f"  Overall bias rate: {corpus.overall_bias_rate:.1%}", fg=rate_colour, bold=True
        )
        if corpus.most_prevalent_bias:
            click.echo(f"  Most prevalent  : {corpus.most_prevalent_bias}")
        if corpus.worst_sessions:
            click.echo(f"  Worst sessions  : {', '.join(corpus.worst_sessions)}")
        if corpus.corpus_narrative:
            click.echo(f"\n  Narrative:\n  {corpus.corpus_narrative}")

        click.echo()
        click.secho("  Per-session summary:", bold=True)
        for r in sorted(session_reports, key=lambda x: x.bias_rate, reverse=True):
            colour = (
                "red" if r.bias_rate > 0.3 else "yellow" if r.bias_rate > 0.1 else "green"
            )
            click.secho(
                f"    {r.session_id:<30} bias rate: {r.bias_rate:.1%} "
                f"({r.biased_turns}/{r.total_interviewer_turns} turns)",
                fg=colour,
            )

        if output:
            click.secho(f"\nFull report available in: {output}", fg="cyan")


# ---------------------------------------------------------------------------
# list-types
# ---------------------------------------------------------------------------


@cli.command("list-types")
def list_types() -> None:
    """List all detectable bias types with descriptions."""
    from bias_audit.detector import _REGISTRY

    click.echo()
    click.secho("Detectable Bias Types", bold=True)
    click.echo("─" * 60)
    for bias_type, config in _REGISTRY.items():
        label = BIAS_LABELS.get(bias_type, bias_type.value)
        sev = config["severity"].value.upper()
        colour = _SEVERITY_COLOURS.get(config["severity"], "white")
        click.secho(f"  {label}", bold=True)
        click.secho(f"  Severity : {sev}", fg=colour)
        click.echo(f"  {config['explanation']}")
        click.echo()


if __name__ == "__main__":
    cli()
