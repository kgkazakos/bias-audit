"""Optional DSPy-backed LLM enrichment for bias instances and corpus narratives."""

from __future__ import annotations

from typing import Optional

from bias_audit.models import BiasInstance, CorpusReport, TranscriptReport

_SUPPORTED_BACKENDS = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "gemini": "gemini/gemini-3.6-flash",
}


def _configure_dspy(model_string: str) -> bool:
    """Configure DSPy with LiteLLM. Returns True on success."""
    try:
        import dspy  # type: ignore

        lm = dspy.LM(model=model_string)
        dspy.configure(lm=lm)
        return True
    except Exception:
        return False


class _BiasEnrichment:
    """DSPy signature: enrich a single bias instance."""

    @staticmethod
    def signature_class():  # type: ignore
        try:
            import dspy  # type: ignore

            class BiasEnrichmentSig(dspy.Signature):
                """
                You are an expert UX research methodologist.
                Given a detected interviewer bias instance, provide:
                1. A nuanced rationale explaining why this is methodologically problematic.
                2. A concrete, neutral alternative phrasing the interviewer could have used instead.
                """

                bias_type: str = dspy.InputField(desc="The category of bias detected")
                interviewer_turn: str = dspy.InputField(
                    desc="The interviewer's exact words"
                )
                rule_explanation: str = dspy.InputField(
                    desc="The rule-based explanation for flagging this turn"
                )
                rationale: str = dspy.OutputField(
                    desc=(
                        "2–3 sentence methodological rationale explaining the risk "
                        "this bias introduces to data validity"
                    )
                )
                suggested_alternative: str = dspy.OutputField(
                    desc="A neutral, open-ended rephrasing of the interviewer's turn"
                )

            return BiasEnrichmentSig
        except ImportError:
            return None


class _CorpusNarrative:
    """DSPy signature: narrative summary of corpus bias patterns."""

    @staticmethod
    def signature_class():  # type: ignore
        try:
            import dspy  # type: ignore

            class CorpusNarrativeSig(dspy.Signature):
                """
                You are an expert UX research methodologist writing a data quality assessment.
                Given corpus-level bias statistics, produce a concise methodological
                narrative suitable for inclusion in a research report.
                """

                total_sessions: str = dspy.InputField(desc="Number of sessions analysed")
                overall_bias_rate: str = dspy.InputField(
                    desc="Proportion of interviewer turns containing at least one bias flag"
                )
                top_bias_types: str = dspy.InputField(
                    desc="Comma-separated list of the most prevalent bias types and their counts"
                )
                worst_sessions: str = dspy.InputField(
                    desc="Session IDs with the highest bias rates"
                )
                narrative: str = dspy.OutputField(
                    desc=(
                        "3–5 sentence methodological assessment of data quality risks "
                        "introduced by the observed bias patterns, written in academic register"
                    )
                )

            return CorpusNarrativeSig
        except ImportError:
            return None


def enrich_instances(
    instances: list[BiasInstance],
    llm_backend: str,
) -> list[BiasInstance]:
    """
    Use DSPy ChainOfThought to add llm_rationale and suggested_alternative
    to each bias instance. Returns the same list (mutated in-place).
    """
    model_string = _SUPPORTED_BACKENDS.get(llm_backend, llm_backend)
    if not _configure_dspy(model_string):
        return instances

    try:
        import dspy  # type: ignore

        sig_cls = _BiasEnrichment.signature_class()
        if sig_cls is None:
            return instances

        predictor = dspy.ChainOfThought(sig_cls)

        for inst in instances:
            try:
                result = predictor(
                    bias_type=inst.bias_type.value,
                    interviewer_turn=inst.text,
                    rule_explanation=inst.explanation,
                )
                inst.llm_rationale = getattr(result, "rationale", None)
                inst.suggested_alternative = getattr(result, "suggested_alternative", None)
            except Exception:
                # Per-instance failure is non-fatal
                pass

    except Exception:
        pass

    return instances


def enrich_transcript_report(
    report: TranscriptReport,
    llm_backend: str,
) -> TranscriptReport:
    """Add LLM rationale to all instances in a transcript report."""
    report.instances = enrich_instances(report.instances, llm_backend)
    return report


def enrich_corpus_report(
    report: CorpusReport,
    llm_backend: str,
) -> CorpusReport:
    """Add LLM narrative summary to a corpus report."""
    model_string = _SUPPORTED_BACKENDS.get(llm_backend, llm_backend)
    if not _configure_dspy(model_string):
        return report

    try:
        import dspy  # type: ignore

        sig_cls = _CorpusNarrative.signature_class()
        if sig_cls is None:
            return report

        predictor = dspy.ChainOfThought(sig_cls)

        top_types = sorted(
            report.corpus_bias_type_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_str = ", ".join(f"{k} ({v})" for k, v in top_types)

        result = predictor(
            total_sessions=str(report.total_sessions),
            overall_bias_rate=f"{report.overall_bias_rate:.1%}",
            top_bias_types=top_str,
            worst_sessions=", ".join(report.worst_sessions),
        )
        report.corpus_narrative = getattr(result, "narrative", None)

    except Exception:
        pass

    return report


def resolve_backend(backend_key: str) -> Optional[str]:
    """Return the LiteLLM model string for a given backend key, or None."""
    return _SUPPORTED_BACKENDS.get(backend_key, backend_key if "/" in backend_key else None)
