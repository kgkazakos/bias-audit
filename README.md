# bias-audit

**Detect and categorise interviewer bias patterns across transcript corpora.**

Most teams assume their interviews are neutral. `bias-audit` runs the check. It scans interviewer turns for nine established bias categories — leading questions, loaded assumptions, answer suggestion, social desirability framing, and more — producing per-session and corpus-level reports with severity ratings and optional LLM-generated alternatives.

Works fully offline with zero API keys. LLM enrichment (rationale + rephrasing suggestions) is an optional add-on.

---

## The problem

Interviewer bias is one of the most common threats to qualitative data validity — and one of the least systematically checked. Researchers know the theory. In practice, under time pressure and with hypotheses forming mid-session, it creeps in. Loaded questions. Leading framings. Agreement injections that prime participants. By the time analysis begins, the signal is already contaminated.

`bias-audit` surfaces that contamination quantitatively, turning a heuristic sense-check into a repeatable data quality step.

---

## Features

- **9 bias categories** with severity ratings (HIGH / MEDIUM / LOW)
  - Leading Question, Loaded Question, Double-Barreled, Closed Question
  - Confirmation Reinforcement, Agreement Injection, Hypothetical Grounding
  - Answer Suggestion, Social Desirability Framing
- **Single-session analysis** — `bias-audit analyze transcript.json`
- **Corpus batch analysis** — `bias-audit batch ./transcripts/`
- **Per-turn flagging** — excerpt + explanation for every detected instance
- **Corpus aggregation** — overall bias rate, worst sessions, most prevalent type
- **Configurable severity threshold** — filter to MEDIUM/HIGH only
- **JSON output** — pipe into dashboards, CI scripts, or research logs
- **Optional LLM enrichment** — DSPy ChainOfThought adds rationale and neutral rephrasing suggestions
- **144 tests** passing across Python 3.10 / 3.11 / 3.12
- **Zero API keys required** — fully functional rule-based engine out of the box

---

## Install

```bash
pip install bias-audit
```

With optional LLM enrichment:

```bash
pip install "bias-audit[llm,openai]"
# or
pip install "bias-audit[llm,anthropic]"
# or
pip install "bias-audit[llm,google]"
```

---

## Usage

### Analyse a single transcript

```bash
bias-audit analyze interview_01.json

# With severity filter (skip LOW-severity closed questions)
bias-audit analyze interview_01.json --min-severity medium

# Custom interviewer speaker label
bias-audit analyze session.json --interviewer "Moderator"

# JSON output for piping
bias-audit analyze interview_01.json --format json | jq .bias_rate

# Write to file
bias-audit analyze interview_01.json --format json --output report.json
```

### Batch analysis across a corpus

```bash
bias-audit batch ./transcripts/

# JSON corpus report
bias-audit batch ./transcripts/ --format json --output corpus_report.json

# Filter to high-severity only
bias-audit batch ./transcripts/ --min-severity high
```

### List all detectable bias types

```bash
bias-audit list-types
```

### Optional LLM enrichment

```bash
# Add rationale and rephrasing suggestions via OpenAI
bias-audit analyze interview.json --llm-backend openai

# Or Anthropic / Gemini
bias-audit analyze interview.json --llm-backend anthropic
bias-audit analyze interview.json --llm-backend gemini
```

LLM enrichment adds `llm_rationale` and `suggested_alternative` fields to each flagged instance. The rule-based core runs identically with or without it.

---

## Input formats

### JSON (recommended)

```json
[
  {"speaker": "Interviewer", "text": "Tell me about your experience."},
  {"speaker": "Participant", "text": "It was fine overall."},
  {"speaker": "Interviewer", "text": "Don't you think the new layout is better?"}
]
```

Also accepts `role` / `content` key pairs (OpenAI message format).

### Plain text

```
Interviewer: Tell me about your experience.
Participant: It was fine overall.
Interviewer: Don't you think the new layout is better?
```

Supported extensions: `.json`, `.txt`, `.md`

---

## Bias types

| Bias Type | Severity | Example |
|-----------|----------|---------|
| Leading Question | HIGH | "Don't you think the layout is better?" |
| Loaded Question | HIGH | "Why did you struggle to find the button?" |
| Answer Suggestion | HIGH | "Was it because the icon was too small?" |
| Social Desirability | HIGH | "Most users say the nav is confusing. Did you?" |
| Confirmation Reinforcement | MEDIUM | "So basically you're saying it's broken, right?" |
| Double-Barreled | MEDIUM | "How did you feel and what would you change?" |
| Hypothetical Grounding | MEDIUM | "If you were redesigning this, what would you do?" |
| Agreement Injection | LOW | "Exactly!" / "Of course!" |
| Closed Question | LOW | "Did you find the export button?" |

---

## Output format

```json
{
  "session_id": "interview_01",
  "total_interviewer_turns": 11,
  "biased_turns": 4,
  "bias_rate": 0.3636,
  "most_common_bias": "leading_question",
  "instances": [
    {
      "turn_index": 2,
      "bias_type": "leading_question",
      "severity": "high",
      "excerpt": "Don't you think the new layout is better?",
      "explanation": "Leading question that presupposes...",
      "llm_rationale": null,
      "suggested_alternative": null
    }
  ]
}
```

---

## Python API

```python
from bias_audit import (
    parse_transcript,
    detect_bias_in_transcript,
    build_transcript_report,
    build_corpus_report,
)
from bias_audit.models import Severity
from pathlib import Path

# Single session
turns = parse_transcript(Path("interview.json"))
instances, total_turns = detect_bias_in_transcript(
    turns,
    interviewer_label="Interviewer",
    min_severity=Severity.MEDIUM,
)
report = build_transcript_report("session_01", instances, total_turns)
print(f"Bias rate: {report.bias_rate:.1%}")

# Corpus
session_reports = []
for f in Path("transcripts/").glob("*.json"):
    t = parse_transcript(f)
    insts, total = detect_bias_in_transcript(t, "Interviewer")
    session_reports.append(build_transcript_report(f.stem, insts, total))

corpus = build_corpus_report(session_reports)
print(f"Overall bias rate: {corpus.overall_bias_rate:.1%}")
print(f"Worst sessions: {corpus.worst_sessions}")
```

---

## Part of the CPR Ecosystem

`bias-audit` is Tool 14 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source Python tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools:**
- [`qual-sniffer`](https://github.com/kgkazakos/qual-sniffer) — detects methodological weaknesses in discussion guides *before* fieldwork (complementary: prevents bias, while `bias-audit` detects it after)
- [`guide-scorer`](https://github.com/kgkazakos/guide-scorer) — scores discussion guides across 8 quality dimensions
- [`say-do-lite`](https://github.com/kgkazakos/say-do-lite) — lightweight Say-Do Gap detection from transcripts
- [`coding-eval`](https://github.com/kgkazakos/coding-eval) — measures LLM thematic coding accuracy against human coders

**The CPR discipline** treats product research as an engineering problem: reproducible methods, versioned artefacts, quantifiable quality, and systematic bias control.

---

## Development

```bash
git clone https://github.com/kgkazakos/bias-audit
cd bias-audit
pip install -e ".[dev]"
pytest tests/ -v
ruff check bias_audit/ tests/
```

CI runs on Python 3.10, 3.11, and 3.12.

---

## License

MIT — Kostas Kazakos, 2026
