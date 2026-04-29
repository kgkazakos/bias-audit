"""Tests for bias_audit.detector — rule-based detection engine."""


from bias_audit.detector import (
    BIAS_LABELS,
    detect_bias_in_transcript,
    detect_bias_in_turn,
)
from bias_audit.models import BiasType, Severity, TranscriptTurn


def _turn(text: str, index: int = 0, speaker: str = "Interviewer") -> TranscriptTurn:
    return TranscriptTurn(turn_index=index, speaker=speaker, text=text)


def _has_type(instances, bias_type: BiasType) -> bool:
    return any(i.bias_type == bias_type for i in instances)


# ---------------------------------------------------------------------------
# Leading question
# ---------------------------------------------------------------------------


class TestLeadingQuestion:
    def test_dont_you_think(self):
        instances = detect_bias_in_turn(_turn("Don't you think the new layout is better?"))
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_wouldnt_you_say(self):
        instances = detect_bias_in_turn(_turn("Wouldn't you say that was a good experience?"))
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_isnt_it_true(self):
        instances = detect_bias_in_turn(_turn("Isn't it true that the navigation is confusing?"))
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_you_would_agree(self):
        instances = detect_bias_in_turn(_turn("You would agree that was frustrating, right?"))
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_surely_you(self):
        instances = detect_bias_in_turn(_turn("Surely you noticed the button was missing?"))
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_neutral_question_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Can you walk me through what happened next?"))
        assert not _has_type(instances, BiasType.LEADING_QUESTION)

    def test_open_probe_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Tell me more about that experience."))
        assert not _has_type(instances, BiasType.LEADING_QUESTION)


# ---------------------------------------------------------------------------
# Loaded question
# ---------------------------------------------------------------------------


class TestLoadedQuestion:
    def test_why_did_you_struggle(self):
        instances = detect_bias_in_turn(_turn("Why did you struggle to find the export button?"))
        assert _has_type(instances, BiasType.LOADED_QUESTION)

    def test_what_made_it_difficult(self):
        instances = detect_bias_in_turn(_turn("What made it so difficult to complete the task?"))
        assert _has_type(instances, BiasType.LOADED_QUESTION)

    def test_when_did_you_give_up(self):
        instances = detect_bias_in_turn(_turn("When did you give up on the onboarding?"))
        assert _has_type(instances, BiasType.LOADED_QUESTION)

    def test_what_went_wrong(self):
        instances = detect_bias_in_turn(_turn("What went wrong during the checkout flow?"))
        assert _has_type(instances, BiasType.LOADED_QUESTION)

    def test_neutral_why_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Why did you choose that option?"))
        assert not _has_type(instances, BiasType.LOADED_QUESTION)


# ---------------------------------------------------------------------------
# Double-barreled
# ---------------------------------------------------------------------------


class TestDoubleBarreled:
    def test_two_question_words(self):
        instances = detect_bias_in_turn(
            _turn("How did you feel about the experience and what would you change?")
        )
        assert _has_type(instances, BiasType.DOUBLE_BARRELED)

    def test_and_also(self):
        instances = detect_bias_in_turn(
            _turn("Tell me about the login flow and also about the dashboard?")
        )
        assert _has_type(instances, BiasType.DOUBLE_BARRELED)

    def test_as_well_as(self):
        instances = detect_bias_in_turn(
            _turn("What did you think of the onboarding as well as the help docs?")
        )
        assert _has_type(instances, BiasType.DOUBLE_BARRELED)

    def test_single_question_not_flagged(self):
        instances = detect_bias_in_turn(_turn("How did you feel about the experience?"))
        assert not _has_type(instances, BiasType.DOUBLE_BARRELED)


# ---------------------------------------------------------------------------
# Closed question
# ---------------------------------------------------------------------------


class TestClosedQuestion:
    def test_did_you(self):
        instances = detect_bias_in_turn(_turn("Did you find the export button?"))
        assert _has_type(instances, BiasType.CLOSED_QUESTION)

    def test_is_it(self):
        instances = detect_bias_in_turn(_turn("Is it clear where to find the settings?"))
        assert _has_type(instances, BiasType.CLOSED_QUESTION)

    def test_do_you(self):
        instances = detect_bias_in_turn(_turn("Do you use this feature regularly?"))
        assert _has_type(instances, BiasType.CLOSED_QUESTION)

    def test_would_you_open_ended(self):
        # "Would you describe..." is hypothetical, not necessarily closed
        # but "Would you" as a starter can be closed
        instances = detect_bias_in_turn(_turn("Would you recommend this to a colleague?"))
        assert _has_type(instances, BiasType.CLOSED_QUESTION)

    def test_open_question_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Tell me about your typical workflow."))
        assert not _has_type(instances, BiasType.CLOSED_QUESTION)

    def test_what_question_not_flagged_as_closed(self):
        instances = detect_bias_in_turn(_turn("What happens when you click that button?"))
        assert not _has_type(instances, BiasType.CLOSED_QUESTION)


# ---------------------------------------------------------------------------
# Confirmation reinforcement
# ---------------------------------------------------------------------------


class TestConfirmationReinforcement:
    def test_so_basically(self):
        instances = detect_bias_in_turn(
            _turn("So basically you're saying the navigation is broken?")
        )
        assert _has_type(instances, BiasType.CONFIRMATION_REINFORCEMENT)

    def test_just_as_expected(self):
        instances = detect_bias_in_turn(
            _turn("That's just as I expected from our earlier research.")
        )
        assert _has_type(instances, BiasType.CONFIRMATION_REINFORCEMENT)

    def test_right_tag(self):
        instances = detect_bias_in_turn(_turn("So the export is the main bottleneck, right?"))
        assert _has_type(instances, BiasType.CONFIRMATION_REINFORCEMENT)

    def test_correct_tag(self):
        instances = detect_bias_in_turn(_turn("That confirms what I thought about the UX, correct?"))
        assert _has_type(instances, BiasType.CONFIRMATION_REINFORCEMENT)

    def test_neutral_summary_not_flagged(self):
        instances = detect_bias_in_turn(_turn("So you use it in the morning. Tell me more."))
        assert not _has_type(instances, BiasType.CONFIRMATION_REINFORCEMENT)


# ---------------------------------------------------------------------------
# Agreement injection
# ---------------------------------------------------------------------------


class TestAgreementInjection:
    def test_exactly_standalone(self):
        instances = detect_bias_in_turn(_turn("Exactly!"))
        assert _has_type(instances, BiasType.AGREEMENT_INJECTION)

    def test_of_course(self):
        instances = detect_bias_in_turn(_turn("Of course."))
        assert _has_type(instances, BiasType.AGREEMENT_INJECTION)

    def test_absolutely(self):
        instances = detect_bias_in_turn(_turn("Absolutely!"))
        assert _has_type(instances, BiasType.AGREEMENT_INJECTION)

    def test_exactly_then_question(self):
        instances = detect_bias_in_turn(_turn("Exactly. So tell me more about that."))
        assert _has_type(instances, BiasType.AGREEMENT_INJECTION)

    def test_neutral_acknowledgement_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Okay, that makes sense."))
        assert not _has_type(instances, BiasType.AGREEMENT_INJECTION)


# ---------------------------------------------------------------------------
# Hypothetical grounding
# ---------------------------------------------------------------------------


class TestHypotheticalGrounding:
    def test_if_you_were(self):
        instances = detect_bias_in_turn(_turn("If you were redesigning this, what would you change?"))
        assert _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)

    def test_imagine_if(self):
        instances = detect_bias_in_turn(_turn("Imagine if the button was at the top — would that help?"))
        assert _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)

    def test_how_would_you(self):
        instances = detect_bias_in_turn(_turn("How would you feel about a simpler interface?"))
        assert _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)

    def test_could_you_imagine(self):
        instances = detect_bias_in_turn(_turn("Could you imagine using this on a phone instead?"))
        assert _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)

    def test_lets_say(self):
        instances = detect_bias_in_turn(_turn("Let's say you needed to export urgently. What would you do?"))
        assert _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)

    def test_past_behaviour_not_flagged(self):
        instances = detect_bias_in_turn(_turn("Walk me through the last time you used it."))
        assert not _has_type(instances, BiasType.HYPOTHETICAL_GROUNDING)


# ---------------------------------------------------------------------------
# Answer suggestion
# ---------------------------------------------------------------------------


class TestAnswerSuggestion:
    def test_was_it_because(self):
        instances = detect_bias_in_turn(_turn("Was it because the button was too small?"))
        assert _has_type(instances, BiasType.ANSWER_SUGGESTION)

    def test_did_you_feel_frustrated(self):
        instances = detect_bias_in_turn(_turn("Did you feel frustrated when that happened?"))
        assert _has_type(instances, BiasType.ANSWER_SUGGESTION)

    def test_was_that_confusing(self):
        instances = detect_bias_in_turn(_turn("Was that confusing for you?"))
        assert _has_type(instances, BiasType.ANSWER_SUGGESTION)

    def test_so_it_felt_like(self):
        instances = detect_bias_in_turn(_turn("So it felt like you were stuck at that point?"))
        assert _has_type(instances, BiasType.ANSWER_SUGGESTION)

    def test_neutral_experience_not_flagged(self):
        instances = detect_bias_in_turn(_turn("How did you feel at that point?"))
        assert not _has_type(instances, BiasType.ANSWER_SUGGESTION)


# ---------------------------------------------------------------------------
# Social desirability
# ---------------------------------------------------------------------------


class TestSocialDesirability:
    def test_most_users(self):
        instances = detect_bias_in_turn(
            _turn("Most users we've spoken to found this confusing. Did you?")
        )
        assert _has_type(instances, BiasType.SOCIAL_DESIRABILITY)

    def test_everyone_else(self):
        instances = detect_bias_in_turn(
            _turn("Everyone else we've spoken to prefers the old design.")
        )
        assert _has_type(instances, BiasType.SOCIAL_DESIRABILITY)

    def test_studies_show(self):
        instances = detect_bias_in_turn(_turn("Studies show that simpler interfaces are preferred."))
        assert _has_type(instances, BiasType.SOCIAL_DESIRABILITY)

    def test_research_suggests(self):
        instances = detect_bias_in_turn(_turn("Research suggests people usually prefer dark mode."))
        assert _has_type(instances, BiasType.SOCIAL_DESIRABILITY)

    def test_neutral_reference_not_flagged(self):
        instances = detect_bias_in_turn(_turn("What do you think about the colour scheme?"))
        assert not _has_type(instances, BiasType.SOCIAL_DESIRABILITY)


# ---------------------------------------------------------------------------
# detect_bias_in_transcript
# ---------------------------------------------------------------------------


class TestDetectInTranscript:
    def _sample_turns(self):
        return [
            TranscriptTurn(0, "Interviewer", "Don't you think the layout is great?"),
            TranscriptTurn(1, "Participant", "It's okay."),
            TranscriptTurn(2, "Interviewer", "Tell me more about what you liked."),
            TranscriptTurn(3, "Participant", "The colours are nice."),
            TranscriptTurn(4, "Interviewer", "Most users say it's the best feature. Agree?"),
        ]

    def test_only_interviewer_turns_checked(self):
        turns = self._sample_turns()
        instances, total_turns = detect_bias_in_transcript(turns, "Interviewer")
        for inst in instances:
            assert inst.speaker == "Interviewer"

    def test_total_interviewer_turns_count(self):
        turns = self._sample_turns()
        _, total_turns = detect_bias_in_transcript(turns, "Interviewer")
        assert total_turns == 3  # turns 0, 2, 4

    def test_clean_interviewer_no_instances(self):
        turns = [
            TranscriptTurn(0, "Interviewer", "Walk me through your experience."),
            TranscriptTurn(1, "Participant", "Sure."),
            TranscriptTurn(2, "Interviewer", "What happened next?"),
        ]
        instances, _ = detect_bias_in_transcript(turns, "Interviewer")
        assert instances == []

    def test_min_severity_filters_low(self):
        turns = [
            TranscriptTurn(0, "Interviewer", "Did you finish the task?"),  # closed=LOW
        ]
        instances_all, _ = detect_bias_in_transcript(turns, "Interviewer", min_severity=Severity.LOW)
        instances_med, _ = detect_bias_in_transcript(turns, "Interviewer", min_severity=Severity.MEDIUM)
        assert len(instances_all) >= len(instances_med)

    def test_case_insensitive_speaker_matching(self):
        turns = [
            TranscriptTurn(0, "interviewer", "Don't you think this is great?"),
        ]
        instances, total = detect_bias_in_transcript(turns, "Interviewer")
        assert total == 1
        assert _has_type(instances, BiasType.LEADING_QUESTION)

    def test_unknown_speaker_ignored(self):
        turns = [
            TranscriptTurn(0, "SomeOtherPerson", "Don't you think this is bad?"),
        ]
        instances, total = detect_bias_in_transcript(turns, "Interviewer")
        assert total == 0
        assert instances == []

    def test_multiple_bias_types_in_one_turn(self):
        # "Don't you think..." (leading) + "Did you feel frustrated..." (answer suggestion)
        turn = TranscriptTurn(0, "Interviewer", "Don't you think it was frustrating? Was it because of the design?")
        instances = detect_bias_in_turn(turn)
        types = {i.bias_type for i in instances}
        assert len(types) > 1

    def test_no_duplicate_bias_type_per_turn(self):
        # Each bias type should appear at most once per turn
        turn = TranscriptTurn(0, "Interviewer", "Don't you think... isn't it true... wouldn't you agree?")
        instances = detect_bias_in_turn(turn)
        types = [i.bias_type for i in instances]
        assert len(types) == len(set(types))


# ---------------------------------------------------------------------------
# BIAS_LABELS completeness
# ---------------------------------------------------------------------------


class TestBiasLabels:
    def test_all_types_have_labels(self):
        for bias_type in BiasType:
            assert bias_type in BIAS_LABELS

    def test_labels_are_strings(self):
        for label in BIAS_LABELS.values():
            assert isinstance(label, str)
            assert len(label) > 0
