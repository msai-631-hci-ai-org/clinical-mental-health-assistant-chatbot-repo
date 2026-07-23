"""Deterministic, model-independent safety routing."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyRoute:
    """A response that must bypass retrieval and generation."""

    category: str
    response: str


_SELF_HARM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:i\s+am|i['’]m|i)\s+(?:want|wish|plan|intend|going)\s+to\s+(?:die|kill myself|end my life|hurt myself|harm myself)\b",
        r"\bi\s+(?:have|made)\s+(?:a\s+)?plan\s+to\s+(?:die|kill myself|end my life)\b",
        r"\b(?:i[' ]?m|i am|i feel)\s+suicidal\b",
        r"\bi\s+(?:might|will|could)\s+(?:hurt|harm|kill)\s+myself\b",
        r"\bi\s+(?:cannot|can't|do not|don't)\s+keep\s+myself\s+safe\b",
        r"\bi\s+(?:have|made|wrote)\s+(?:a\s+)?suicide\s+(?:plan|note)\b",
        r"\bi\s+(?:just\s+)?(?:overdosed|cut myself|attempted suicide)\b",
        r"\bi\s+(?:do not|don't)\s+want\s+to\s+live(?:\s+anymore)?\b",
        r"\bi\s+wish\s+i\s+(?:was|were)\s+dead\b",
        r"\b(?:i\s+am|i['’]m|i)\s+(?:thinking|thought)\s+about\s+(?:suicide|killing myself|ending my life|hurting myself|harming myself)\b",
        r"\bi\s+have\s+suicidal\s+thoughts\b",
        r"\bam\s+i\s+suicidal\b",
    )
)

_HARM_OTHERS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:i\s+am|i['’]m|i)\s+(?:want|plan|intend|going)\s+to\s+(?:kill|seriously hurt|hurt|harm)\s+(?:him|her|them|someone|people)\b",
        r"\bi\s+(?:might|will)\s+(?:kill|seriously hurt|hurt|harm)\s+(?:him|her|them|someone|people)\b",
    )
)

_DIAGNOSIS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:can|could|would|will)\s+you\s+diagnos(?:e|is)\b",
        r"\bdo\s+i\s+have\s+(?:a\s+)?(?:mental\s+health\s+)?(?:condition|disorder|depression|anxiety|bipolar|ptsd|ocd|adhd)\b",
        r"\bam\s+i\s+(?:depressed|bipolar|mentally\s+ill)\b",
        r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+me\b",
        r"\b(?:can|could|would)\s+you\s+(?:tell|determine|figure\s+out)\b.{0,50}\b(?:if|whether)\s+i\s+have\s+(?:a\s+)?(?:mental\s+health\s+)?(?:condition|disorder|depression|anxiety|bipolar|ptsd|ocd|adhd)\b",
        r"\bdo\s+you\s+think\s+i\s+have\s+(?:a\s+)?(?:mental\s+health\s+)?(?:condition|disorder|depression|anxiety|bipolar|ptsd|ocd|adhd)\b",
    )
)

_MEDICATION_CHANGE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:should|can|could|may)\s+i\s+(?:stop|start|change|increase|decrease|reduce|switch|skip)\b.{0,60}\b(?:medication|medicine|meds|dose|antidepressant|prescription|pills?)\b",
        r"\b(?:should|can|could|may)\s+i\b.{0,30}\b(?:medication|medicine|meds|dose|antidepressant|prescription|pills?)\b.{0,30}\b(?:stop|start|change|increase|decrease|reduce|switch|skip)\b",
        r"\bis\s+it\s+safe\s+to\s+(?:stop|start|change|increase|decrease|reduce|switch|skip)\b.{0,60}\b(?:medication|medicine|meds|dose|antidepressant|prescription|pills?)\b",
        r"\b(?:what|which|how\s+much)\s+(?:dose|dosage|amount)\b.{0,40}\b(?:should|can|could)\s+i\s+(?:take|use)\b",
    )
)

_CRISIS_RESPONSE = """
I’m glad you said something. I’m not able to provide emergency support, and
your immediate safety matters more than continuing this chat.

- If you may act now, have already harmed yourself, or are in immediate danger,
  contact local emergency services or go to the nearest emergency department.
- In the U.S. and its territories, call or text **988** or use
  [988lifeline.org](https://988lifeline.org/) to reach the Suicide & Crisis
  Lifeline. If you are elsewhere, use your local emergency or crisis service.
- If it is safe to do so, move away from anything you could use to hurt
  yourself and contact a trusted person who can stay with you.

Can you contact emergency support or a trusted person right now?
""".strip()

_VIOLENCE_RESPONSE = """
I can’t help with harming someone. If anyone may be in immediate danger,
contact local emergency services now and create physical distance from weapons
or the person involved. If you can do so safely, reach out to a trusted person
or qualified crisis professional who can stay with you while the immediate risk
passes.
""".strip()

_DIAGNOSIS_RESPONSE = """
I can’t diagnose a mental health condition or determine whether you meet
diagnostic criteria. A qualified health professional can consider your
symptoms, how long they have lasted, how they affect daily life, medications,
and possible medical causes. I can still share general, source-grounded
information or help you prepare questions for an appointment.
""".strip()

_MEDICATION_CHANGE_RESPONSE = """
I can’t advise you to start, stop, skip, or change a medication or dose. Contact
the prescribing clinician or a pharmacist for guidance based on the specific
medicine and your circumstances. If you may have taken too much, are having a
severe reaction, or are in immediate danger, contact local emergency medical
services now.
""".strip()


def route_for_safety(message: str) -> SafetyRoute | None:
    """Return a crisis route for direct first-person danger disclosures."""

    normalized = " ".join(message.split())
    if any(pattern.search(normalized) for pattern in _SELF_HARM_PATTERNS):
        return SafetyRoute(category="self_harm", response=_CRISIS_RESPONSE)
    if any(pattern.search(normalized) for pattern in _HARM_OTHERS_PATTERNS):
        return SafetyRoute(category="harm_others", response=_VIOLENCE_RESPONSE)
    if any(pattern.search(normalized) for pattern in _DIAGNOSIS_PATTERNS):
        return SafetyRoute(category="diagnosis_boundary", response=_DIAGNOSIS_RESPONSE)
    if any(pattern.search(normalized) for pattern in _MEDICATION_CHANGE_PATTERNS):
        return SafetyRoute(
            category="medication_boundary",
            response=_MEDICATION_CHANGE_RESPONSE,
        )
    return None
