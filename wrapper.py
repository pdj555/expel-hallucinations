"""A three-state output contract enforced around the Anthropic SDK.

Every assistant span is one of:
  Cited(text, source)     — entailed by a labeled source provided to the model
  Inference(text, conf)   — model judgment with explicit confidence in [0, 1]
  Abstention(text)        — model declines, with a brief reason

Anything outside these tags is a parse error. The wrapper retries the model
with a correction prompt up to `max_retries` times; on persistent failure it
downgrades the reply to an Abstention so the eval can score it as a contract
failure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Span types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cited:
    text: str
    source: str


@dataclass(frozen=True)
class Inference:
    text: str
    confidence: float  # in [0, 1]


@dataclass(frozen=True)
class Abstention:
    text: str


Span = Cited | Inference | Abstention


@dataclass(frozen=True)
class Source:
    id: str
    text: str


@dataclass
class CompletedResponse:
    spans: list[Span]
    raw_replies: list[str] = field(default_factory=list)
    retries_used: int = 0
    contract_failure: bool = False


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


# Matches [CITED:source]body[/CITED], [INF:0.83]body[/INF], [ABSTAIN]body[/ABSTAIN].
# `[ABSTAIN:reason]…[/ABSTAIN]` is also tolerated; the payload is ignored.
_SPAN_RE = re.compile(
    r"\[(CITED|INF|ABSTAIN)(?::([^\]]*))?\](.*?)\[/\1\]",
    re.DOTALL,
)


def parse_response(text: str, *, allowed_sources: Sequence[str] | None = None) -> list[Span]:
    """Parse a model reply into a list of typed spans.

    Raises ParseError on:
      - any non-whitespace text outside a recognized span
      - CITED span with empty source, or with source not in allowed_sources (if given)
      - INF span with non-numeric or out-of-range confidence
      - empty reply (no spans)
    """
    spans: list[Span] = []
    cursor = 0
    for m in _SPAN_RE.finditer(text):
        gap = text[cursor : m.start()]
        if gap.strip():
            raise ParseError(f"Unmarked text between spans: {gap.strip()!r}")
        kind = m.group(1)
        payload = (m.group(2) or "").strip()
        body = m.group(3).strip()
        if kind == "CITED":
            if not payload:
                raise ParseError(f"CITED span missing source id: {m.group(0)!r}")
            if allowed_sources is not None and payload not in allowed_sources:
                raise ParseError(
                    f"CITED span references unknown source {payload!r}; "
                    f"allowed: {sorted(allowed_sources)}"
                )
            spans.append(Cited(text=body, source=payload))
        elif kind == "INF":
            try:
                conf = float(payload)
            except (TypeError, ValueError) as e:
                raise ParseError(f"INF confidence not numeric: {payload!r}") from e
            if not 0.0 <= conf <= 1.0:
                raise ParseError(f"INF confidence out of [0, 1]: {conf}")
            spans.append(Inference(text=body, confidence=conf))
        else:  # ABSTAIN
            spans.append(Abstention(text=body))
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        raise ParseError(f"Unmarked text after spans: {tail.strip()!r}")
    if not spans:
        raise ParseError("No spans found in reply")
    return spans


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You answer questions under a strict output contract.

EVERY part of your reply must be one of three tagged spans:

[CITED:<source_id>]content[/CITED]
  Use when the content is directly entailed by a labeled source provided in the
  user message. <source_id> must exactly match one of the provided source IDs.

[INF:<confidence>]content[/INF]
  Use when the content is your inference (not directly cited).
  <confidence> is a number in [0, 1] reflecting your warranted confidence.
  Be calibrated: 0.9 means you'd be wrong about 10% of the time on similar claims.
  If you don't know, low confidence is honest.

[ABSTAIN]reason[/ABSTAIN]
  Use when you cannot answer reliably even with low-confidence inference.
  Briefly state why.

Rules:
- Every claim must be inside exactly one tag. No prose, headings, or commentary
  outside tags. Multiple spans per reply are fine; separate them with whitespace.
- Prefer [CITED:...] when a source supports the claim. Prefer [ABSTAIN] over a
  guess when no source applies and your warranted confidence is very low.
- Do not invent source IDs. Only cite IDs explicitly listed in the user message.

Examples
--------

Q: What does the function `f` do?
Sources:
- file.py: "def f(x): return x * 2"
Reply:
[CITED:file.py]`f` returns its argument multiplied by 2.[/CITED]

Q: Will this design scale to one billion users?
Sources: (none)
Reply:
[INF:0.2]Unlikely without sharding the database.[/INF] [ABSTAIN]No deployment details were given.[/ABSTAIN]

Q: When did the Roman Empire fall?
Sources: (none)
Reply:
[INF:0.7]Conventionally 476 AD for the Western Empire.[/INF] [INF:0.5]The Eastern (Byzantine) Empire continued until 1453.[/INF]
"""


CORRECTION_PROMPT = """Your previous reply did not satisfy the output contract.

Parse error: {error}

Reply to the original question again. EVERY part of your output must be wrapped in
exactly one of:
  [CITED:<source_id>]…[/CITED]
  [INF:<confidence>]…[/INF]
  [ABSTAIN]…[/ABSTAIN]

No prose outside tags. If unsure, use [ABSTAIN]…[/ABSTAIN].
"""


def render_user_message(question: str, sources: Sequence[Source] | None) -> str:
    parts = [f"Question: {question}", ""]
    if sources:
        parts.append("Sources:")
        for s in sources:
            parts.append(f"- [{s.id}] {s.text}")
    else:
        parts.append("Sources: (none)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """Wraps anthropic.Anthropic with the three-state contract.

    Usage:
        wrapped = Client()
        resp = wrapped.complete("What does f do?", sources=[Source("file.py", "def f...")])
        for span in resp.spans:
            ...
    """

    def __init__(
        self,
        *,
        client: Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 2,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.client = client or Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(
        self,
        question: str,
        *,
        sources: Sequence[Source] | None = None,
    ) -> CompletedResponse:
        allowed = [s.id for s in sources] if sources else None
        messages: list[MessageParam] = [
            {"role": "user", "content": render_user_message(question, sources)}
        ]
        raw_replies: list[str] = []
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            reply = self._call(messages)
            raw_replies.append(reply)
            try:
                spans = parse_response(reply, allowed_sources=allowed)
            except ParseError as e:
                last_error = str(e)
                if attempt == self.max_retries:
                    break
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {"role": "user", "content": CORRECTION_PROMPT.format(error=last_error)}
                )
                continue
            return CompletedResponse(
                spans=spans,
                raw_replies=raw_replies,
                retries_used=attempt,
            )

        downgrade = _downgrade(raw_replies[-1], last_error or "unknown parse failure")
        return CompletedResponse(
            spans=[downgrade],
            raw_replies=raw_replies,
            retries_used=self.max_retries,
            contract_failure=True,
        )

    def _call(self, messages: list[MessageParam]) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        chunks: list[str] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                chunks.append(block.text)
        return "".join(chunks)


def _downgrade(raw: str, error: str) -> Span:
    """Convert a final unparseable reply to a single Abstention span.

    The eval scores contract-failure abstentions separately so a model that
    routinely fails the contract is penalized rather than silently treated as
    confidently wrong.
    """
    snippet = raw.strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    return Abstention(text=f"contract failure ({error}); raw: {snippet}")
