"""Module 4b: plain-English reasons for flagged rows, and the token/cost log.

The verdict and matched_note_id are decided before this module runs (rag_engine.py). This
module only words the `reason` column. Two ways to do that:

- template (default): fixed sentences filled from the row's numbers and the note text.
  Zero LLM calls, zero cost, byte-identical output on every run.
- LLM (optional, --llm ollama | anthropic): one call per flagged row at temperature 0. The
  model only sees the row's facts and the note text that rag_engine already selected. Its
  answer is checked (right verdict word, right note id, no other note ids, no invented
  percentages) and replaced by the template if any check fails, so the model can change
  wording but never the verdict, the cited note, or a number.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date

from rag_engine import MatchResult, NoteRecord

NOTE_ID_PATTERN = re.compile(r"\bN\d{3}\b")
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


@dataclass(frozen=True)
class RowFacts:
    route: str
    week_of: date
    cost_per_tonne_km: float
    pct_vs_own_history: float
    pct_vs_peers: float
    own_history_weeks: int
    flag_basis: str
    match: MatchResult

    @property
    def key(self) -> str:
        return f"{self.route} {self.week_of.isoformat()}"


# --------------------------------------------------------------------------- template

def _describe(note: NoteRecord) -> str:
    return f"{note.note_id} ({note.date.isoformat()})"


def template_reason(facts: RowFacts) -> str:
    match = facts.match
    if match.verdict == "justified":
        note = match.note
        return (
            f"Justified: matches note {note.note_id} dated {note.date.isoformat()} "
            f"({note.applies_to}, in force {note.window_label()}): {note.text} "
            f"The week of {facts.week_of.isoformat()} falls inside that window, so the rise has a documented cause."
        )
    if match.nearest_rejected:
        parts = [f"{_describe(n)} {match.rejected[n.note_id]}" for n in match.nearest_rejected]
        if len(parts) == 1:
            n = match.nearest_rejected[0]
            closest = f"the closest note, {_describe(n)}, {match.rejected[n.note_id]}"
        else:
            closest = "the closest notes do not count: " + "; ".join(parts)
        return (
            f"Unexplained: {closest}. No note describes a reason for this rise on {facts.route}; "
            f"flagged for human review."
        )
    return (
        f"Unexplained: no note covers {facts.route} for the week of {facts.week_of.isoformat()}. "
        f"The cost rise has no documented cause and is worth a human review."
    )


# --------------------------------------------------------------------------- LLM prompt + checks

SYSTEM_PROMPT = (
    "You write one or two plain-English sentences explaining a freight cost flag. "
    "Use only the facts given. The verdict and the cited note are already decided: do not change them, "
    "do not cite any other note, and do not add numbers, dates or causes that are not in the facts. "
    "Start with the verdict word given (Justified or Unexplained)."
)


def build_prompt(facts: RowFacts) -> str:
    match = facts.match
    lines = [
        f"Route: {facts.route}",
        f"Week of (Monday): {facts.week_of.isoformat()}",
        f"Cost per tonne-km: {facts.cost_per_tonne_km:.2f}",
        f"Change vs this route's own trailing average: {facts.pct_vs_own_history:+.1f}%",
        f"Change vs similar-length routes this week: {facts.pct_vs_peers:+.1f}%",
        f"Flagged because it is unusual vs: {facts.flag_basis}",
    ]
    if match.verdict == "justified":
        note = match.note
        lines += [
            "Verdict: Justified",
            f"Cited note: {note.note_id}, dated {note.date.isoformat()}, applies to {note.applies_to}, "
            f"in force {note.window_label()}",
            f"Note text: {note.text}",
            "Say which note explains it and reflect its content and dates.",
        ]
    else:
        lines.append("Verdict: Unexplained (no note supports this rise)")
        for note in match.nearest_rejected:
            lines.append(f"Nearby note that does NOT count: {note.note_id} ({note.date.isoformat()}): "
                         f"{note.text} Reason it does not count: {match.rejected[note.note_id]}")
        lines.append("Say that no supporting note was found and that it needs human review.")
    return "\n".join(lines)


def check_reason(text: str, facts: RowFacts) -> list[str]:
    """Problems with an LLM-written reason; an empty list means it is safe to use."""
    problems = []
    match = facts.match
    stripped = text.strip()
    if not stripped or len(stripped) > 600:
        problems.append("empty or too long")
    if match.verdict == "justified":
        if not stripped.lower().startswith("justified"):
            problems.append("does not start with the verdict 'Justified'")
        if match.matched_note_id not in stripped:
            problems.append(f"does not cite {match.matched_note_id}")
        allowed_ids = {match.matched_note_id}
        context_text = match.note.text
    else:
        if not stripped.lower().startswith("unexplained"):
            problems.append("does not start with the verdict 'Unexplained'")
        allowed_ids = {n.note_id for n in match.nearest_rejected}
        context_text = " ".join(n.text for n in match.nearest_rejected)
    extra_ids = set(NOTE_ID_PATTERN.findall(stripped)) - allowed_ids
    if extra_ids:
        problems.append(f"mentions notes outside the facts: {sorted(extra_ids)}")
    allowed_pcts = {f"{abs(facts.pct_vs_own_history):.1f}", f"{abs(facts.pct_vs_peers):.1f}"}
    allowed_pcts |= set(re.findall(r"\d+(?:\.\d+)?", context_text))
    invented = [p for p in PERCENT_PATTERN.findall(stripped) if p not in allowed_pcts and f"{float(p):.1f}" not in allowed_pcts]
    if invented:
        problems.append(f"percentages not in the facts: {invented}")
    return problems


# --------------------------------------------------------------------------- usage log

@dataclass
class UsageLog:
    provider: str
    model: str | None
    temperature: float | None
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    calls: list[dict] = field(default_factory=list)
    failed_calls: list[dict] = field(default_factory=list)

    def record(self, row: str, input_tokens: int, output_tokens: int, used: bool, note: str = "") -> None:
        self.calls.append({"row": row, "input_tokens": input_tokens, "output_tokens": output_tokens,
                           "llm_text_used": used, "note": note})

    def record_failure(self, row: str, error: str) -> None:
        self.failed_calls.append({"row": row, "error": error})

    def summary(self) -> dict:
        total_in = sum(c["input_tokens"] for c in self.calls)
        total_out = sum(c["output_tokens"] for c in self.calls)
        cost = total_in / 1e6 * self.input_usd_per_mtok + total_out / 1e6 * self.output_usd_per_mtok
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "llm_calls": len(self.calls),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "rate_usd_per_million_tokens": {"input": self.input_usd_per_mtok, "output": self.output_usd_per_mtok},
            "estimated_cost_usd": round(cost, 6),
            "calls_where_llm_text_failed_checks": sum(not c["llm_text_used"] for c in self.calls),
            "failed_requests": len(self.failed_calls),
        }


# --------------------------------------------------------------------------- writers

class TemplateWriter:
    provider, model, temperature = "none (template)", None, None
    input_usd_per_mtok = output_usd_per_mtok = 0.0

    def complete(self, system: str, prompt: str) -> tuple[str, int, int]:
        raise NotImplementedError("TemplateWriter makes no LLM calls")


class OllamaWriter:
    """Open-source model served locally by Ollama (free). Temperature 0 and a fixed seed."""

    provider, temperature = "ollama", 0.0
    input_usd_per_mtok = output_usd_per_mtok = 0.0

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, system: str, prompt: str) -> tuple[str, int, int]:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
        return data["message"]["content"], int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))


class AnthropicWriter:
    """Hosted Claude model through the official SDK.

    claude-haiku-4-5 is used because it is low cost and still accepts temperature=0; the newer
    Claude models reject sampling parameters, and the brief asks for temperature 0.
    """

    provider, temperature = "anthropic", 0.0
    input_usd_per_mtok, output_usd_per_mtok = 1.00, 5.00  # published rate for claude-haiku-4-5

    def __init__(self, model: str | None = None):
        try:
            import anthropic  # optional dependency, only needed for this writer
        except ImportError as exc:
            raise RuntimeError("--llm anthropic needs the SDK: pip install anthropic") from exc

        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self.client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            # SDK 1.x dropped the temperature keyword; the API still honours it for this model.
            extra_body={"temperature": 0},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens


def make_writer(provider: str):
    if provider == "none":
        return TemplateWriter()
    if provider == "ollama":
        return OllamaWriter()
    if provider == "anthropic":
        return AnthropicWriter()
    raise ValueError(f"Unknown LLM provider {provider!r}; choose none, ollama or anthropic")


class ReasonWriter:
    def __init__(self, provider: str = "none"):
        self.writer = make_writer(provider)
        self.log = UsageLog(self.writer.provider, self.writer.model, self.writer.temperature,
                            self.writer.input_usd_per_mtok, self.writer.output_usd_per_mtok)
        self.unavailable: str | None = None  # set after the first failed request; later rows skip the LLM

    def reason(self, facts: RowFacts) -> str:
        fallback = template_reason(facts)
        if isinstance(self.writer, TemplateWriter) or self.unavailable:
            return fallback
        try:
            text, tokens_in, tokens_out = self.writer.complete(SYSTEM_PROMPT, build_prompt(facts))
        except Exception as exc:  # network, auth or model errors must never stop the run
            self.unavailable = f"{type(exc).__name__}: {exc}"
            self.log.record_failure(facts.key, self.unavailable)
            return fallback
        text = " ".join(text.split())
        problems = check_reason(text, facts)
        self.log.record(facts.key, tokens_in, tokens_out, used=not problems, note="; ".join(problems))
        return fallback if problems else text


def estimate_llm_usage(all_facts: list[RowFacts], output_tokens_per_call: int = 80) -> dict:
    """Rough size of an LLM run over the same rows (about 4 characters per token)."""
    input_tokens = sum((len(SYSTEM_PROMPT) + len(build_prompt(f))) // 4 for f in all_facts)
    output_tokens = output_tokens_per_call * len(all_facts)
    rate = AnthropicWriter.input_usd_per_mtok, AnthropicWriter.output_usd_per_mtok
    return {
        "method": "estimate: characters / 4 for input, 80 output tokens per call",
        "llm_calls": len(all_facts),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "cost_usd_with_ollama_open_source": 0.0,
        "cost_usd_with_claude_haiku_4_5": round(input_tokens / 1e6 * rate[0] + output_tokens / 1e6 * rate[1], 6),
    }


def usage_report(writer: ReasonWriter, all_facts: list[RowFacts]) -> dict:
    report = {"this_run": writer.log.summary(), "per_call": writer.log.calls,
              "failed_requests": writer.log.failed_calls}
    report["if_llm_wording_is_enabled"] = estimate_llm_usage(all_facts)
    return json.loads(json.dumps(report, default=str))

