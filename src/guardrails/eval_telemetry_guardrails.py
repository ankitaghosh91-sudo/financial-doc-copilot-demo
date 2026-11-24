import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class GuardrailEvent:
    stage: str          # "input" | "context" | "output"
    rule: str           # e.g., "prompt_injection", "empty_context", "numeric_hallucination"
    action: str         # "block" | "abstain" | "flag"
    detail: str = ""


@dataclass
class QueryTelemetry:
    query_id: str
    question: str
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    max_tokens_configured: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_source: str = "unknown"  # "provider" | "estimated" | "none"
    estimated_cost_usd: float = 0.0
    guardrail_events: List[GuardrailEvent] = field(default_factory=list)
    abstained: bool = False
    blocked: bool = False


class StageTimer:
    def __init__(self, store: Dict[str, float], key: str):
        self.store = store
        self.key = key
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store[self.key] = (time.perf_counter() - self.start) * 1000.0


class GuardrailEngine:
    """Lightweight pre-retrieval, context-sufficiency, and post-generation checks."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
        r"act\s+as\s+system\s+administrator",
        r"dump\s+(all\s+)?(documents|database|api\s*keys)",
    ]

    ABSTENTION_PHRASES = [
        "not mentioned in the provided context",
        "cannot be determined from the provided context",
        "insufficient information in the context",
        "not found in the provided documents",
    ]

    def __init__(self, max_query_chars: int = 2000, min_context_chars: int = 40):
        self.max_query_chars = max_query_chars
        self.min_context_chars = min_context_chars
        self._injection_re = re.compile("|".join(self.INJECTION_PATTERNS), re.IGNORECASE)

    def check_input(self, question: str) -> List[GuardrailEvent]:
        events: List[GuardrailEvent] = []
        q = (question or "").strip()

        if not q:
            events.append(GuardrailEvent("input", "empty_query", "block", "Query is empty."))
            return events

        if len(q) > self.max_query_chars:
            events.append(
                GuardrailEvent(
                    "input",
                    "query_too_long",
                    "flag",
                    f"Length {len(q)} exceeds {self.max_query_chars} chars.",
                )
            )

        if self._injection_re.search(q):
            events.append(
                GuardrailEvent(
                    "input",
                    "prompt_injection",
                    "block",
                    "Matched prompt-injection pattern.",
                )
            )

        return events

    def check_context(self, contexts: List[str]) -> List[GuardrailEvent]:
        events: List[GuardrailEvent] = []
        combined = "\n".join(c for c in (contexts or []) if c).strip()

        if not contexts or len(combined) < self.min_context_chars:
            events.append(
                GuardrailEvent(
                    "context",
                    "insufficient_context",
                    "abstain",
                    f"Combined context length={len(combined)} < {self.min_context_chars}.",
                )
            )

        return events

    def check_output(self, answer: str, contexts: List[str]) -> Tuple[List[GuardrailEvent], bool]:
        events: List[GuardrailEvent] = []
        ans = (answer or "").strip()
        ans_lower = ans.lower()
        ctx_text = "\n".join(contexts or [])

        abstained = any(phrase in ans_lower for phrase in self.ABSTENTION_PHRASES)
        if abstained:
            events.append(
                GuardrailEvent("output", "model_abstained", "flag", "Answer indicates missing context.")
            )
            return events, True

        # Deterministic check: flag dollar/currency figures in answer that do not appear in retrieved contexts
        ans_amounts = set(re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?", ans))
        if ans_amounts:
            ctx_normalized = re.sub(r"\s+", "", ctx_text)
            ungrounded = [
                amt for amt in ans_amounts if re.sub(r"\s+", "", amt) not in ctx_normalized
            ]
            if ungrounded:
                events.append(
                    GuardrailEvent(
                        "output",
                        "ungrounded_currency_figure",
                        "flag",
                        f"Amounts not in context: {sorted(ungrounded)}",
                    )
                )

        return events, False


def extract_tokens_and_cost(
    llm_response: Any,
    question: str,
    contexts: List[str],
    answer: str,
    input_cost_per_1m_usd: float = 0.0,
    output_cost_per_1m_usd: float = 0.0,
) -> Tuple[int, int, int, str, float]:
    """Extracts token counts from common LLM response shapes; falls back to ~4 chars/token."""
    prompt_tokens = 0
    completion_tokens = 0
    source = "none"

    usage = getattr(llm_response, "usage", None) or getattr(llm_response, "usage_metadata", None)
    if isinstance(llm_response, dict):
        usage = usage or llm_response.get("usage") or llm_response.get("usage_metadata")

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if prompt_tokens or completion_tokens:
            source = "provider"
    elif usage is not None:
        prompt_tokens = int(
            getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        )
        if prompt_tokens or completion_tokens:
            source = "provider"

    if source == "none":
        est_input_chars = len(question or "") + sum(len(c or "") for c in (contexts or []))
        est_output_chars = len(answer or "")
        prompt_tokens = max(1, math.ceil(est_input_chars / 4)) if est_input_chars else 0
        completion_tokens = max(1, math.ceil(est_output_chars / 4)) if est_output_chars else 0
        source = "estimated"

    total_tokens = prompt_tokens + completion_tokens
    cost_usd = (prompt_tokens / 1_000_000.0) * input_cost_per_1m_usd + (
        completion_tokens / 1_000_000.0
    ) * output_cost_per_1m_usd
    return prompt_tokens, completion_tokens, total_tokens, source, cost_usd


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * (p / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(s[lo], 2)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)


def summarize_telemetry(records: List[QueryTelemetry]) -> Dict[str, Any]:
    totals = [r.latencies_ms.get("total_ms", 0.0) for r in records]
    retrieval_ms = [r.latencies_ms.get("retrieval_ms", 0.0) for r in records]
    rerank_ms = [r.latencies_ms.get("rerank_ms", 0.0) for r in records]
    generation_ms = [r.latencies_ms.get("generation_ms", 0.0) for r in records]
    max_tok_vals = [r.max_tokens_configured for r in records if r.max_tokens_configured > 0]

    rule_counts: Dict[str, int] = {}
    for r in records:
        for ev in r.guardrail_events:
            rule_counts[ev.rule] = rule_counts.get(ev.rule, 0) + 1

    total_cost = sum(r.estimated_cost_usd for r in records)
    n = len(records) or 1

    return {
        "num_queries": len(records),
        "latency_ms": {
            "total": {"mean": round(sum(totals) / n, 2), "p50": percentile(totals, 50), "p95": percentile(totals, 95)},
            "retrieval": {"mean": round(sum(retrieval_ms) / n, 2), "p95": percentile(retrieval_ms, 95)},
            "rerank": {"mean": round(sum(rerank_ms) / n, 2), "p95": percentile(rerank_ms, 95)},
            "generation": {"mean": round(sum(generation_ms) / n, 2), "p95": percentile(generation_ms, 95)},
        },
        "tokens": {
            "max_tokens_per_call": max_tok_vals[0] if max_tok_vals else 0,
            "prompt_total": sum(r.prompt_tokens for r in records),
            "completion_total": sum(r.completion_tokens for r in records),
            "total": sum(r.total_tokens for r in records),
            "avg_per_query": round(sum(r.total_tokens for r in records) / n, 1),
        },
        "cost_usd": {
            "total": round(total_cost, 6),
            "avg_per_query": round(total_cost / n, 6),
        },
        "guardrails": {
            "blocked_queries": sum(1 for r in records if r.blocked),
            "abstained_queries": sum(1 for r in records if r.abstained),
            "total_events": sum(len(r.guardrail_events) for r in records),
            "by_rule": rule_counts,
        },
    }


def run_monitored_query(
    query_id: str,
    question: str,
    retrieve_fn: Callable[[str], List[str]],
    rerank_fn: Optional[Callable[[str, List[str]], List[str]]],
    generate_fn: Callable[[str, List[str]], Any],
    extract_answer_fn: Callable[[Any], str],
    guardrails: GuardrailEngine,
    input_cost_per_1m_usd: float = 0.0,
    output_cost_per_1m_usd: float = 0.0,
    max_tokens_configured: int = 0,
) -> Tuple[str, List[str], QueryTelemetry]:
    tel = QueryTelemetry(
        query_id=query_id,
        question=question,
        max_tokens_configured=max_tokens_configured,
    )
    t0 = time.perf_counter()

    with StageTimer(tel.latencies_ms, "guardrail_input_ms"):
        in_events = guardrails.check_input(question)
        tel.guardrail_events.extend(in_events)
        if any(e.action == "block" for e in in_events):
            tel.blocked = True
            tel.latencies_ms["total_ms"] = (time.perf_counter() - t0) * 1000.0
            return "Request blocked by input guardrail.", [], tel

    with StageTimer(tel.latencies_ms, "retrieval_ms"):
        candidates = retrieve_fn(question)

    with StageTimer(tel.latencies_ms, "rerank_ms"):
        contexts = rerank_fn(question, candidates) if rerank_fn else candidates

    with StageTimer(tel.latencies_ms, "guardrail_context_ms"):
        ctx_events = guardrails.check_context(contexts)
        tel.guardrail_events.extend(ctx_events)
        if any(e.action == "abstain" for e in ctx_events):
            tel.abstained = True
            tel.latencies_ms["total_ms"] = (time.perf_counter() - t0) * 1000.0
            return "Insufficient information in the retrieved context to answer accurately.", contexts, tel

    with StageTimer(tel.latencies_ms, "generation_ms"):
        raw_resp = generate_fn(question, contexts)
        answer = extract_answer_fn(raw_resp)

    with StageTimer(tel.latencies_ms, "guardrail_output_ms"):
        out_events, abstained = guardrails.check_output(answer, contexts)
        tel.guardrail_events.extend(out_events)
        if abstained:
            tel.abstained = True

    p_tok, c_tok, tot_tok, src, cost = extract_tokens_and_cost(
        raw_resp,
        question=question,
        contexts=contexts,
        answer=answer,
        input_cost_per_1m_usd=input_cost_per_1m_usd,
        output_cost_per_1m_usd=output_cost_per_1m_usd,
    )
    tel.prompt_tokens = p_tok
    tel.completion_tokens = c_tok
    tel.total_tokens = tot_tok
    tel.token_source = src
    tel.estimated_cost_usd = cost
    tel.latencies_ms["total_ms"] = (time.perf_counter() - t0) * 1000.0

    return answer, contexts, tel