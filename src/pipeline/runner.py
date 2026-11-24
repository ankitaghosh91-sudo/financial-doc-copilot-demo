"""One small injectable function that runs the copilot from input to response."""

from dataclasses import dataclass, field
import logging
from time import perf_counter
from typing import Any, Callable

from src.models import Document, Evidence
from src.strategies import (
    AnswerStrategyBase,
    AutoRAG,
    ConversationMemory,
    DatasetAnswer,
    EvaluatorBase,
    MemoryStrategyBase,
    PromptStrategyBase,
    RAGStrategyBase,
    SimpleEvaluator,
    SimplePrompt,
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "quota exceeded",
    "too many requests",
    "resource_exhausted",
)
_TOKEN_LENGTH_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "max_tokens",
    "token limit",
    "tokens exceed",
    "prompt is too long",
    "input length",
    "413",
)
_NETWORK_TLS_MARKERS = (
    "ssl",
    "certificate",
    "cert",
    "timeout",
    "timed out",
    "connectionerror",
    "connection refused",
    "name or service not known",
    "max retries exceeded",
)


def classify_pipeline_error(exc: Exception) -> str:
    """Categorizes exceptions into actionable operational buckets for logs and UI."""
    exc_type = type(exc).__name__.lower()
    msg = str(exc).lower()
    combined = f"{exc_type} {msg}"

    if "ratelimitexhausted" in exc_type or any(m in combined for m in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(m in combined for m in _TOKEN_LENGTH_MARKERS):
        return "token_or_context_length"
    if any(m in combined for m in _NETWORK_TLS_MARKERS):
        return "network_or_tls"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "validation_or_config"
    return "runtime_error"


@dataclass
class PipelineStrategies:
    """Swap one component at a time; the rest of the flow remains unchanged."""

    rag: RAGStrategyBase = field(default_factory=AutoRAG)
    answerer: AnswerStrategyBase = field(default_factory=DatasetAnswer)
    prompt: PromptStrategyBase = field(default_factory=SimplePrompt)
    memory: MemoryStrategyBase = field(default_factory=ConversationMemory)
    evaluator: EvaluatorBase = field(default_factory=SimpleEvaluator)
    agent_graph: Any = None
    request_validator: Callable[[str], None] | None = None
    response_validator: Callable[[dict, list[Evidence]], dict] | None = None
    router: Callable[[str, Any], str] | None = None


def validate_request(question: str) -> None:
    if not question or not question.strip():
        raise ValueError("Enter a question.")


def validate_response(answer: dict, evidence: list[Evidence]) -> dict:
    if not isinstance(answer, dict):
        raise ValueError("Answer strategy must return a dictionary.")
    if not isinstance(answer.get("answer"), str):
        raise ValueError("Answer strategy must return an answer string.")
    if not isinstance(answer.get("should_refuse"), bool):
        raise ValueError("Answer strategy must return a should_refuse boolean.")
    if not isinstance(answer.get("reason", ""), str):
        raise ValueError("Answer strategy refusal reason must be text.")
    valid_ids = {item.doc_id for item in evidence}
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or any(
        not isinstance(item, dict) or item.get("doc_id") not in valid_ids for item in citations
    ):
        return {
            "answer": "I cannot verify the answer citation.",
            "citations": [],
            "should_refuse": True,
            "reason": "Citation validation failed.",
        }
    return answer


def route_request(question: str, image: Any = None) -> str:
    return "visual" if image is not None else "text"


def _extract_graph_steps(state_out: dict[str, Any]) -> list[str]:
    trace = state_out.get("trace") or []
    node_names: list[str] = []
    for entry in trace:
        if isinstance(entry, str) and ":" in entry:
            node = entry.split(":", 1)[0].strip()
            if node and node not in node_names:
                node_names.append(node)
    return node_names or [
        "supervisor",
        "query_planning",
        "retrieval",
        "evidence_analysis",
        "answer_generation",
        "verification",
        "finalize",
    ]


def _reranked_to_evidence(reranked_docs: list[dict[str, Any]]) -> list[Evidence]:
    out: list[Evidence] = []
    for item in reranked_docs or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            Evidence(
                doc_id=str(item.get("doc_id") or ""),
                chunk_id=str(item.get("chunk_id") or ""),
                text=text,
                score=float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            )
        )
    return out


def _safe_memory_update(
    memory: MemoryStrategyBase,
    session_id: str,
    question: str,
    answer_text: str,
) -> None:
    try:
        memory.add(session_id, question, answer_text)
    except Exception as exc:
        logger.warning(
            "MEMORY_WRITE_FAILED | session_id=%s | error_type=%s | detail=%s",
            session_id,
            type(exc).__name__,
            exc,
        )


def _safe_evaluate(
    evaluator: EvaluatorBase,
    answer: dict[str, Any],
    evidence: list[Evidence],
    expected_answer: str | None,
) -> dict[str, Any]:
    try:
        return evaluator.evaluate(answer, evidence, expected_answer)
    except Exception as exc:
        logger.warning(
            "EVALUATOR_FAILED | evaluator=%s | error_type=%s | detail=%s",
            type(evaluator).__name__,
            type(exc).__name__,
            exc,
        )
        return {}


def run_pipeline(
    question: str,
    documents: list[Document],
    session_id: str = "default",
    expected_answer: str | None = None,
    image: Any = None,
    strategies: PipelineStrategies | None = None,
    scope: str = "global",
    doc_id: str | None = None,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run agent StateGraph when configured, otherwise linear pipeline."""
    choices = strategies or PipelineStrategies()
    validate = choices.request_validator or validate_request
    response_check = choices.response_validator or validate_response
    router = choices.router or route_request

    validate(question)
    route = router(question, image)
    q_clean = question.strip()
    est_q_tokens = max(1, (len(q_clean) + 3) // 4)
    total_doc_chars = sum(len(getattr(d, "content", "") or "") for d in (documents or []))

    logger.info(
        "PIPELINE_STARTED | session_id=%s | scope=%s | doc_id=%s | route=%s | q_chars=%d | est_q_tokens=%d | docs=%d | total_doc_chars=%d | has_image=%s",
        session_id,
        scope,
        doc_id,
        route,
        len(q_clean),
        est_q_tokens,
        len(documents or []),
        total_doc_chars,
        image is not None,
    )

    try:
        history = choices.memory.get(session_id)
    except Exception as exc:
        logger.warning(
            "MEMORY_READ_FAILED | session_id=%s | error_type=%s | detail=%s",
            session_id,
            type(exc).__name__,
            exc,
        )
        history = []

    if choices.agent_graph is not None:
        graph_started = perf_counter()
        invoke_kwargs: dict[str, Any] = {}
        if callbacks:
            invoke_kwargs["config"] = {"callbacks": callbacks}

        try:
            state_out = choices.agent_graph.invoke(
                {
                    "question": q_clean,
                    "documents": documents,
                },
                **invoke_kwargs,
            )
        except Exception as exc:
            err_cat = classify_pipeline_error(exc)
            logger.exception(
                "AGENT_GRAPH_FAILED | session_id=%s | category=%s | duration_ms=%.2f | q_chars=%d | est_q_tokens=%d | error_type=%s",
                session_id,
                err_cat,
                (perf_counter() - graph_started) * 1000,
                len(q_clean),
                est_q_tokens,
                type(exc).__name__,
            )
            raise

        final_output = state_out.get("final_output") or {}
        reranked_docs = state_out.get("reranked_docs") or []
        evidence = _reranked_to_evidence(reranked_docs)
        evidence_chars = sum(len(ev.text) for ev in evidence)

        raw_citations = final_output.get("citations") or state_out.get("citations") or []
        doc_by_chunk = {
            str(d.get("chunk_id") or ""): str(d.get("doc_id") or "")
            for d in reranked_docs
            if d.get("chunk_id")
        }
        normalized_citations = []
        for c in raw_citations:
            if isinstance(c, dict):
                cid = str(c.get("chunk_id", ""))
                did = str(c.get("doc_id") or doc_by_chunk.get(cid, ""))
                if did:
                    normalized_citations.append({"doc_id": did, "chunk_id": cid})
            elif isinstance(c, str):
                did = doc_by_chunk.get(c, "")
                if did:
                    normalized_citations.append({"doc_id": did, "chunk_id": c})

        answer = {
            "answer": str(final_output.get("answer") or state_out.get("draft_answer") or ""),
            "should_refuse": bool(
                final_output.get("should_refuse", state_out.get("should_refuse", False))
            ),
            "reason": str(
                final_output.get("refusal_reason") or state_out.get("refusal_reason") or ""
            ),
            "citations": normalized_citations,
        }
        logger.info(
            "AGENT_GRAPH_COMPLETED | session_id=%s | duration_ms=%.2f | evidence_count=%d | evidence_chars=%d | refused=%s | citations=%s",
            session_id,
            (perf_counter() - graph_started) * 1000,
            len(evidence),
            evidence_chars,
            answer["should_refuse"],
            answer["citations"],
        )
        metrics = _safe_evaluate(choices.evaluator, answer, evidence, expected_answer)
        answer = response_check(answer, evidence)
        _safe_memory_update(choices.memory, session_id, q_clean, answer.get("answer", ""))
        steps = ["validate request", f"route: {route}", *_extract_graph_steps(state_out)]

        return {
            "answer": answer,
            "route": route,
            "evidence": evidence,
            "history": history,
            "metrics": metrics,
            "steps": steps,
            "graph_state": state_out,
        }

    retrieval_started = perf_counter()
    try:
        evidence = choices.rag.retrieve(
            q_clean,
            documents,
            image,
            scope=scope,
            doc_id=doc_id,
        )
    except Exception as exc:
        err_cat = classify_pipeline_error(exc)
        logger.exception(
            "RETRIEVAL_FAILED | session_id=%s | category=%s | duration_ms=%.2f | q_chars=%d | error_type=%s",
            session_id,
            err_cat,
            (perf_counter() - retrieval_started) * 1000,
            len(q_clean),
            type(exc).__name__,
        )
        raise

    evidence_chars = sum(len(item.text or "") for item in (evidence or []))
    est_prompt_tokens = max(1, (len(q_clean) + evidence_chars + 3) // 4)
    logger.info(
        "RETRIEVAL_COMPLETED | session_id=%s | duration_ms=%.2f | count=%d | evidence_chars=%d | est_prompt_tokens=%d",
        session_id,
        (perf_counter() - retrieval_started) * 1000,
        len(evidence),
        evidence_chars,
        est_prompt_tokens,
    )
    logger.info(
        "ANSWER_GENERATION_STARTED | session_id=%s | strategy=%s | evidence_count=%d | est_prompt_tokens=%d",
        session_id,
        type(choices.answerer).__name__,
        len(evidence),
        est_prompt_tokens,
    )

    answer_started = perf_counter()
    try:
        answer = choices.answerer.answer(
            question=q_clean,
            evidence=evidence,
            prompt=choices.prompt,
            history=history,
            documents=documents,
        )
    except Exception as exc:
        err_cat = classify_pipeline_error(exc)
        logger.exception(
            "ANSWER_GENERATION_FAILED | session_id=%s | category=%s | strategy=%s | duration_ms=%.2f | evidence_chars=%d | est_prompt_tokens=%d | error_type=%s",
            session_id,
            err_cat,
            type(choices.answerer).__name__,
            (perf_counter() - answer_started) * 1000,
            evidence_chars,
            est_prompt_tokens,
            type(exc).__name__,
        )
        raise

    logger.info(
        "ANSWER_GENERATION_COMPLETED | session_id=%s | duration_ms=%.2f | refused=%s | citations=%s",
        session_id,
        (perf_counter() - answer_started) * 1000,
        answer.get("should_refuse"),
        answer.get("citations"),
    )

    metrics = _safe_evaluate(choices.evaluator, answer, evidence, expected_answer)
    answer = response_check(answer, evidence)
    _safe_memory_update(choices.memory, session_id, q_clean, answer.get("answer", ""))

    return {
        "answer": answer,
        "route": route,
        "evidence": evidence,
        "history": history,
        "metrics": metrics,
        "steps": [
            "validate request",
            f"route: {route}",
            "load memory",
            "search documents",
            "answer with prompt strategy",
            "evaluate",
            "validate response",
        ],
    }