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
    return node_names or ["supervisor", "query_planning", "retrieval", "evidence_analysis", "answer_generation", "verification", "finalize"]


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


def run_pipeline(
    question: str,
    documents: list[Document],
    session_id: str = "default",
    expected_answer: str | None = None,
    image: Any = None,
    strategies: PipelineStrategies | None = None,
    scope: str = "global",
    doc_id: str | None = None,
) -> dict:
    """Run agent StateGraph when configured, otherwise linear pipeline."""
    choices = strategies or PipelineStrategies()
    validate = choices.request_validator or validate_request
    response_check = choices.response_validator or validate_response
    router = choices.router or route_request

    validate(question)
    logger.info(
        "PIPELINE_STARTED | session_id=%s | scope=%s | doc_id=%s",
        session_id,
        scope,
        doc_id,
    )
    route = router(question, image)
    history = choices.memory.get(session_id)

    if choices.agent_graph is not None:
        graph_started = perf_counter()
        state_out = choices.agent_graph.invoke(
            {
                "question": question,
                "documents": documents,
            }
        )
        final_output = state_out.get("final_output") or {}
        reranked_docs = state_out.get("reranked_docs") or []
        evidence = _reranked_to_evidence(reranked_docs)

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
            "AGENT_GRAPH_COMPLETED | duration_ms=%.2f | refused=%s | citations=%s",
            (perf_counter() - graph_started) * 1000,
            answer["should_refuse"],
            answer["citations"],
        )
        metrics = choices.evaluator.evaluate(answer, evidence, expected_answer)
        answer = response_check(answer, evidence)
        choices.memory.add(session_id, question, answer["answer"])
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

    evidence = choices.rag.retrieve(
        question,
        documents,
        image,
        scope=scope,
        doc_id=doc_id,
    )
    logger.info(
        "RETRIEVAL_COMPLETED | count=%d | chunks=%s",
        len(evidence),
        [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "score": round(item.score, 4),
            }
            for item in evidence
        ],
    )
    logger.info(
        "ANSWER_GENERATION_STARTED | strategy=%s | evidence_count=%d",
        type(choices.answerer).__name__,
        len(evidence),
    )

    answer_started = perf_counter()

    try:
        answer = choices.answerer.answer(
            question=question,
            evidence=evidence,
            prompt=choices.prompt,
            history=history,
            documents=documents,
        )
    except Exception:
        logger.exception("ANSWER_GENERATION_FAILED")
        raise

    logger.info(
        "ANSWER_GENERATION_COMPLETED | duration_ms=%.2f | refused=%s | citations=%s",
        (perf_counter() - answer_started) * 1000,
        answer.get("should_refuse"),
        answer.get("citations"),
    )

    metrics = choices.evaluator.evaluate(answer, evidence, expected_answer)
    answer = response_check(answer, evidence)
    choices.memory.add(session_id, question, answer["answer"])

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