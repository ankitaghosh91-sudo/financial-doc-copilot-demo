import re
from collections.abc import Callable
from typing import Any, Literal

from src.agents.state import AgentRAGState

VerifyFn = Callable[[AgentRAGState], tuple[bool, str, list[str], list[str]]]

CURRENCY_PATTERN = re.compile(
    r"(?:\$|usd|eur|gbp|inr|₹)\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\.\d{2}\b",
    re.IGNORECASE,
)

MIN_GROUNDING_OVERLAP = 0.22


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9_]{3,}", text or "")}


def _default_verify_fn(
    state: AgentRAGState,
) -> tuple[bool, str, list[str], list[str]]:
    draft = (state.get("draft_answer") or "").strip()
    should_refuse = bool(state.get("should_refuse", False))
    refusal_reason = (state.get("refusal_reason") or "").strip()
    reranked_docs = state.get("reranked_docs") or []
    raw_citations = state.get("citations") or []
    evidence_sufficient = bool(state.get("evidence_sufficient", False))
    existing_gaps = list(state.get("evidence_gaps") or [])

    valid_chunk_ids = {
        str(d.get("chunk_id"))
        for d in reranked_docs
        if d.get("chunk_id")
    }
    valid_citations = [
        str(cid) for cid in raw_citations if str(cid) in valid_chunk_ids
    ]
    invalid_citations = [
        str(cid) for cid in raw_citations if str(cid) not in valid_chunk_ids
    ]

    top_rerank_score = 0.0
    if reranked_docs:
        first_doc = reranked_docs[0]
        top_rerank_score = float(first_doc.get("rerank_score", first_doc.get("score", 0.0)))

    top5_chunk_ids = {
        str(d.get("chunk_id"))
        for d in reranked_docs[:5]
        if d.get("chunk_id")
    }

    issues: list[str] = []

    if should_refuse:
        issues.append(
            f"Answer refused due to insufficient evidence: {refusal_reason or 'unspecified'}"
        )
    elif not draft:
        issues.append("Draft answer is empty.")

    if not should_refuse and not valid_citations:
        issues.append("No valid citations matching reranked chunk IDs.")

    if invalid_citations:
        issues.append(f"Dropped invalid citations: {invalid_citations}")

    cited_docs = [d for d in reranked_docs if str(d.get("chunk_id")) in set(valid_citations)]
    cited_text = " ".join(
        (d.get("text") or "") for d in cited_docs if (d.get("text") or "").strip()
    )
    fallback_text = " ".join(
        (d.get("text") or "") for d in reranked_docs if (d.get("text") or "").strip()
    )

    ans_tokens = _tokens(draft)
    cited_tokens = _tokens(cited_text)
    fallback_tokens = _tokens(fallback_text)

    grounding_ok = True
    if not should_refuse and ans_tokens:
        if top_rerank_score < 0.0:
            has_top5_citation = any(cid in top5_chunk_ids for cid in valid_citations)
            if not has_top5_citation:
                grounding_ok = False
                issues.append(
                    "No cited chunk appeared in the top 5 reranked chunks for a negative reranker score."
                )

            if cited_tokens:
                cited_overlap = len(ans_tokens & cited_tokens) / max(1, len(ans_tokens))
                if cited_overlap < MIN_GROUNDING_OVERLAP:
                    grounding_ok = False
                    issues.append(
                        f"Low grounding overlap ({cited_overlap:.2f}) between draft answer and cited chunk text."
                    )
            else:
                grounding_ok = False
                issues.append("Draft answer has no cited chunk text to verify grounding.")
        elif fallback_tokens:
            overlap = len(ans_tokens & fallback_tokens) / max(1, len(ans_tokens))
            if overlap < MIN_GROUNDING_OVERLAP:
                grounding_ok = False
                issues.append(
                    f"Low factual grounding overlap ({overlap:.2f}) between draft answer and retrieved evidence."
                )

    if not evidence_sufficient and (not valid_citations or not grounding_ok):
        issues.append("Upstream evidence analysis marked evidence as insufficient.")

    updated_gaps = list(dict.fromkeys([*existing_gaps, *issues]))
    verified = (not should_refuse) and len(issues) == 0
    notes = (
        f"Verified with {len(valid_citations)} valid citation(s)."
        if verified
        else "; ".join(issues)
    )
    return verified, notes, valid_citations, updated_gaps


def make_verification_node(verify_fn: VerifyFn = _default_verify_fn):
    def verification_node(state: AgentRAGState) -> dict[str, Any]:
        trace = list(state.get("trace", []))
        verified, notes, valid_citations, updated_gaps = verify_fn(state)

        attempt = int(state.get("attempt", 0) or 0)
        max_attempts = int(state.get("max_attempts", 2) or 2)
        should_refuse = bool(state.get("should_refuse", False))
        refusal_reason = str(state.get("refusal_reason") or "")
        draft = (state.get("draft_answer") or "").strip()

        reranked_docs = state.get("reranked_docs") or []
        cited_docs = [d for d in reranked_docs if str(d.get("chunk_id")) in set(valid_citations)]
        evidence_text = " ".join(
            (d.get("text") or "")
            for d in (cited_docs or reranked_docs)
            if (d.get("text") or "").strip()
        )

        draft_amounts = {
            m.group(0).lower().lstrip("$").strip()
            for m in CURRENCY_PATTERN.finditer(draft)
        }
        ev_amounts = {
            m.group(0).lower().lstrip("$").strip()
            for m in CURRENCY_PATTERN.finditer(evidence_text)
        }
        ungrounded_amounts = sorted(amt for amt in draft_amounts if amt not in ev_amounts)
        if ungrounded_amounts and not should_refuse:
            verified = False
            notes = f"{notes}; ungrounded_currency_figure: {ungrounded_amounts}".lstrip("; ")

        # If max attempts are reached and verification still fails, enforce clean refusal
        if not verified and attempt >= max_attempts:
            should_refuse = True
            refusal_reason = (
                "The indexed documents don't contain enough verified information to answer this question. "
                "Try rephrasing with more specific terms or details."
            )
            valid_citations = []
            draft = refusal_reason

        trace.append(
            f"verification: verified={verified} should_refuse={should_refuse} valid_citations={len(valid_citations)} gaps={len(updated_gaps)}"
        )
        return {
            "verified": verified,
            "verification_notes": notes,
            "citations": valid_citations,
            "citation_pairs": [] if should_refuse else state.get("citation_pairs", []),
            "should_refuse": should_refuse,
            "refusal_reason": refusal_reason,
            "draft_answer": draft,
            "evidence_gaps": updated_gaps,
            "trace": trace,
        }

    return verification_node


def route_after_verification(
    state: AgentRAGState,
) -> Literal["supervisor", "finalize"]:
    if state.get("verified", False):
        return "finalize"
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 2)
    if attempt < max_attempts:
        return "supervisor"
    return "finalize"


verification_node = make_verification_node()