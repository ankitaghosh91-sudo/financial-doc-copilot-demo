from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import types
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _ensure_ragas_vertexai_shim() -> None:
  """Prevents ragas import crash when langchain_community removes chat_models.vertexai."""
  mod_name = "langchain_community.chat_models.vertexai"
  if mod_name not in sys.modules:
    try:
      __import__(mod_name)
    except ModuleNotFoundError:
      stub = types.ModuleType(mod_name)
      stub.ChatVertexAI = type("ChatVertexAI", (object,), {})
      sys.modules[mod_name] = stub


# Must run before importing ragas so ragas.llms.base does not fail on missing vertexai
_ensure_ragas_vertexai_shim()

from datasets import Dataset  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from src.config import AppConfig  # noqa: E402
from src.evaluation.llm_client import LLMClient  # noqa: E402
from src.guardrails.eval_telemetry_guardrails import (  # noqa: E402
    GuardrailEngine,
    QueryTelemetry,
    StageTimer,
    extract_tokens_and_cost,
    summarize_telemetry,
)
from src.models.records import Document  # noqa: E402
from src.pipeline.factory import build_strategies_from_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_DOC_ID_TAG_RE = re.compile(r"\bdoc_id=([^\s\]]+)")
_SECTION_TAG_RE = re.compile(r"\bSection:\s*([^\s\|\]]+)")
_CHUNK_ID_TAG_RE = re.compile(r"\bchunk_id=([^\s\|\]]+)")
_DOC_ID_PROSE_RE = re.compile(
    r"\b(?:Document\s*ID|Doc\s*ID|Document)\s*[:#-]?\s*([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)
_BRACKET_ID_RE = re.compile(r"\[([^\[\]\s]+)\]")
_CHUNK_REF_RE = re.compile(r"\bchunk\s*#?\s*(\d+)\b", re.IGNORECASE)
_CHILD_ID_RE = re.compile(r"^(.+?):child:(\d+):(\d+)$", re.IGNORECASE)
_PARENT_ID_RE = re.compile(r"^(.+?):parent:(\d+)$", re.IGNORECASE)


def _get_max_tokens_cap() -> int:
  return min(1024, max(1, int(os.getenv("LLM_MAX_TOKENS", "512"))))


def _init_langfuse_handler() -> tuple[Any, Any, Any]:
  """Initializes Langfuse client, propagate_attributes helper, and LangChain CallbackHandler."""
  if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
    return None, None, None
  try:
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = get_client()
    handler = CallbackHandler()
    logger.info("Langfuse tracing enabled.")
    return client, propagate_attributes, handler
  except Exception as exc:
    logger.warning("Langfuse tracing disabled due to init error: %s", exc)
    return None, None, None


def _clip_str(val: Any, max_len: int = 190) -> str:
  s = str(val or "").strip()
  return s[:max_len] if len(s) > max_len else s


def _normalize_chunk_id(
    raw_id: Any,
    fallback_doc_id: str = "",
    default_parent_idx: int = 0,
    default_child_idx: int = 0,
) -> str:
  """Normalizes chunk IDs into '<doc_id>:child:<parent_idx>:<child_idx>' format."""
  s = str(raw_id or "").strip()
  doc_fallback = str(fallback_doc_id or "").strip()

  if s:
    m_child = _CHILD_ID_RE.match(s)
    if m_child:
      doc_part, p_idx, c_idx = m_child.groups()
      return f"{doc_part.strip()}:child:{int(p_idx)}:{int(c_idx)}"

    m_parent = _PARENT_ID_RE.match(s)
    if m_parent:
      doc_part, p_idx = m_parent.groups()
      return f"{doc_part.strip()}:child:{int(p_idx)}:{default_child_idx}"

  target_doc = doc_fallback or s
  if target_doc:
    return f"{target_doc}:child:{default_parent_idx}:{default_child_idx}"
  return ""


def _build_ragas_judge_and_embeddings() -> tuple[Any, Any]:
  """Builds RAGAS LLM (using project LLMClient) + local HuggingFaceEmbeddings."""

  class _ProjectLLMClientChatModel(BaseChatModel):
    client: Any = None
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
      return "project-llm-client"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
      formatted: list[dict[str, str]] = []
      for m in messages:
        role = "user"
        if m.type == "system":
          role = "system"
        elif m.type == "ai":
          role = "assistant"
        content = (
            m.content if isinstance(m.content, str) else json.dumps(m.content)
        )
        formatted.append({"role": role, "content": content})

      text = self.client.generate(
          messages=formatted,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
      ).strip()
      return ChatResult(
          generations=[ChatGeneration(message=AIMessage(content=text))]
      )

  embed_model_name = os.getenv(
      "RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
  )
  hf_embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
  ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

  judge_chat = _ProjectLLMClientChatModel(
      client=LLMClient(),
      temperature=0.0,
      max_tokens=min(2048, max(512, int(os.getenv("RAGAS_MAX_TOKENS", "1024")))),
  )
  ragas_llm = LangchainLLMWrapper(judge_chat)
  return ragas_llm, ragas_embeddings


def _build_model_callable_from_client(client: LLMClient | None = None) -> Any:
  llm = client or LLMClient()

  def _call_model(prompt: str) -> dict[str, Any]:
    max_tokens = _get_max_tokens_cap()
    raw_text = llm.generate(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    ).strip()

    parsed: dict[str, Any] | None = None
    candidate = raw_text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced:
      candidate = fenced.group(1)
    if candidate.startswith("{") and candidate.endswith("}"):
      try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
          parsed = obj
      except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
      ans = str(parsed.get("answer") or parsed.get("content") or raw_text).strip()
      cits = parsed.get("citations")
      if not isinstance(cits, list):
        cits = []
      return {
          "answer": ans,
          "citations": cits,
          "should_refuse": bool(parsed.get("should_refuse", False)),
          "reason": str(parsed.get("reason") or ""),
      }

    return {
        "answer": raw_text,
        "citations": [],
        "should_refuse": False,
        "reason": "",
    }

  return _call_model


def load_documents(docs_path: str) -> list[Document]:
  path = Path(docs_path)
  if not path.exists():
    raise FileNotFoundError(f"Documents file not found: {docs_path}")
  with open(path, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)
  return [
      Document(
          doc_id=doc.get("doc_id"),
          content=doc.get("content", ""),
          qa_pairs=doc.get("qa_pairs", []),
          image_path=doc.get("image_path", None),
      )
      for doc in raw_docs
  ]


def load_labels(labels_path: str) -> list[dict[str, Any]]:
  path = Path(labels_path)
  if not path.exists():
    raise FileNotFoundError(f"Labels file not found: {labels_path}")
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def _extract_ground_truth(item: dict[str, Any]) -> str:
  for key in (
      "gold_answer",
      "ground_truth",
      "answer",
      "expected_answer",
      "reference",
  ):
    val = item.get(key)
    if isinstance(val, str) and val.strip():
      return val.strip()
  return ""


def _extract_relevant_chunk_ids(item: dict[str, Any]) -> list[str]:
  """Extracts and normalizes gold chunk IDs from 'relevant_chunk_id' or related keys."""
  doc_id = str(item.get("doc_id") or "").strip()
  raw_vals: list[Any] = []

  for key in (
      "relevant_chunk_id",
      "relevant_chunk_ids",
      "gold_chunk_id",
      "gold_chunk_ids",
      "chunk_id",
      "chunk_ids",
      "target_chunk_id",
      "evidence_chunk_id",
  ):
    val = item.get(key)
    if isinstance(val, list):
      raw_vals.extend(val)
    elif val is not None and str(val).strip():
      raw_vals.append(val)

  meta = item.get("metadata")
  if isinstance(meta, dict):
    for key in (
        "relevant_chunk_id",
        "relevant_chunk_ids",
        "gold_chunk_id",
        "chunk_id",
    ):
      val = meta.get(key)
      if isinstance(val, list):
        raw_vals.extend(val)
      elif val is not None and str(val).strip():
        raw_vals.append(val)

  normalized: list[str] = []
  seen: set[str] = set()
  for rv in raw_vals:
    norm = _normalize_chunk_id(rv, fallback_doc_id=doc_id)
    if norm and norm not in seen:
      seen.add(norm)
      normalized.append(norm)

  if normalized:
    return normalized

  fallback = _normalize_chunk_id(doc_id, fallback_doc_id=doc_id)
  return [fallback] if fallback else []


def _extract_doc_id_from_item(doc: Any, text: str) -> str:
  if isinstance(doc, dict):
    for k in ("doc_id", "source_id", "document_id"):
      if doc.get(k):
        return str(doc[k]).strip()
    meta = doc.get("metadata")
    if isinstance(meta, dict) and meta.get("doc_id"):
      return str(meta["doc_id"]).strip()
  elif doc is not None:
    for attr in ("doc_id", "source_id", "document_id"):
      val = getattr(doc, attr, None)
      if val:
        return str(val).strip()
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict) and meta.get("doc_id"):
      return str(meta["doc_id"]).strip()

  match = _DOC_ID_TAG_RE.search(text or "")
  return match.group(1).strip() if match else ""


def _extract_chunk_id_from_item(doc: Any, text: str, fallback_doc_id: str) -> str:
  raw_id = ""
  parent_idx = 0
  child_idx = 0

  if isinstance(doc, dict):
    parent_idx = int(doc.get("parent_idx") or 0)
    child_idx = int(doc.get("child_idx") or 0)
    for k in ("chunk_id", "section_id", "node_id", "id"):
      val = doc.get(k)
      if val is not None and str(val).strip():
        raw_id = str(val).strip()
        break
    if not raw_id:
      meta = doc.get("metadata")
      if isinstance(meta, dict):
        parent_idx = int(meta.get("parent_idx") or parent_idx)
        child_idx = int(meta.get("child_idx") or child_idx)
        for k in ("chunk_id", "section_id", "node_id", "section"):
          val = meta.get(k)
          if val is not None and str(val).strip():
            raw_id = str(val).strip()
            break
  elif doc is not None:
    parent_idx = int(getattr(doc, "parent_idx", 0) or 0)
    child_idx = int(getattr(doc, "child_idx", 0) or 0)
    for attr in ("chunk_id", "section_id", "node_id", "id"):
      val = getattr(doc, attr, None)
      if val is not None and str(val).strip():
        raw_id = str(val).strip()
        break
    if not raw_id:
      meta = getattr(doc, "metadata", None)
      if isinstance(meta, dict):
        parent_idx = int(meta.get("parent_idx") or parent_idx)
        child_idx = int(meta.get("child_idx") or child_idx)
        for k in ("chunk_id", "section_id", "node_id", "section"):
          val = meta.get(k)
          if val is not None and str(val).strip():
            raw_id = str(val).strip()
            break

  if not raw_id:
    for rx in (_CHUNK_ID_TAG_RE, _SECTION_TAG_RE):
      m = rx.search(text or "")
      if m:
        raw_id = m.group(1).strip()
        break

  return _normalize_chunk_id(
      raw_id,
      fallback_doc_id=fallback_doc_id,
      default_parent_idx=parent_idx,
      default_child_idx=child_idx,
  )


def _extract_rank_from_item(doc: Any, fallback_rank: int) -> int:
  if isinstance(doc, dict):
    for k in ("rerank_rank", "rank", "position"):
      val = doc.get(k)
      if isinstance(val, int) and val > 0:
        return val
      if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip())
    meta = doc.get("metadata")
    if isinstance(meta, dict):
      for k in ("rerank_rank", "rank"):
        val = meta.get(k)
        if isinstance(val, int) and val > 0:
          return val
  elif doc is not None:
    for attr in ("rerank_rank", "rank"):
      val = getattr(doc, attr, None)
      if isinstance(val, int) and val > 0:
        return val
  return fallback_rank


def _extract_contexts_and_chunk_meta(
    output: Any,
    max_docs: int = 5,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
  if not isinstance(output, dict):
    return [], [], []
  final_out = (
      output.get("final_output")
      if isinstance(output.get("final_output"), dict)
      else output
  )
  docs = (
      final_out.get("reranked_docs")
      or final_out.get("evidence")
      or final_out.get("candidates")
      or []
  )
  contexts: list[str] = []
  doc_ids: list[str] = []
  chunk_entries: list[dict[str, Any]] = []

  for pos, doc in enumerate(docs[:max_docs], start=1):
    if isinstance(doc, dict):
      txt = doc.get("text") or doc.get("page_content") or ""
    else:
      txt = (
          getattr(doc, "text", None)
          or getattr(doc, "page_content", None)
          or ""
      )
    txt_str = str(txt).strip()
    if txt_str:
      d_id = _extract_doc_id_from_item(doc, txt_str)
      c_id = _extract_chunk_id_from_item(doc, txt_str, fallback_doc_id=d_id)
      rank = _extract_rank_from_item(doc, fallback_rank=pos)
      contexts.append(txt_str)
      doc_ids.append(d_id)
      chunk_entries.append({
          "rank": rank,
          "chunk_id": c_id,
          "doc_id": d_id,
      })

  return contexts, doc_ids, chunk_entries


def _extract_cited_chunk_ids(
    raw_resp: Any,
    answer: str,
    chunk_entries: list[dict[str, Any]] | None = None,
) -> list[str]:
  cited: list[str] = []
  seen: set[str] = set()

  rank_to_chunk: dict[int, str] = {}
  doc_to_first_chunk: dict[str, str] = {}
  for entry in chunk_entries or []:
    r_val = entry.get("rank")
    c_val = str(entry.get("chunk_id") or "").strip()
    d_val = str(entry.get("doc_id") or "").strip()
    if isinstance(r_val, int) and c_val:
      rank_to_chunk[r_val] = c_val
    if d_val and c_val and d_val not in doc_to_first_chunk:
      doc_to_first_chunk[d_val] = c_val

  def _add_chunk(cid: str) -> None:
    c = cid.strip()
    if c and c not in seen:
      seen.add(c)
      cited.append(c)

  def _add_str(raw: str, fallback_doc_id: str = "") -> None:
    s = raw.strip().strip(",.;:")
    if not s:
      return
    if s in doc_to_first_chunk:
      _add_chunk(doc_to_first_chunk[s])
      return
    norm = _normalize_chunk_id(s, fallback_doc_id=fallback_doc_id)
    if norm:
      _add_chunk(norm)

  def _add(val: Any) -> None:
    if isinstance(val, dict):
      doc_part = str(val.get("doc_id") or "").strip()
      for k in ("chunk_id", "section_id", "id", "doc_id"):
        if val.get(k):
          _add_str(str(val[k]), fallback_doc_id=doc_part)
      return
    _add_str(str(val or ""), fallback_doc_id="")

  if isinstance(raw_resp, dict):
    final_out = (
        raw_resp.get("final_output")
        if isinstance(raw_resp.get("final_output"), dict)
        else raw_resp
    )
    for cits_key in (
        "citations",
        "cited_chunks",
        "cited_doc_ids",
        "sources",
        "references",
    ):
      cits = final_out.get(cits_key)
      if isinstance(cits, list):
        for c in cits:
          _add(c)

  for rx in (
      _CHUNK_ID_TAG_RE,
      _SECTION_TAG_RE,
      _DOC_ID_TAG_RE,
      _DOC_ID_PROSE_RE,
      _BRACKET_ID_RE,
  ):
    for m in rx.finditer(answer or ""):
      _add_str(m.group(1), fallback_doc_id="")

  for m in _CHUNK_REF_RE.finditer(answer or ""):
    ref_rank = int(m.group(1))
    if ref_rank in rank_to_chunk:
      _add_chunk(rank_to_chunk[ref_rank])

  return cited


def _extract_answer_from_output(output: Any) -> tuple[str, bool]:
  if not isinstance(output, dict):
    return str(output or "").strip(), False
  final_out = (
      output.get("final_output")
      if isinstance(output.get("final_output"), dict)
      else output
  )
  answer = str(
      final_out.get("answer")
      or final_out.get("draft_answer")
      or output.get("draft_answer")
      or ""
  ).strip()
  should_refuse = bool(
      final_out.get("should_refuse", output.get("should_refuse", False))
  )
  reason = str(
      final_out.get("refusal_reason", output.get("refusal_reason", "")) or ""
  ).strip()
  if should_refuse and not answer:
    answer = f"Insufficient information in the retrieved context: {reason}".strip()
  return answer, should_refuse


def _compute_retrieval_metrics(
    chunk_entries: list[dict[str, Any]],
    retrieved_doc_ids: list[str],
    relevant_chunk_ids: list[str],
    expected_doc_id: Any,
) -> tuple[float, float]:
  """Computes (recall_at_k, reciprocal_rank) against relevant_chunk_id first, else doc_id."""
  if relevant_chunk_ids:
    expected_set = {str(x).strip() for x in relevant_chunk_ids if str(x).strip()}
    candidates = [
        _normalize_chunk_id(
            c.get("chunk_id"),
            fallback_doc_id=str(c.get("doc_id") or ""),
        )
        for c in chunk_entries
    ]
    candidates = [c for c in candidates if c]
  else:
    if isinstance(expected_doc_id, (list, tuple, set)):
      expected_set = {str(x).strip() for x in expected_doc_id if str(x).strip()}
    else:
      single = str(expected_doc_id or "").strip()
      expected_set = {single} if single else set()
    candidates = [str(d).strip() for d in retrieved_doc_ids if str(d).strip()]

  if not expected_set or not candidates:
    return 0.0, 0.0

  matched_ranks = [
      rank for rank, cid in enumerate(candidates, start=1) if cid in expected_set
  ]
  if not matched_ranks:
    return 0.0, 0.0

  found_set = {cid for cid in candidates if cid in expected_set}
  recall = round(len(found_set) / len(expected_set), 4)
  mrr = round(1.0 / matched_ranks[0], 4)
  return recall, mrr


def _invoke_pipeline(
    strategies: Any,
    question: str,
    documents: list[Document],
    max_docs: int = 5,
    langfuse_handler: Any = None,
    propagate_fn: Any = None,
    trace_attrs: dict[str, Any] | None = None,
) -> tuple[
    str, list[str], list[str], list[dict[str, Any]], list[str], bool, Any, str | None
]:
  """Runs compiled LangGraph if present; otherwise falls back to RAG + Prompt + Answerer."""
  trace_id: str | None = None
  if getattr(strategies, "agent_graph", None) is not None:
    invoke_kwargs: dict[str, Any] = {}
    if langfuse_handler is not None:
      invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}
    prop_ctx = (
        propagate_fn(**trace_attrs)
        if (propagate_fn is not None and trace_attrs)
        else nullcontext()
    )
    with prop_ctx:
      raw_out = strategies.agent_graph.invoke(
          {"question": question}, **invoke_kwargs
      )
    if langfuse_handler is not None:
      trace_id = getattr(langfuse_handler, "last_trace_id", None)
    answer, refused = _extract_answer_from_output(raw_out)
    contexts, doc_ids, chunk_entries = _extract_contexts_and_chunk_meta(
        raw_out, max_docs=max_docs
    )
    cited_ids = _extract_cited_chunk_ids(
        raw_out, answer, chunk_entries=chunk_entries
    )
    return (
        answer,
        contexts,
        doc_ids,
        chunk_entries,
        cited_ids,
        refused,
        raw_out,
        trace_id,
    )

  rag = strategies.rag
  prompt = strategies.prompt
  answerer = strategies.answerer

  evidence = []
  if hasattr(rag, "retrieve"):
    evidence = rag.retrieve(question, documents=documents, limit=max_docs)
  elif hasattr(rag, "search"):
    evidence = rag.search(query=question, documents=documents, limit=max_docs)

  contexts: list[str] = []
  doc_ids: list[str] = []
  chunk_entries: list[dict[str, Any]] = []
  for pos, item in enumerate((evidence or [])[:max_docs], start=1):
    txt = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
    txt_str = str(txt or "").strip()
    if txt_str:
      d_id = _extract_doc_id_from_item(item, txt_str)
      c_id = _extract_chunk_id_from_item(item, txt_str, fallback_doc_id=d_id)
      rank = _extract_rank_from_item(item, fallback_rank=pos)
      contexts.append(txt_str)
      doc_ids.append(d_id)
      chunk_entries.append({
          "rank": rank,
          "chunk_id": c_id,
          "doc_id": d_id,
      })

  raw_resp = answerer.answer(
      question=question,
      evidence=evidence[:max_docs] if isinstance(evidence, list) else [],
      prompt=prompt,
      documents=documents,
  )
  if isinstance(raw_resp, dict):
    answer = str(
        raw_resp.get("answer") or raw_resp.get("draft_answer") or ""
    ).strip()
    refused = bool(raw_resp.get("should_refuse", False))
  else:
    answer = str(raw_resp or "").strip()
    refused = False

  cited_ids = _extract_cited_chunk_ids(
      raw_resp, answer, chunk_entries=chunk_entries
  )
  return (
      answer,
      contexts,
      doc_ids,
      chunk_entries,
      cited_ids,
      refused,
      raw_resp,
      None,
  )


def _log_langfuse_row_scores(
    langfuse_client: Any,
    row: dict[str, Any],
    max_docs: int = 5,
) -> None:
  """Logs retrieval and RAGAS scores onto the trace in Langfuse when a trace_id is present."""
  trace_id = row.get("langfuse_trace_id")
  if langfuse_client is None or not trace_id:
    return
  ragas_scores = row.get("ragas") or {}
  metric_map = {
      f"recall_at_{max_docs}": row.get("recall"),
      "mrr": row.get("mrr"),
      "faithfulness": ragas_scores.get("faithfulness"),
      "answer_relevancy": ragas_scores.get("answer_relevancy"),
      "context_precision": ragas_scores.get("context_precision"),
  }
  for name, val in metric_map.items():
    if val is None:
      continue
    try:
      langfuse_client.create_score(
          trace_id=str(trace_id),
          name=name,
          value=float(val),
      )
    except Exception as exc:
      logger.debug("Failed to log Langfuse score %s: %s", name, exc)


def _compute_ragas_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """Computes RAGAS faithfulness, answer_relevancy, and context_precision; attaches per-row scores."""
  try:
    valid_indices = [
        idx
        for idx, r in enumerate(rows)
        if r.get("question") and r.get("answer") and r.get("contexts")
    ]
    if not valid_indices:
      return {
          "status": "skipped_no_valid_rows",
          "evaluated_count": 0,
          "scores": {},
      }

    valid_rows = [rows[idx] for idx in valid_indices]
    ds_payload = {
        "question": [r["question"] for r in valid_rows],
        "answer": [r["answer"] for r in valid_rows],
        "contexts": [r["contexts"] for r in valid_rows],
        "ground_truth": [r.get("ground_truth") or r["answer"] for r in valid_rows],
    }
    hf_ds = Dataset.from_dict(ds_payload)

    ragas_llm, ragas_embeddings = _build_ragas_judge_and_embeddings()
    ragas_res = evaluate(
        hf_ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=5),
    )
    df = ragas_res.to_pandas()

    metric_cols = ("faithfulness", "answer_relevancy", "context_precision")
    for row_pos, orig_idx in enumerate(valid_indices):
      row_scores: dict[str, float | None] = {}
      for col in metric_cols:
        if col in df.columns and row_pos < len(df):
          val = df.iloc[row_pos][col]
          row_scores[col] = (
              round(float(val), 4) if val is not None and val == val else None
          )
        else:
          row_scores[col] = None
      rows[orig_idx]["ragas"] = row_scores

    summary: dict[str, float] = {}
    for col in metric_cols:
      if col in df.columns:
        series = df[col].dropna()
        if len(series) > 0:
          summary[col] = round(float(series.mean()), 4)

    return {
        "status": "ok",
        "evaluated_count": len(valid_rows),
        "scores": summary,
    }
  except Exception as exc:
    logger.warning("RAGAS evaluation skipped/failed: %s", exc, exc_info=True)
    return {
        "status": "error",
        "error": str(exc),
        "evaluated_count": 0,
        "scores": {},
    }


def write_tabular_report_txt(
    txt_path: str,
    rows: list[dict[str, Any]],
    telemetry_summary: dict[str, Any],
    ragas_summary: dict[str, Any],
    retrieval_summary: dict[str, Any],
    max_tokens_cap: int,
) -> None:
  """Writes a tabular summary report plus compact per-question chunk IDs, ranks & citations."""
  out_txt = Path(txt_path)
  out_txt.parent.mkdir(parents=True, exist_ok=True)

  headers = [
      "Q#",
      "Q_ID",
      "Doc_ID",
      "MaxTok",
      "InTok",
      "OutTok",
      "Cost($)",
      "Lat(ms)",
      "Recall",
      "MRR",
      "Faith",
      "AnsRel",
      "CtxPrec",
      "Flags",
      "Question",
  ]

  table_rows: list[list[Any]] = []
  for r in rows:
    tel = r.get("telemetry") or {}
    lat = tel.get("latencies_ms") or {}
    ragas_row = r.get("ragas") or {}
    flags = [
        ev.get("rule", "")
        for ev in (tel.get("guardrail_events") or [])
        if ev.get("action") in ("block", "abstain", "flag")
    ]
    q_clean = " ".join(str(r.get("question") or "").split())
    if len(q_clean) > 48:
      q_clean = q_clean[:45] + "..."

    def _fmt_metric(val: Any, decimals: int = 4) -> str:
      if val is None or val == "":
        return "N/A"
      try:
        return f"{float(val):.{decimals}f}"
      except (TypeError, ValueError):
        return "N/A"

    table_rows.append([
        r.get("question_number", ""),
        str(r.get("question_id") or "")[:10],
        str(r.get("doc_id") or "")[:12],
        tel.get("max_tokens_configured") or max_tokens_cap,
        tel.get("prompt_tokens", 0),
        tel.get("completion_tokens", 0),
        f"{float(tel.get('estimated_cost_usd', 0.0)):.6f}",
        f"{float(lat.get('total_ms', 0.0)):.1f}",
        _fmt_metric(r.get("recall"), 2),
        _fmt_metric(r.get("mrr"), 4),
        _fmt_metric(ragas_row.get("faithfulness"), 4),
        _fmt_metric(ragas_row.get("answer_relevancy"), 4),
        _fmt_metric(ragas_row.get("context_precision"), 4),
        ",".join(flags) if flags else "-",
        q_clean,
    ])

  lines = [
      "E2E EVALUATION PER-QUESTION DIAGNOSTIC REPORT",
      "=" * 150,
      f"Total Queries : {len(rows)}",
      f"Retrieval     : {json.dumps(retrieval_summary)}",
      f"RAGAS         : {json.dumps(ragas_summary)}",
      f"Telemetry     : {json.dumps(telemetry_summary)}",
      "=" * 150,
      "",
  ]

  try:
    from tabulate import tabulate  # type: ignore

    lines.append(tabulate(table_rows, headers=headers, tablefmt="grid"))
  except ImportError:
    col_widths = [
        max(len(str(h)), *(len(str(row[i])) for row in table_rows))
        for i, h in enumerate(headers)
    ]

    def _fmt_line(cols: list[Any]) -> str:
      return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in col_widths)
    lines.extend([_fmt_line(headers), sep, *(_fmt_line(row) for row in table_rows)])

  lines.extend([
      "",
      "PER-QUESTION GOLD ANSWER, GOLD CHUNK ID, RETRIEVED CHUNK IDS & CITATIONS",
      "=" * 150,
  ])

  for r in rows:
    q_num = r.get("question_number", "")
    q_id = r.get("question_id", "")
    doc_id = str(r.get("doc_id") or "")
    rel_ids = r.get("relevant_chunk_ids") or []
    if rel_ids:
      gold_chunk_str = ", ".join(
          _normalize_chunk_id(x, fallback_doc_id=doc_id) for x in rel_ids
      )
    else:
      gold_chunk_str = (
          _normalize_chunk_id(
              r.get("gold_chunk_id") or doc_id, fallback_doc_id=doc_id
          )
          or "N/A"
      )

    gold_ans = " ".join(str(r.get("ground_truth") or "").split()) or "N/A"
    chunks = r.get("retrieved_chunks") or []
    if chunks:
      ret_str = ", ".join(
          f"#{c.get('rank')}:{_normalize_chunk_id(c.get('chunk_id'), fallback_doc_id=str(c.get('doc_id') or ''))}"
          for c in chunks
      )
    else:
      ret_str = (
          ", ".join(
              f"#{i}:{_normalize_chunk_id(c_id, fallback_doc_id=str(c_id or ''))}"
              for i, c_id in enumerate(r.get("retrieved_doc_ids") or [], start=1)
          )
          or "-"
      )

    cited = [
        _normalize_chunk_id(x, fallback_doc_id=doc_id)
        for x in (r.get("cited_chunk_ids") or [])
        if _normalize_chunk_id(x, fallback_doc_id=doc_id)
    ]
    cited_str = ", ".join(cited) if cited else "-"

    lines.append(
        f"Q#{q_num} ({q_id}) | GoldAns: {gold_ans} | GoldChunkID:"
        f" {gold_chunk_str} | Retrieved(Rank:ChunkID): [{ret_str}] |"
        f" CitedChunkIDs: [{cited_str}]"
    )

  with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")


def run_e2e_evaluation(
    docs_path: str,
    labels_path: str,
    output_path: str,
    checkpoint_path: str,
    report_txt_path: str | None = None,
    input_cost_per_1m_usd: float = 0.0,
    output_cost_per_1m_usd: float = 0.0,
    max_docs: int = 5,
    skip_ragas: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
  documents = load_documents(docs_path)
  labels = load_labels(labels_path)
  if limit is not None and limit > 0:
    labels = labels[:limit]

  max_tokens_cap = _get_max_tokens_cap()
  langfuse_client, propagate_fn, langfuse_handler = _init_langfuse_handler()

  base_config = AppConfig().from_env()
  model_callable = (
      _build_model_callable_from_client()
      if getattr(base_config, "answer_strategy", "") == "llm"
      else None
  )
  strategies = build_strategies_from_config(
      config=base_config,
      model_callable=model_callable,
      documents=documents,
  )
  guardrails = GuardrailEngine()

  ckpt_file = Path(checkpoint_path)
  ckpt_file.parent.mkdir(parents=True, exist_ok=True)

  completed_by_idx: dict[int, dict[str, Any]] = {}
  if ckpt_file.exists():
    with open(ckpt_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        rec = json.loads(line)
        completed_by_idx[int(rec["question_number"])] = rec
    logger.info(
        "Resumed %d completed rows from checkpoint %s",
        len(completed_by_idx),
        ckpt_file,
    )

  telemetry_records: list[QueryTelemetry] = []
  all_rows: list[dict[str, Any]] = []

  try:
    with open(ckpt_file, "a", encoding="utf-8") as ckpt_fp:
      for idx, item in enumerate(labels, start=1):
        question = str(item.get("question") or "").strip()
        if not question:
          continue

        expected_doc_id = item.get("doc_id") or item.get("relevant_doc_ids")
        relevant_chunk_ids = _extract_relevant_chunk_ids(item)
        gold_chunk_id = relevant_chunk_ids[0] if relevant_chunk_ids else ""
        ground_truth_val = _extract_ground_truth(item)

        if idx in completed_by_idx:
          row = dict(completed_by_idx[idx])
          doc_ids = list(row.get("retrieved_doc_ids") or [])
          contexts = list(row.get("contexts") or [])
          if not doc_ids and contexts:
            doc_ids = [_extract_doc_id_from_item(None, c) for c in contexts]
            row["retrieved_doc_ids"] = doc_ids

          chunk_entries = list(row.get("retrieved_chunks") or [])
          if not chunk_entries and contexts:
            for pos, ctx in enumerate(contexts, start=1):
              d_id = doc_ids[pos - 1] if pos - 1 < len(doc_ids) else ""
              c_id = _extract_chunk_id_from_item(None, ctx, fallback_doc_id=d_id)
              chunk_entries.append(
                  {"rank": pos, "chunk_id": c_id, "doc_id": d_id}
              )
          else:
            for pos, c_entry in enumerate(chunk_entries, start=1):
              d_id = str(
                  c_entry.get("doc_id")
                  or (doc_ids[pos - 1] if pos - 1 < len(doc_ids) else "")
              )
              c_entry["rank"] = int(c_entry.get("rank") or pos)
              c_entry["doc_id"] = d_id
              c_entry["chunk_id"] = _normalize_chunk_id(
                  c_entry.get("chunk_id"), fallback_doc_id=d_id
              )
          row["retrieved_chunks"] = chunk_entries

          if ground_truth_val:
            row["ground_truth"] = ground_truth_val
          row["relevant_chunk_ids"] = relevant_chunk_ids
          row["gold_chunk_id"] = gold_chunk_id

          recomputed_cited = _extract_cited_chunk_ids(
              None,
              str(row.get("answer") or ""),
              chunk_entries=chunk_entries,
          )
          row["cited_chunk_ids"] = recomputed_cited

          recall, mrr = _compute_retrieval_metrics(
              chunk_entries=chunk_entries,
              retrieved_doc_ids=doc_ids,
              relevant_chunk_ids=relevant_chunk_ids,
              expected_doc_id=expected_doc_id,
          )
          row["recall"] = recall
          row["mrr"] = mrr
          row.setdefault(
              "ragas",
              {
                  "faithfulness": None,
                  "answer_relevancy": None,
                  "context_precision": None,
              },
          )
          row.setdefault("langfuse_trace_id", None)

          all_rows.append(row)
          t_dict = row.get("telemetry") or {}
          telemetry_records.append(
              QueryTelemetry(
                  query_id=str(row.get("question_id") or idx),
                  question=question,
                  latencies_ms=t_dict.get("latencies_ms", {}),
                  max_tokens_configured=int(
                      t_dict.get("max_tokens_configured") or max_tokens_cap
                  ),
                  prompt_tokens=int(t_dict.get("prompt_tokens", 0)),
                  completion_tokens=int(t_dict.get("completion_tokens", 0)),
                  total_tokens=int(t_dict.get("total_tokens", 0)),
                  token_source=str(t_dict.get("token_source", "none")),
                  estimated_cost_usd=float(
                      t_dict.get("estimated_cost_usd", 0.0)
                  ),
                  abstained=bool(row.get("abstained", False)),
                  blocked=bool(row.get("blocked", False)),
              )
          )
          continue

        if idx == 1 or idx % 10 == 0:
          logger.info("Running E2E query %d/%d", idx, len(labels))

        q_id_str = _clip_str(item.get("question_id") or f"q_{idx}")
        trace_attrs = {
            "session_id": q_id_str,
            "tags": ["e2e_eval", "financial_rag"],
            "metadata": {
                "question_id": q_id_str,
                "doc_id": _clip_str(item.get("doc_id") or ""),
                "gold_chunk_id": _clip_str(gold_chunk_id),
            },
        }

        tel = QueryTelemetry(
            query_id=str(item.get("question_id") or idx),
            question=question,
            max_tokens_configured=max_tokens_cap,
        )
        t0 = time.perf_counter()
        answer = ""
        contexts: list[str] = []
        retrieved_doc_ids: list[str] = []
        chunk_entries: list[dict[str, Any]] = []
        cited_chunk_ids: list[str] = []
        raw_resp: Any = None
        trace_id: str | None = None

        with StageTimer(tel.latencies_ms, "guardrail_input_ms"):
          in_events = guardrails.check_input(question)
          tel.guardrail_events.extend(in_events)

        if any(e.action == "block" for e in in_events):
          tel.blocked = True
          answer = "Request blocked by input guardrail."
        else:
          with StageTimer(tel.latencies_ms, "pipeline_ms"):
            (
                answer,
                contexts,
                retrieved_doc_ids,
                chunk_entries,
                cited_chunk_ids,
                refused,
                raw_resp,
                trace_id,
            ) = _invoke_pipeline(
                strategies=strategies,
                question=question,
                documents=documents,
                max_docs=max_docs,
                langfuse_handler=langfuse_handler,
                propagate_fn=propagate_fn,
                trace_attrs=trace_attrs,
            )

          with StageTimer(tel.latencies_ms, "guardrail_context_ms"):
            ctx_events = guardrails.check_context(contexts)
            tel.guardrail_events.extend(ctx_events)

          with StageTimer(tel.latencies_ms, "guardrail_output_ms"):
            out_events, out_abstained = guardrails.check_output(answer, contexts)
            tel.guardrail_events.extend(out_events)
            tel.abstained = bool(refused or out_abstained)

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

        recall, mrr = _compute_retrieval_metrics(
            chunk_entries=chunk_entries,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            expected_doc_id=expected_doc_id,
        )

        telemetry_records.append(tel)
        row = {
            "question_number": idx,
            "question_id": item.get("question_id"),
            "doc_id": item.get("doc_id"),
            "gold_chunk_id": gold_chunk_id,
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_chunks": chunk_entries,
            "cited_chunk_ids": cited_chunk_ids,
            "question": question,
            "ground_truth": ground_truth_val,
            "answer": answer,
            "contexts": contexts,
            "recall": recall,
            "mrr": mrr,
            "ragas": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            },
            "langfuse_trace_id": trace_id,
            "abstained": tel.abstained,
            "blocked": tel.blocked,
            "telemetry": {
                "latencies_ms": tel.latencies_ms,
                "max_tokens_configured": tel.max_tokens_configured,
                "prompt_tokens": tel.prompt_tokens,
                "completion_tokens": tel.completion_tokens,
                "total_tokens": tel.total_tokens,
                "token_source": tel.token_source,
                "estimated_cost_usd": tel.estimated_cost_usd,
                "guardrail_events": [asdict(ev) for ev in tel.guardrail_events],
            },
        }
        all_rows.append(row)
        ckpt_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        ckpt_fp.flush()

    telemetry_summary = summarize_telemetry(telemetry_records)
    ragas_summary = (
        {"status": "skipped_by_flag", "scores": {}}
        if skip_ragas
        else _compute_ragas_scores(all_rows)
    )

    if langfuse_client is not None:
      for r in all_rows:
        _log_langfuse_row_scores(langfuse_client, r, max_docs=max_docs)

    n_rows = len(all_rows) or 1
    retrieval_summary = {
        "k": max_docs,
        "evaluated_count": len(all_rows),
        "mean_recall": round(
            sum(float(r.get("recall", 0.0)) for r in all_rows) / n_rows, 4
        ),
        "mean_mrr": round(
            sum(float(r.get("mrr", 0.0)) for r in all_rows) / n_rows, 4
        ),
    }

    report = {
        "config": {
            "search_strategy": getattr(base_config, "search_strategy", None),
            "chunking_strategy": getattr(base_config, "chunking_strategy", None),
            "retrieval_limit": getattr(base_config, "retrieval_limit", None),
            "max_docs_to_generator": max_docs,
            "max_tokens_configured": max_tokens_cap,
        },
        "retrieval_summary": retrieval_summary,
        "telemetry_summary": telemetry_summary,
        "ragas_summary": ragas_summary,
        "results": all_rows,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, ensure_ascii=False)

    txt_file = (
        Path(report_txt_path) if report_txt_path else out_file.with_suffix(".txt")
    )
    write_tabular_report_txt(
        txt_path=str(txt_file),
        rows=all_rows,
        telemetry_summary=telemetry_summary,
        ragas_summary=ragas_summary,
        retrieval_summary=retrieval_summary,
        max_tokens_cap=max_tokens_cap,
    )

    print("\n=== RETRIEVAL SUMMARY (RECALL & MRR) ===")
    print(json.dumps(retrieval_summary, indent=2))
    print("\n=== E2E TELEMETRY & GUARDRAILS SUMMARY ===")
    print(json.dumps(telemetry_summary, indent=2))
    print("\n=== RAGAS SUMMARY ===")
    print(json.dumps(ragas_summary, indent=2))
    logger.info(
        "Saved full E2E JSON report to %s and tabular TXT report to %s",
        out_file,
        txt_file,
    )
    return report
  finally:
    if langfuse_client is not None:
      try:
        langfuse_client.flush()
      except Exception as exc:
        logger.warning("Langfuse flush failed: %s", exc)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Run E2E RAG evaluation with guardrails, telemetry & RAGAS."
      )
  )
  parser.add_argument(
      "--input", required=True, help="Path to documents JSON file."
  )
  parser.add_argument(
      "--labels", required=True, help="Path to gold set / labels JSON file."
  )
  parser.add_argument(
      "--output", required=True, help="Path to output JSON report."
  )
  parser.add_argument(
      "--report-txt",
      default=None,
      help="Optional path to tabular .txt report (defaults to <output>.txt).",
  )
  parser.add_argument(
      "--checkpoint",
      default="outputs/e2e_checkpoint.jsonl",
      help="Path to JSONL checkpoint file for crash/rate-limit resume.",
  )
  parser.add_argument(
      "--input-cost-per-1m",
      type=float,
      default=0.0,
      help="Generator input cost per 1M tokens (USD).",
  )
  parser.add_argument(
      "--output-cost-per-1m",
      type=float,
      default=0.0,
      help="Generator output cost per 1M tokens (USD).",
  )
  parser.add_argument(
      "--max-docs",
      type=int,
      default=5,
      help="Max retrieved contexts passed to generator/RAGAS.",
  )
  parser.add_argument(
      "--skip-ragas",
      action="store_true",
      help="Run telemetry + guardrails first without RAGAS.",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=None,
      help="Optional cap on number of rows to evaluate.",
  )
  args = parser.parse_args()

  run_e2e_evaluation(
      docs_path=args.input,
      labels_path=args.labels,
      output_path=args.output,
      checkpoint_path=args.checkpoint,
      report_txt_path=args.report_txt,
      input_cost_per_1m_usd=args.input_cost_per_1m,
      output_cost_per_1m_usd=args.output_cost_per_1m,
      max_docs=args.max_docs,
      skip_ragas=args.skip_ragas,
      limit=args.limit,
  )


if __name__ == "__main__":
  main()